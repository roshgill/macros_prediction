"""Blueprint LLM analysis — GPT-4o with tool calls over Neon pgvector knowledge base."""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_db_url = os.environ["DATABASE_URL"]

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_blueprint_knowledge",
        "description": (
            "Search Bryan Johnson's Blueprint protocol knowledge base for dietary "
            "guidelines relevant to the identified food. Use this to find what Blueprint "
            "says about specific foods, nutrients, or dietary practices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific search query about food or nutrition to look up in the "
                        "Blueprint knowledge base. Be specific, e.g. 'are nuts and seeds "
                        "approved in Blueprint protocol' or 'Blueprint stance on processed meats'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

_FORMAT_TOOL = {
    "type": "function",
    "function": {
        "name": "format_response",
        "description": "Format the final Blueprint analysis as structured output.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Blueprint compliance score 0-100",
                },
                "foods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of identified food items",
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "One short sentence (max 15 words) on how well this meal aligns "
                        "with Blueprint. Never mention missing items or suggest additions."
                    ),
                },
                "suggestion": {
                    "type": "string",
                    "description": (
                        "Max ~50 chars. Only for clearly unhealthy meals (score < 50). "
                        "Must be empty string otherwise. Never suggest adding oil, "
                        "vegetables, or legumes."
                    ),
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["title", "url"],
                    },
                    "description": (
                        "Deduplicated list of sources referenced during analysis. "
                        "Include title and url from the search results."
                    ),
                },
            },
            "required": ["score", "foods", "summary", "sources"],
        },
    },
}

_SYSTEM_PROMPT = """You are a Blueprint protocol food analyst. Your users are biohackers who are trying their best — your job is to be their advocate, not their critic.

Workflow:
1. Identify food items you can clearly distinguish. If an item is ambiguous (e.g. unclear sauce, unidentifiable side), note the uncertainty — do NOT penalize it or include it in scoring.
2. Use search_blueprint_knowledge to research each identified food. Search thoroughly — especially for items like red meat, which Blueprint occasionally allows. Make multiple searches to find supporting evidence.
3. Score 0-100. Be generous. Only penalize clearly unhealthy items (fast food, fried food, processed junk, alcohol, sugary drinks). Whole foods with vegetables deserve high scores even if not perfectly Blueprint-canonical.
4. Summary: ONE short sentence (max 15 words) on how well this meal aligns with Blueprint. Never mention what's missing, never suggest additions.
5. Suggestion: Only for clearly problematic meals (score < 50) with obviously unhealthy items. Must be empty string otherwise. Never suggest adding oil, vegetables, or legumes.
6. Rescans: If a previous analysis is provided, the user is correcting you. If their correction describes a healthier item than what you assumed, raise the score accordingly.

After research, call format_response with your analysis. Include deduplicated sources with title and url from search results."""


# ---------------------------------------------------------------------------
# Embedding + vector search
# ---------------------------------------------------------------------------

def _embed(text: str) -> list[float]:
    """Embed text using text-embedding-3-small at 1024 dimensions to match DB."""
    resp = _client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=1024,
    )
    return resp.data[0].embedding


def search_blueprint_knowledge(query: str, limit: int = 5) -> str:
    """Cosine similarity search; returns formatted string for the LLM."""
    embedding = _embed(query)
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    sql = """
        SELECT c.content, s.title AS source_title, s.url AS source_url,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN sources s ON c.source_id = s.id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
    """
    with psycopg2.connect(_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vec_str, vec_str, limit))
            rows = cur.fetchall()

    return "\n\n---\n\n".join(
        f"[Source: {r[1]}] (url: {r[2] or 'none'}) (similarity: {float(r[3]):.3f})\n{r[0]}"
        for i, r in enumerate(rows)
    )


# ---------------------------------------------------------------------------
# Agentic analysis loop
# ---------------------------------------------------------------------------

def analyze_meal(
    dish_name: str,
    macros: dict,
    targets: dict,
    user_weight_kg: float,
    user_height_cm: float,
    user_note: str = "",
    previous_result: dict[str, Any] | None = None,
) -> dict:
    """
    Run GPT-4o with tool calls to score a meal against Blueprint protocol.

    Returns:
        {score, foods, summary, suggestion, sources}
    """
    text_parts = [
        f"Analyze this meal against the Blueprint protocol.\n\n"
        f"Dish: {dish_name}\n"
        f"Macros (per 100g): {macros['kcal_per_100g']:.1f} kcal, "
        f"{macros['protein_g']:.1f}g protein, {macros['carb_g']:.1f}g carbs, {macros['fat_g']:.1f}g fat\n"
        f"User: {user_weight_kg:.1f}kg, {user_height_cm:.0f}cm (BSA {targets['user_bsa']} m²)\n"
        f"Per-meal targets: {targets['per_meal_kcal']:.0f} kcal, "
        f"{targets['per_meal_protein_g']:.0f}g protein, "
        f"{targets['per_meal_carb_g']:.0f}g carbs, "
        f"{targets['per_meal_fat_g']:.0f}g fat"
    ]
    if user_note:
        text_parts.append(f"User note: {user_note}")
    if previous_result:
        text_parts.append(
            f"Previous analysis (user is re-scanning with updated description — adjust accordingly): "
            f"Score: {previous_result.get('score')}/100, "
            f"Foods: {', '.join(previous_result.get('foods', []))}, "
            f"Summary: {previous_result.get('summary', '')}"
        )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(text_parts)},
    ]

    MAX_ROUNDS = 5
    for round_num in range(MAX_ROUNDS):
        response = _client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[_SEARCH_TOOL],
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # No more tool calls — move to structured output
            break

        messages.append(msg)

        # Execute tool calls
        tool_results = []
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_blueprint_knowledge":
                result = search_blueprint_knowledge(args["query"])
            else:
                result = "Unknown tool"
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if round_num == MAX_ROUNDS - 1:
            # Bail-out: force final answer
            for tr in tool_results:
                tr["content"] = "Max search rounds reached. Provide your final answer now."

        messages.extend(tool_results)

    # Final structured call via format_response tool
    messages.append({"role": "assistant", "content": msg.content or ""})
    final = _client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[_FORMAT_TOOL],
        tool_choice={"type": "function", "function": {"name": "format_response"}},
        temperature=0.2,
    )

    tc = final.choices[0].message.tool_calls[0]
    parsed = json.loads(tc.function.arguments)

    # Increment scan stats (best-effort)
    try:
        with psycopg2.connect(_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE scan_stats SET count = count + 1 WHERE id = 1")
    except Exception:
        pass

    return {
        "score": int(parsed.get("score", 50)),
        "foods": parsed.get("foods", []),
        "summary": parsed.get("summary", ""),
        "suggestion": parsed.get("suggestion", ""),
        "sources": parsed.get("sources", []),
    }
