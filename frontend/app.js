/* Optimal — frontend logic */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────────
let currentFile       = null;
let currentPrediction = null;
let gradcamActive     = false;

// ── DOM refs ───────────────────────────────────────────────────────────────────
const dropZone        = document.getElementById('drop-zone');
const fileInput       = document.getElementById('file-input');
const results         = document.getElementById('results');
const shimmer         = document.getElementById('shimmer');
const macroData       = document.getElementById('macro-data');
const resultImg       = document.getElementById('result-img');
const gradcamOverlay  = document.getElementById('gradcam-overlay');
const gradcamImg      = document.getElementById('gradcam-img');
const gradcamBtn      = document.getElementById('gradcam-btn');
const gradcamBtnText  = document.getElementById('gradcam-btn-text');
const gradcamSpinner  = document.getElementById('gradcam-spinner');
const dishLabel       = document.getElementById('dish-label');
const inferenceTime   = document.getElementById('inference-time');
const samplesRow        = document.getElementById('samples-row');
const bodyWeightInput   = document.getElementById('body-weight');
const heightFtInput     = document.getElementById('height-ft');
const heightInInput     = document.getElementById('height-in');
const analysisPanel     = document.getElementById('analysis-panel');
const analysisLoading   = document.getElementById('analysis-loading');
const bpScore           = document.getElementById('bp-score');
const bpSuggestion      = document.getElementById('bp-suggestion');
const bpSources         = document.getElementById('bp-sources');
const bpSummary         = document.getElementById('bp-summary');
const tipsText          = document.getElementById('tips-text');

// ── Blueprint tips ─────────────────────────────────────────────────────────────
const TIPS = [
  "Don't eat within 3 hours of sleep — fasting overnight supports cellular repair.",
  "Target 25g of protein per meal. Spread it evenly; don't front-load.",
  "Eat the same meals daily to eliminate decision fatigue and master absorption.",
  "80% of your diet should come from whole plants. Diversity is the goal.",
  "Extra virgin olive oil every day — 30ml is the minimum effective dose.",
  "Limit your eating window to 6–8 hours. Time-restricted eating amplifies every other intervention.",
  "Caloric restriction is precision, not deprivation. Know your numbers exactly.",
  "Measure biomarkers quarterly — data beats intuition every single time.",
  "Dark leafy greens at every meal. Sulforaphane is your best ally.",
  "Avoid ultra-processed foods entirely. If it has more than 5 ingredients, reconsider.",
];

function startTips() {
  const el = document.getElementById('tips-text');
  if (!el) return;
  let idx = Math.floor(Math.random() * TIPS.length);
  el.textContent = TIPS[idx];
  setInterval(() => {
    el.classList.add('fade');
    setTimeout(() => {
      idx = (idx + 1) % TIPS.length;
      el.textContent = TIPS[idx];
      el.classList.remove('fade');
    }, 450);
  }, 6000);
}

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
  setupUploadZone();
  startTips();
});

// ── Sample images ──────────────────────────────────────────────────────────────
async function loadSamples() {
  try {
    const resp = await fetch('/samples');
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.samples || data.samples.length === 0) {
      samplesRow.innerHTML = '<p style="color:var(--text-dim);font-size:0.75rem;text-align:center;grid-column:1/-1">No samples found.</p>';
      return;
    }
    samplesRow.innerHTML = data.samples.map(url => `
      <img src="${url}" class="sample-img" alt="Sample food" loading="lazy"
           onclick="submitSampleUrl('${url}')" />
    `).join('');
  } catch {
    samplesRow.innerHTML = '';
  }
}

async function submitSampleUrl(url) {
  try {
    const resp = await fetch(url);
    const blob = await resp.blob();
    const file = new File([blob], url.split('/').pop(), { type: blob.type });
    await handleFile(file);
  } catch {
    showError('Could not load sample image.');
  }
}

// ── Upload zone ────────────────────────────────────────────────────────────────
function setupUploadZone() {
  const uploadBar    = document.getElementById('upload-bar');
  const metricsBar   = document.getElementById('metrics-bar');

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  // Metrics inputs must not trigger the file picker
  metricsBar.addEventListener('click', e => e.stopPropagation());

  uploadBar.addEventListener('dragover',  e => { e.preventDefault(); uploadBar.classList.add('drag-over'); });
  uploadBar.addEventListener('dragleave', e => { if (!uploadBar.contains(e.relatedTarget)) uploadBar.classList.remove('drag-over'); });
  uploadBar.addEventListener('drop', e => {
    e.preventDefault();
    uploadBar.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

// ── Core flow ──────────────────────────────────────────────────────────────────
async function handleFile(file) {
  clearError();
  if (!file.type.startsWith('image/')) { showError('Please upload an image file (JPEG, PNG, WEBP).'); return; }
  if (file.size > 10 * 1024 * 1024)   { showError('File too large — max 10 MB.'); return; }

  currentFile   = file;
  gradcamActive = false;
  gradcamOverlay.classList.add('hidden');
  gradcamBtn.classList.remove('active');
  gradcamBtnText.textContent = 'Why this prediction?';

  // Reset analysis panel
  analysisPanel.classList.add('hidden');
  analysisLoading.classList.add('hidden');

  resultImg.src = URL.createObjectURL(file);
  results.classList.remove('hidden');
  shimmer.classList.remove('hidden');
  macroData.classList.add('hidden');
  dishLabel.textContent = '';
  results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  await runPredict(file);
}

async function runPredict(file) {
  const weightLbs = parseFloat(bodyWeightInput.value) || 154;
  const heightFt  = parseFloat(heightFtInput.value)   || 6;
  const heightIn  = parseFloat(heightInInput.value)   || 0;
  const weightKg  = weightLbs * 0.453592;
  const heightCm  = (heightFt * 12 + heightIn) * 2.54;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('body_weight_kg', weightKg);
  fd.append('height_cm', heightCm);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) { shimmer.classList.add('hidden'); showError(data.detail || 'Prediction failed.'); return; }
    currentPrediction = data;
    renderResults(data);
    runAnalyze(data);
  } catch {
    shimmer.classList.add('hidden');
    showError('Network error — is the server running?');
  }
}

async function runAnalyze(data) {
  if (!data.top_classes?.length) return;

  analysisLoading.classList.remove('hidden');

  const weightLbs = parseFloat(bodyWeightInput.value) || 154;
  const heightFt  = parseFloat(heightFtInput.value)   || 6;
  const heightIn  = parseFloat(heightInInput.value)   || 0;
  const weightKg  = weightLbs * 0.453592;
  const heightCm  = (heightFt * 12 + heightIn) * 2.54;

  const fd = new FormData();
  fd.append('dish_name',     data.top_classes[0].name.replace(/_/g, ' '));
  fd.append('kcal_per_100g', data.kcal_per_100g);
  fd.append('protein_g',     data.protein_g);
  fd.append('carb_g',        data.carb_g);
  fd.append('fat_g',         data.fat_g);
  fd.append('body_weight_kg', weightKg);
  fd.append('height_cm',      heightCm);

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: fd });
    const result = await resp.json();
    analysisLoading.classList.add('hidden');
    if (resp.ok) {
      bpScore.textContent = result.score;
      bpScore.className = 'bp-score-num mono ' + scoreClass(result.score);
      bpSummary.textContent = result.summary || '';
      bpSuggestion.textContent = result.suggestion || '';
      bpSuggestion.style.display = result.suggestion ? '' : 'none';
      bpSources.innerHTML = (result.sources || []).map(s =>
        `<a href="${s.url || '#'}" target="_blank" rel="noopener">${s.title}</a>`
      ).join('');
      analysisPanel.classList.remove('hidden');
    }
  } catch {
    analysisLoading.classList.add('hidden');
  }
}

function scoreClass(score) {
  if (score >= 80) return 'score-high';
  if (score >= 60) return 'score-mid';
  if (score >= 40) return 'score-low';
  return 'score-off';
}

function renderResults(data) {
  shimmer.classList.add('hidden');
  macroData.classList.remove('hidden');

  if (data.top_classes?.length > 0) {
    const { name, confidence } = data.top_classes[0];
    dishLabel.innerHTML = `Looks like: <strong>${name.replace(/_/g, ' ')}</strong> <span style="color:var(--text-dim)">(${Math.round(confidence * 100)}%)</span>`;
  }

  inferenceTime.textContent = `Predicted in ${data.inference_ms} ms`;

  if (data.targets) {
    const t = data.targets;
    setProgress('kcal',    data.kcal_per_100g, t.per_meal_kcal,      'kcal');
    setProgress('protein', data.protein_g,     t.per_meal_protein_g, 'g');
    setProgress('carbs',   data.carb_g,        t.per_meal_carb_g,    'g');
    setProgress('fat',     data.fat_g,         t.per_meal_fat_g,     'g');

    document.getElementById('score-targets').textContent =
      `BSA ${t.user_bsa} m² · scale ${t.scale_factor}× · targets per day`;
  }
}

// ── Macro progress bars ────────────────────────────────────────────────────────
function setProgress(id, value, target, unit) {
  const pct = Math.min(120, (value / target) * 100);
  document.getElementById(`val-${id}`).textContent   = parseFloat(value).toFixed(1);
  document.getElementById(`target-${id}`).textContent = Math.round(target);
  const bar = document.getElementById(`bar-${id}`);
  bar.style.width = `${Math.min(100, pct)}%`;
  bar.className   = 'prog-fill ' + (pct >= 100 ? 'over' : pct >= 55 ? 'good' : pct >= 25 ? 'mid' : 'low');
}

// ── Grad-CAM ───────────────────────────────────────────────────────────────────
async function toggleGradcam() {
  if (gradcamActive) {
    gradcamOverlay.classList.add('hidden');
    gradcamBtn.classList.remove('active');
    gradcamBtnText.textContent = 'Why this prediction?';
    gradcamActive = false;
    return;
  }
  if (!currentFile) return;

  gradcamBtnText.textContent = 'Loading heatmap…';
  gradcamSpinner.classList.remove('hidden');
  gradcamBtn.disabled = true;

  const fd = new FormData();
  fd.append('file', currentFile);

  try {
    const resp = await fetch('/gradcam', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) { showError(data.detail || 'Grad-CAM failed.'); return; }
    gradcamImg.src = `data:image/png;base64,${data.heatmap}`;
    gradcamOverlay.classList.remove('hidden');
    gradcamBtn.classList.add('active');
    gradcamBtnText.textContent = 'Hide heatmap';
    gradcamActive = true;
  } catch {
    showError('Could not load Grad-CAM heatmap.');
    gradcamBtnText.textContent = 'Why this prediction?';
  } finally {
    gradcamSpinner.classList.add('hidden');
    gradcamBtn.disabled = false;
  }
}

// ── Error handling ─────────────────────────────────────────────────────────────
function showError(msg) {
  document.getElementById('error-text').textContent = msg;
  document.getElementById('error-banner').classList.remove('hidden');
}

function clearError() {
  document.getElementById('error-banner').classList.add('hidden');
}
