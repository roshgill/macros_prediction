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
const weightSlider    = document.getElementById('weight-slider');
const sliderVal       = document.getElementById('slider-val');
const inferenceTime   = document.getElementById('inference-time');
const samplesRow      = document.getElementById('samples-row');
const bodyWeightInput = document.getElementById('body-weight');
const heightFtInput   = document.getElementById('height-ft');
const heightInInput   = document.getElementById('height-in');

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
  setupUploadZone();
  weightSlider.addEventListener('input', onSliderChange);
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
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
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

  resultImg.src = URL.createObjectURL(file);
  results.classList.remove('hidden');
  shimmer.classList.remove('hidden');
  macroData.classList.add('hidden');
  dishLabel.textContent = '';
  results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  await runPredict(file);
}

async function runPredict(file) {
  const servingG     = parseInt(weightSlider.value, 10);
  const weightLbs = parseFloat(bodyWeightInput.value) || 154;
  const heightFt  = parseFloat(heightFtInput.value)   || 6;
  const heightIn  = parseFloat(heightInInput.value)   || 0;
  const weightKg  = weightLbs * 0.453592;
  const heightCm  = (heightFt * 12 + heightIn) * 2.54;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('serving_g', servingG);
  fd.append('body_weight_kg', weightKg);
  fd.append('height_cm', heightCm);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) { shimmer.classList.add('hidden'); showError(data.detail || 'Prediction failed.'); return; }
    currentPrediction = data;
    renderResults(data, servingG);
  } catch {
    shimmer.classList.add('hidden');
    showError('Network error — is the server running?');
  }
}

function renderResults(data, weight) {
  shimmer.classList.add('hidden');
  macroData.classList.remove('hidden');

  // Per-100g
  document.getElementById('kcal-100g').textContent    = fmt(data.kcal_per_100g);
  document.getElementById('protein-100g').textContent = fmt(data.protein_g);
  document.getElementById('carbs-100g').textContent   = fmt(data.carb_g);
  document.getElementById('fat-100g').textContent     = fmt(data.fat_g);

  // Uncertainties
  document.getElementById('kcal-unc').textContent    = `±${fmt(data.uncertainty.kcal)} kcal`;
  document.getElementById('protein-unc').textContent = `±${fmt(data.uncertainty.protein)} g`;
  document.getElementById('carbs-unc').textContent   = `±${fmt(data.uncertainty.carb)} g`;
  document.getElementById('fat-unc').textContent     = `±${fmt(data.uncertainty.fat)} g`;

  updateServing(weight, data);

  // Dish label
  if (data.top_classes?.length > 0) {
    const { name, confidence } = data.top_classes[0];
    dishLabel.innerHTML = `Looks like: <strong>${name}</strong> <span style="color:var(--text-dim)">(${Math.round(confidence * 100)}%)</span>`;
  }

  inferenceTime.textContent = `Predicted in ${data.inference_ms} ms`;

  // Score panel
  if (data.score) renderScore(data.score, data);
}

// ── Slider ─────────────────────────────────────────────────────────────────────
function onSliderChange() {
  const w = parseInt(weightSlider.value, 10);
  sliderVal.textContent = w;
  if (currentPrediction) updateServing(w, currentPrediction);
}

function updateServing(weight, data) {
  const f = weight / 100;
  document.getElementById('kcal-srv').textContent    = fmt(data.kcal_per_100g * f);
  document.getElementById('protein-srv').textContent = fmt(data.protein_g     * f);
  document.getElementById('carbs-srv').textContent   = fmt(data.carb_g        * f);
  document.getElementById('fat-srv').textContent     = fmt(data.fat_g         * f);
}

// ── Score panel ────────────────────────────────────────────────────────────────
function renderScore(score, data = {}) {
  document.getElementById('score-number').textContent = score.overall;

  const verdictEl = document.getElementById('score-verdict');
  verdictEl.textContent = score.verdict;
  verdictEl.className = 'score-verdict ' + score.verdict.toLowerCase().replace(' ', '-');

  const sub = score.subscores;
  setBar('protein', sub.protein);
  setBar('ratio',   sub.macro_ratio);
  setBar('density', sub.calorie_density);
  setBar('size',    sub.portion_size);

  document.getElementById('score-advice').textContent = score.advice;

  if (data.targets) {
    const t = data.targets;
    document.getElementById('score-targets').textContent =
      `BSA ${t.user_bsa} m² · scale ${t.scale_factor}× · target ${Math.round(t.per_meal_protein_g)}g protein/meal`;
  }
}

function setBar(id, value) {
  const fill = document.getElementById(`bar-${id}`);
  const val  = document.getElementById(`val-${id}`);
  fill.style.width = `${Math.min(100, Math.max(0, value))}%`;
  fill.className = 'score-bar-fill ' + (value >= 75 ? 'high' : value >= 50 ? 'medium' : 'low');
  val.textContent = Math.round(value);
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

// ── Helpers ────────────────────────────────────────────────────────────────────
function fmt(val) {
  const n = parseFloat(val);
  return isNaN(n) ? '—' : n.toFixed(1);
}
