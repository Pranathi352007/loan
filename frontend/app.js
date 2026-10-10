const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const integerFields = new Set(["no_of_dependents", "loan_term", "cibil_score"]);
let featureFields = [];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}
function niceFeature(value) { return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase()); }
function percent(value) { const number = Number(value); return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—"; }
function numberFmt(value) { return Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 }) : "—"; }

function renderFormFields(container, prefix) {
  const html = featureFields.map((field) => {
    const id = `${prefix}-${field.name}`;
    if (field.kind === "select") {
      const options = (field.options || []).map((option) => `<option value="${escapeHtml(option)}" ${String(option) === String(field.default) ? "selected" : ""}>${escapeHtml(option)}</option>`).join("");
      return `<label class="field" for="${id}"><span>${escapeHtml(field.label)}</span><select id="${id}" name="${escapeHtml(field.name)}" required>${options}</select></label>`;
    }
    const min = field.min !== undefined ? `min="${field.min}"` : "";
    const max = field.max !== undefined ? `max="${field.max}"` : "";
    const step = field.step !== undefined ? `step="${field.step}"` : 'step="any"';
    const hint = field.name === "loan_amount" ? "Enter the requested principal, not the total repayment." : field.name === "cibil_score" ? "Use a score between 300 and 900." : "";
    return `<label class="field" for="${id}"><span>${escapeHtml(field.label)}</span><input id="${id}" name="${escapeHtml(field.name)}" type="number" ${min} ${max} ${step} value="${escapeHtml(field.default ?? "")}" required /><small>${escapeHtml(hint)}</small></label>`;
  }).join("");
  container.innerHTML = html;
}

function collectForm(form, prefix) {
  const data = {};
  for (const field of featureFields) {
    const element = form.elements[field.name];
    if (!element) throw new Error(`Missing form field: ${field.label}`);
    if (field.kind === "select") data[field.name] = element.value;
    else {
      const value = element.value.trim();
      if (value === "") throw new Error(`${field.label} is required.`);
      data[field.name] = integerFields.has(field.name) ? Number.parseInt(value, 10) : Number(value);
      if (!Number.isFinite(data[field.name])) throw new Error(`${field.label} must be a valid number.`);
    }
  }
  return data;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status}).`);
  return data;
}

function showError(target, error) {
  target.innerHTML = `<div class="error-box"><strong>Could not complete the request.</strong><br>${escapeHtml(error.message || error)}</div>`;
}
function setBusy(button, busy, busyText) {
  if (busy) { button.dataset.originalText = button.innerHTML; button.disabled = true; button.innerHTML = `${escapeHtml(busyText)}…`; }
  else { button.disabled = false; if (button.dataset.originalText) button.innerHTML = button.dataset.originalText; }
}

function renderPrediction(data) {
  const approved = data.is_approved;
  const approve = Number(data.approval_probability);
  const reject = Number(data.rejection_probability);
  return `<article class="result-card">
    <div class="result-header"><div><p class="eyebrow">PREDICTION RESULT</p><h3 class="result-title">${escapeHtml(data.predicted_status)} in this model</h3><p class="result-subtitle">This is the model output at its selected decision threshold.</p></div><span class="status-chip ${approved ? "approved" : "rejected"}">${approved ? "MODEL: APPROVED" : "MODEL: REJECTED"}</span></div>
    <div class="probability-row"><div class="probability-box"><span class="probability-label">APPROVAL PROBABILITY</span><strong class="probability-value">${percent(approve)}</strong><div class="mini-track"><span style="width:${Math.max(0, Math.min(100, approve * 100))}%"></span></div></div><div class="probability-box"><span class="probability-label">REJECTION PROBABILITY</span><strong class="probability-value">${percent(reject)}</strong><div class="mini-track red"><span style="width:${Math.max(0, Math.min(100, reject * 100))}%"></span></div></div></div>
    <div class="ratio-row"><span>Decision threshold</span><strong>${percent(data.decision_threshold)}</strong></div><div class="ratio-row"><span>Probability margin category</span><strong>${escapeHtml(data.confidence_level || "Not supplied")}</strong></div><div class="ratio-row"><span>Loan amount / annual income</span><strong>${numberFmt(data.debt_to_income_ratio)}×</strong></div>
    <p class="small-disclaimer">${escapeHtml(data.summary)} ${escapeHtml(data.disclaimer)}</p>
  </article>`;
}

function renderExplanation(data) {
  const factors = Array.isArray(data.feature_contributions) ? data.feature_contributions : [];
  const maxValue = Math.max(0.000001, ...factors.map((item) => Math.abs(Number(item.shap_value) || 0)));
  const factorHtml = factors.map((item) => {
    const value = Number(item.shap_value) || 0;
    const positive = value >= 0;
    const width = Math.max(1, Math.min(100, Math.abs(value) / maxValue * 100));
    return `<div class="factor"><div class="factor-top"><span class="factor-name">${escapeHtml(item.display_name || niceFeature(item.feature))}</span><span class="factor-value ${positive ? "positive" : "negative"}">${positive ? "+" : ""}${value.toFixed(4)}</span></div><span class="tag">${escapeHtml(item.impact_strength || "Impact")}</span><div class="factor-detail">Applicant value: ${escapeHtml(item.applicant_value)} · ${escapeHtml(item.direction)}</div><div class="bar-track"><span class="${positive ? "" : "negative"}" style="width:${width}%"></span></div><div class="factor-detail">${escapeHtml(item.explanation || "Contribution to this model output.")}</div></div>`;
  }).join("");
  const renderBullets = (items) => (items || []).length ? `<ul class="bullet-list">${items.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>` : `<p class="audit-copy">No non-zero factors were returned.</p>`;
  return `<article class="result-card"><div class="result-header"><div><p class="eyebrow">INDIVIDUAL EXPLANATION</p><h3 class="result-title">${escapeHtml(data.predicted_status)} · ${percent(data.approval_probability)} approval probability</h3><p class="result-subtitle">Base probability: ${percent(data.base_probability)} · threshold: ${percent(data.decision_threshold)}</p></div><span class="status-chip ${data.is_approved ? "approved" : "rejected"}">${data.is_approved ? "APPROVED" : "REJECTED"}</span></div>
  <h4 class="result-section-title">Feature contributions (SHAP)</h4><p class="audit-copy">Bars are scaled to the largest absolute contribution in this applicant profile. Values are in the model's output scale (log-odds), not percentage-point changes.</p><div class="factor-list">${factorHtml}</div>
  <div class="probability-row"><div><h4 class="result-section-title">Factors supporting approval</h4>${renderBullets(data.top_positive_factors)}</div><div><h4 class="result-section-title">Factors reducing approval</h4>${renderBullets(data.top_negative_factors)}</div></div>
  <h4 class="result-section-title">Main reasons</h4>${renderBullets(data.reasons)}<h4 class="result-section-title">Responsible suggestions</h4><div class="suggestion-list">${(data.suggestions || []).map((x) => `<div class="suggestion">${escapeHtml(x)}</div>`).join("")}</div>
  <div class="model-note">${escapeHtml(data.model_dependence_note || "Global dependency information was not included.")}</div><p class="small-disclaimer">${escapeHtml(data.disclaimer || "SHAP describes this model's behaviour; it does not establish causation or actual bank policy.")}</p></article>`;
}

function renderDependency(report) {
  const ablation = report.feature_removal_experiment || {};
  const full = ablation.model_a_all_features || {};
  const reduced = ablation.model_b_without_dominant_feature || {};
  const delta = ablation.performance_drop || {};
  const shapTop = report.mean_absolute_shap_contribution;
  const folds = report.stability_across_folds || [];
  const correlations = report.high_feature_correlations || [];
  const importanceRows = report.global_importance || report.feature_importance || [];
  const foldConsistent = folds.length ? folds.filter((row) => row.top_feature === report.most_influential_feature).length : null;
  let topHtml = "<p class='audit-copy'>See reports/feature_importance.csv for detailed rankings.</p>";
  if (Array.isArray(importanceRows) && importanceRows.length) {
    const rows = importanceRows.slice(0, 7);
    const maxImp = Math.max(0.000001, ...rows.map((x) => Number(x.mean_abs_shap ?? x.shap_importance ?? x.importance ?? 0)));
    topHtml = `<div class="audit-rows">${rows.map((row) => { const value = Number(row.mean_abs_shap ?? row.shap_importance ?? row.importance ?? 0); const name = row.feature || row.original_feature || row.transformed_feature; return `<div class="audit-row"><span>${escapeHtml(niceFeature(name))}</span><div class="bar-track"><span style="width:${Math.max(1, value / maxImp * 100)}%"></span></div><strong>${value.toFixed(3)}</strong></div>`; }).join("")}</div>`;
  }
  return `<div class="audit-card"><span class="audit-kicker">GLOBAL ANALYSIS</span><h3>Most influential feature</h3><div class="audit-big">${escapeHtml(niceFeature(report.most_influential_feature || "Not available"))}</div><p class="audit-copy">Mean absolute SHAP contribution: ${Number.isFinite(Number(shapTop)) ? Number(shapTop).toFixed(4) : "See report"}. This is a global model-level measure.</p></div>
  <div class="audit-card"><span class="audit-kicker">ABLATION TEST</span><h3>Model with top feature removed</h3><div class="audit-big">${Number.isFinite(Number(reduced.test_roc_auc)) ? Number(reduced.test_roc_auc).toFixed(3) : "—"}</div><p class="audit-copy">ROC-AUC without ${escapeHtml(niceFeature(reduced.removed_feature || report.most_influential_feature || "top feature"))}, compared with ${Number.isFinite(Number(full.test_roc_auc)) ? Number(full.test_roc_auc).toFixed(3) : "—"} for the full feature set.</p><div class="pair-list"><div class="pair-item"><span>ROC-AUC change</span><strong>${Number.isFinite(Number(delta.roc_auc_delta)) ? Number(delta.roc_auc_delta).toFixed(3) : "—"}</strong></div><div class="pair-item"><span>Accuracy change</span><strong>${Number.isFinite(Number(delta.accuracy_delta)) ? Number(delta.accuracy_delta).toFixed(3) : "—"}</strong></div><div class="pair-item"><span>Full model F1</span><strong>${Number.isFinite(Number(full.test_f1)) ? Number(full.test_f1).toFixed(3) : "—"}</strong></div><div class="pair-item"><span>Ablation model F1</span><strong>${Number.isFinite(Number(reduced.test_f1)) ? Number(reduced.test_f1).toFixed(3) : "—"}</strong></div></div></div>
  <div class="audit-card wide"><span class="audit-kicker">IMPORTANCE OVERVIEW</span><h3>Global feature ranking</h3>${topHtml}</div>
  <div class="audit-card"><span class="audit-kicker">CROSS-VALIDATION</span><h3>Ranking stability</h3><div class="audit-big">${foldConsistent === null ? folds.length : `${foldConsistent}/${folds.length}`}</div><p class="audit-copy">Folds where the reported most influential feature was ranked first. ${folds.length ? "Rankings can still vary with data splits." : "Fold details are listed in the saved report."}</p>${folds.length ? `<div class="pair-list">${folds.map((f) => `<div class="pair-item"><span>Fold ${escapeHtml(f.fold)}</span><strong>${escapeHtml(niceFeature(f.top_feature))} · ${(Number(f.top_feature_weight) * 100).toFixed(1)}%</strong></div>`).join("")}</div>` : ""}</div>
  <div class="audit-card"><span class="audit-kicker">FEATURE CORRELATION</span><h3>Highly correlated numeric pairs</h3><div class="audit-big">${correlations.length}</div><p class="audit-copy">Pairs with absolute correlation above the training report's selected cutoff.</p><div class="pair-list">${correlations.length ? correlations.slice(0, 8).map((row) => `<div class="pair-item"><span>${escapeHtml(niceFeature(row.feature_1))} ↔ ${escapeHtml(niceFeature(row.feature_2))}</span><strong>${Number(row.correlation).toFixed(2)}</strong></div>`).join("") : `<div class="pair-item"><span>No pairs listed in report</span><strong>—</strong></div>`}</div></div>`;
}

async function loadModelInfo() {
  try {
    const info = await apiFetch("/api/model-info");
    $("#metric-model").textContent = info.model_name || "Gradient Boosting";
    $("#metric-threshold").textContent = percent(info.selected_threshold);
    $("#metric-auc").textContent = Number.isFinite(Number(info.metrics_summary?.test_roc_auc)) ? Number(info.metrics_summary.test_roc_auc).toFixed(3) : "—";
    $("#metric-feature").textContent = niceFeature(info.most_influential_feature || "Not available");
  } catch (_) { /* Keep static fallback values when the model metadata is absent. */ }
}
async function loadHealth() {
  try {
    const status = await apiFetch("/api/health");
    const el = $("#api-status");
    const dot = document.querySelector(".status-dot");
    if (status.model_artifact_present && status.metadata_present) { el.textContent = "Model artifacts found"; dot.classList.add("ready"); }
    else { el.textContent = "Training setup needed"; dot.classList.add("error"); }
  } catch (_) { $("#api-status").textContent = "API unavailable"; document.querySelector(".status-dot").classList.add("error"); }
}

async function init() {
  try {
    const schema = await apiFetch("/api/schema");
    featureFields = schema.features || [];
    if (!featureFields.length) throw new Error("The training metadata does not contain a feature schema.");
    renderFormFields($("#predict-fields"), "predict");
    renderFormFields($("#explain-fields"), "explain");
  } catch (error) {
    $("#predict-fields").innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
    $("#explain-fields").innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
  await Promise.all([loadHealth(), loadModelInfo()]);
  try { $("#dependency-result").innerHTML = renderDependency(await apiFetch("/api/dependency-analysis")); }
  catch (error) { $("#dependency-result").innerHTML = `<div class="error-box audit-empty">${escapeHtml(error.message)} If the report is missing, run <code>python train_model.py</code> from the project root.</div>`; }
}

$$(".nav-item").forEach((button) => button.addEventListener("click", () => {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item === button));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === button.dataset.tab));
  if (window.innerWidth < 720) button.scrollIntoView({ behavior: "smooth", block: "nearest" });
}));

$("#predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  const target = $("#predict-result");
  try {
    const payload = collectForm(event.currentTarget, "predict");
    setBusy(button, true, "Predicting");
    target.innerHTML = `<div class="empty-result">Evaluating the applicant profile…</div>`;
    target.innerHTML = renderPrediction(await apiFetch("/api/predict", { method: "POST", body: JSON.stringify(payload) }));
  } catch (error) { showError(target, error); }
  finally { setBusy(button, false); }
});

$("#explain-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  const target = $("#explain-result");
  try {
    const payload = collectForm(event.currentTarget, "explain");
    setBusy(button, true, "Explaining");
    target.innerHTML = `<div class="empty-result">Calculating local SHAP contributions…</div>`;
    target.innerHTML = renderExplanation(await apiFetch("/api/explain", { method: "POST", body: JSON.stringify(payload) }));
  } catch (error) { showError(target, error); }
  finally { setBusy(button, false); }
});

init();
