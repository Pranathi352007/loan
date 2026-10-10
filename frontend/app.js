const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const integerFields = new Set(["no_of_dependents", "loan_term", "cibil_score"]);

const FALLBACK_FIELDS = [
  { name: "no_of_dependents", label: "Number of dependents", kind: "number", min: 0, max: 10, step: 1, default: 2 },
  { name: "education", label: "Education level", kind: "select", options: ["Graduate", "Not Graduate"], default: "Graduate" },
  { name: "self_employed", label: "Self-employed", kind: "select", options: ["No", "Yes"], default: "No" },
  { name: "income_annum", label: "Annual income (₹)", kind: "number", min: 1, step: 1000, default: 6000000 },
  { name: "loan_amount", label: "Requested loan amount (₹)", kind: "number", min: 1, step: 1000, default: 15000000 },
  { name: "loan_term", label: "Loan term (years)", kind: "number", min: 1, max: 30, step: 1, default: 10 },
  { name: "cibil_score", label: "CIBIL score", kind: "number", min: 300, max: 900, step: 1, default: 750 },
  { name: "residential_assets_value", label: "Residential assets value (₹)", kind: "number", min: 0, step: 1000, default: 12000000 },
  { name: "commercial_assets_value", label: "Commercial assets value (₹)", kind: "number", min: 0, step: 1000, default: 4000000 },
  { name: "luxury_assets_value", label: "Luxury assets value (₹)", kind: "number", min: 0, step: 1000, default: 18000000 },
  { name: "bank_asset_value", label: "Bank asset value (₹)", kind: "number", min: 0, step: 1000, default: 8000000 }
];
let featureFields = FALLBACK_FIELDS.slice();

function escapeHtml(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
  });
}
function niceFeature(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, function (letter) { return letter.toUpperCase(); });
}
function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? (number * 100).toFixed(1) + "%" : "—";
}
function numberFmt(value, digits) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: digits == null ? 3 : digits }) : "—";
}
function signed(value, digits) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return (number > 0 ? "+" : "") + number.toFixed(digits == null ? 3 : digits);
}

function renderFormFields(container) {
  container.innerHTML = featureFields.map(function (field) {
    const id = container.id + "-" + field.name;
    if (field.kind === "select") {
      const options = (field.options || []).map(function (option) {
        const selected = String(option) === String(field.default) ? " selected" : "";
        return '<option value="' + escapeHtml(option) + '"' + selected + ">" + escapeHtml(option) + "</option>";
      }).join("");
      return '<label class="field" for="' + id + '"><span>' + escapeHtml(field.label) + '</span><select id="' + id + '" name="' + escapeHtml(field.name) + '" required>' + options + "</select></label>";
    }
    const min = field.min !== undefined ? ' min="' + field.min + '"' : "";
    const max = field.max !== undefined ? ' max="' + field.max + '"' : "";
    const step = field.step !== undefined ? ' step="' + field.step + '"' : ' step="any"';
    const value = field.default !== undefined ? ' value="' + escapeHtml(field.default) + '"' : "";
    let hint = "";
    if (field.name === "loan_amount") hint = "Requested principal, not total repayment.";
    if (field.name === "cibil_score") hint = "Use a value from 300 to 900.";
    return '<label class="field" for="' + id + '"><span>' + escapeHtml(field.label) + '</span><input id="' + id + '" name="' + escapeHtml(field.name) + '" type="number" inputmode="decimal"' + min + max + step + value + ' required><small>' + escapeHtml(hint) + "</small></label>";
  }).join("");
}
function captureValues(form) {
  const values = {};
  Array.from(form.elements).forEach(function (element) {
    if (element.name) values[element.name] = element.value;
  });
  return values;
}
function restoreValues(form, values) {
  Object.keys(values || {}).forEach(function (name) {
    const element = form.elements.namedItem(name);
    if (!element) return;
    if (element.tagName === "SELECT" && !Array.from(element.options).some(function (option) { return option.value === values[name]; })) return;
    element.value = values[name];
  });
}
function collectForm(form) {
  const data = {};
  for (const field of featureFields) {
    const element = form.elements.namedItem(field.name);
    if (!element) throw new Error("Missing input: " + field.label);
    const value = element.value.trim();
    if (!value) throw new Error(field.label + " is required.");
    if (field.kind === "select") {
      data[field.name] = value;
    } else {
      const number = Number(value);
      if (!Number.isFinite(number)) throw new Error(field.label + " must be a valid number.");
      if (integerFields.has(field.name) && !Number.isInteger(number)) throw new Error(field.label + " must be a whole number.");
      if (field.min !== undefined && number < field.min) throw new Error(field.label + " must be at least " + field.min + ".");
      if (field.max !== undefined && number > field.max) throw new Error(field.label + " must be no more than " + field.max + ".");
      data[field.name] = number;
    }
  }
  return data;
}

async function apiFetch(path, options) {
  const settings = options || {};
  const response = await fetch(path, {
    method: settings.method || "GET",
    headers: Object.assign({ "Content-Type": "application/json" }, settings.headers || {}),
    body: settings.body
  });
  const data = await response.json().catch(function () { return {}; });
  if (!response.ok) throw new Error(data.detail || ("Request failed (" + response.status + ")."));
  return data;
}
function showError(target, error) {
  target.innerHTML = '<div class="error"><strong>Could not complete the request.</strong><p>' + escapeHtml(error.message || error) + "</p></div>";
}
function setBusy(button, busy, label) {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.disabled = true;
    button.textContent = label + "…";
  } else {
    button.disabled = false;
    if (button.dataset.originalText) button.textContent = button.dataset.originalText;
  }
}
function renderList(items) {
  if (!Array.isArray(items) || items.length === 0) return '<p class="muted">No factors were returned.</p>';
  return '<ul class="result-list">' + items.map(function (item) { return "<li>" + escapeHtml(item) + "</li>"; }).join("") + "</ul>";
}
function renderPrediction(data) {
  const approved = Boolean(data.is_approved);
  return '<article class="result-card">' +
    '<p><span class="status ' + (approved ? "" : "rejected") + '">' + escapeHtml(data.predicted_status) + ' in this model</span></p>' +
    '<p>' + escapeHtml(data.summary) + "</p>" +
    '<div class="metrics"><div class="metric"><span>Estimated approval probability</span><strong>' + percent(data.approval_probability) + '</strong></div>' +
    '<div class="metric"><span>Estimated rejection probability</span><strong>' + percent(data.rejection_probability) + '</strong></div></div>' +
    '<p class="muted">Decision threshold: <strong>' + percent(data.decision_threshold) + '</strong> · Distance category: <strong>' + escapeHtml(data.confidence_level || "Not supplied") + '</strong></p>' +
    '<p class="muted">Loan amount / annual income: ' + numberFmt(data.debt_to_income_ratio, 2) + '×</p>' +
    '<div class="warning">' + escapeHtml(data.disclaimer) + "</div></article>";
}
function renderFactor(item) {
  const value = Number(item.shap_value) || 0;
  const positive = value >= 0;
  const width = Math.max(1, Math.min(100, Math.abs(value) * 100));
  return '<li class="factor"><div class="factor-title"><span>' + escapeHtml(item.display_name || niceFeature(item.feature)) +
    '</span><strong class="' + (positive ? "positive" : "negative") + '">' + signed(value, 4) + '</strong></div>' +
    '<p class="muted">Applicant value: ' + escapeHtml(item.applicant_value) + ' · ' + escapeHtml(item.direction) + "</p>" +
    '<div class="bar"><span class="' + (positive ? "" : "negative") + '" style="width:' + width + '%"></span></div>' +
    '<p>' + escapeHtml(item.explanation || "Contribution to this model output.") + "</p></li>";
}
function renderExplanation(data) {
  const factors = Array.isArray(data.feature_contributions) ? data.feature_contributions : [];
  const first = factors.slice(0, 5).map(renderFactor).join("");
  const rest = factors.slice(5).map(renderFactor).join("");
  return '<article class="result-card"><p><span class="status ' + (data.is_approved ? "" : "rejected") + '">' +
    escapeHtml(data.predicted_status) + ' in this model</span></p><p>Estimated approval probability: <strong>' + percent(data.approval_probability) +
    '</strong> · Decision threshold: <strong>' + percent(data.decision_threshold) + "</strong></p>" +
    '<h3>Top SHAP contributions</h3><p class="muted">Values are on the model output scale (log-odds), not percentage-point changes.</p><ul class="result-list">' + first + "</ul>" +
    (rest ? '<details><summary>View all feature contributions</summary><ul class="result-list">' + rest + "</ul></details>" : "") +
    '<h3>Factors supporting approval</h3>' + renderList(data.top_positive_factors) +
    '<h3>Factors reducing approval</h3>' + renderList(data.top_negative_factors) +
    '<h3>Main reasons</h3>' + renderList(data.reasons) +
    '<h3>Suggestions</h3>' + ((data.suggestions || []).map(function (item) { return '<div class="suggestion">' + escapeHtml(item) + "</div>"; }).join("") || '<p class="muted">No suggestions returned.</p>') +
    '<p class="warning">' + escapeHtml(data.model_dependence_note || "") + "</p>" +
    '<p class="muted">' + escapeHtml(data.disclaimer || "") + "</p></article>";
}
function renderDependency(report) {
  const experiment = report.feature_removal_experiment || {};
  const full = experiment.model_a_all_features || {};
  const reduced = experiment.model_b_without_dominant_feature || {};
  const delta = experiment.performance_drop || {};
  const rows = Array.isArray(report.global_importance) ? report.global_importance.slice(0, 8) : [];
  const maxValue = Math.max(0.000001, ...rows.map(function (row) { return Math.abs(Number(row.mean_abs_shap) || 0); }));
  const ranking = rows.length ? rows.map(function (row) {
    const value = Number(row.mean_abs_shap) || 0;
    return '<div class="importance-row"><span>' + escapeHtml(niceFeature(row.feature)) + '</span><div class="bar"><span style="width:' + Math.max(1, value / maxValue * 100) + '%"></span></div><strong>' + value.toFixed(3) + "</strong></div>";
  }).join("") : '<p class="muted">Global ranking unavailable; see reports/feature_importance.csv.</p>';
  const folds = Array.isArray(report.stability_across_folds) ? report.stability_across_folds : [];
  const foldList = folds.length ? '<ul class="result-list">' + folds.map(function (row) {
    return "<li>Fold " + escapeHtml(row.fold) + ": " + escapeHtml(niceFeature(row.top_feature)) + " ranked first by tree feature importance.</li>";
  }).join("") + "</ul>" : '<p class="muted">Fold rankings are not available.</p>';
  const correlations = Array.isArray(report.high_feature_correlations) ? report.high_feature_correlations : [];
  return '<div class="dependency-card"><h3>Most influential global feature</h3><div class="large">' +
    escapeHtml(niceFeature(report.most_influential_feature || "Not available")) + "</div><p class=\"muted\">Mean absolute SHAP: " +
    numberFmt(report.mean_absolute_shap_contribution, 4) + "</p></div>" +
    '<div class="dependency-card"><h3>Feature removal test</h3><p>Full model ROC-AUC: <strong>' + numberFmt(full.test_roc_auc, 3) +
    '</strong></p><p>Without ' + escapeHtml(niceFeature(reduced.removed_feature || report.most_influential_feature)) +
    ': <strong>' + numberFmt(reduced.test_roc_auc, 3) + "</strong></p><p>ROC-AUC change (reduced minus full): <strong>" +
    signed(delta.roc_auc_delta, 3) + "</strong></p><p class=\"muted\">Accuracy change at the same decision threshold: " + signed(delta.accuracy_delta, 3) + "</p></div>" +
    '<div class="dependency-card wide"><h3>Global feature ranking (SHAP)</h3>' + ranking + "</div>" +
    '<div class="dependency-card"><h3>Fold-by-fold rankings</h3><p class="muted">These use tree impurity importance; they are not the same measure as SHAP importance.</p>' + foldList + "</div>" +
    '<div class="dependency-card"><h3>Correlated numeric features</h3><div class="large">' + correlations.length + "</div><p class=\"muted\">Pairs whose absolute Pearson correlation exceeded 0.50 in the training data.</p>" +
    (correlations.length ? '<ul class="result-list">' + correlations.slice(0, 6).map(function (row) { return "<li>" + escapeHtml(niceFeature(row.feature_1)) + " ↔ " + escapeHtml(niceFeature(row.feature_2)) + ": " + numberFmt(row.correlation, 2) + "</li>"; }).join("") + "</ul>" : "") +
    "</div>";
}
function sameLayout(left, right) {
  const signature = function (fields) {
    return fields.map(function (field) {
      return [field.name, field.label, field.kind, field.default, field.min, field.max, field.step, (field.options || []).join("|")].join("~");
    });
  };
  return JSON.stringify(signature(left)) === JSON.stringify(signature(right));
}

function renderFallbackForms() {
  renderFormFields($("#predict-fields"));
  renderFormFields($("#explain-fields"));
}
function addTabHandlers() {
  $$(".tab-button").forEach(function (button) {
    button.addEventListener("click", function () {
      $$(".tab-button").forEach(function (item) { item.classList.toggle("active", item === button); });
      $$(".panel").forEach(function (panel) { panel.classList.toggle("active", panel.id === button.dataset.tab); });
    });
  });
}
function addFormHandler(formSelector, endpoint, resultSelector, buttonLabel, renderer) {
  $(formSelector).addEventListener("submit", async function (event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector("button[type='submit']");
    const target = $(resultSelector);
    try {
      const payload = collectForm(form);
      setBusy(button, true, buttonLabel);
      target.innerHTML = '<div class="result-card"><p>Working…</p></div>';
      const response = await apiFetch(endpoint, { method: "POST", body: JSON.stringify(payload) });
      target.innerHTML = renderer(response);
    } catch (error) {
      showError(target, error);
    } finally {
      setBusy(button, false);
    }
  });
}

async function loadSchema() {
  try {
    const response = await apiFetch("/api/schema");
    const incoming = response.features || [];
    if (!incoming.length) throw new Error("No feature schema was returned.");
    if (!sameLayout(featureFields, incoming)) {
      const predictValues = captureValues($("#predict-form"));
      const explainValues = captureValues($("#explain-form"));
      featureFields = incoming;
      renderFallbackForms();
      restoreValues($("#predict-form"), predictValues);
      restoreValues($("#explain-form"), explainValues);
    } else {
      featureFields = incoming;
    }
    $("#predict-schema-note").textContent = "Fields loaded from the model metadata.";
    $("#explain-schema-note").textContent = "Fields loaded from the model metadata.";
  } catch (_) {
    $("#predict-schema-note").textContent = "Using built-in fields; the model schema could not be reached.";
    $("#explain-schema-note").textContent = "Using built-in fields; the model schema could not be reached.";
  }
}
async function loadHealth() {
  const statusText = $("#api-status");
  try {
    const health = await apiFetch("/api/health");
    if (health.model_loaded) {
      statusText.textContent = "Model loaded";
      statusText.title = "The trained model was loaded successfully.";
      statusText.classList.remove("error-text");
    } else {
      statusText.textContent = "Model needs setup";
      statusText.title = health.model_error || "The model artifact is missing or could not be loaded. Forms remain available.";
      statusText.classList.add("error-text");
    }
  } catch (_) {
    statusText.textContent = "API unavailable";
    statusText.title = "Start the FastAPI server to submit predictions.";
    statusText.classList.add("error-text");
  }
}
async function loadModelInfo() {
  try {
    const info = await apiFetch("/api/model-info");
    $("#predict-schema-note").title = info.synthetic_disclaimer || "";
  } catch (_) {}
}
async function loadDependency() {
  try {
    const report = await apiFetch("/api/dependency-analysis");
    $("#dependency-result").innerHTML = renderDependency(report);
  } catch (error) {
    $("#dependency-result").innerHTML = '<div class="error">' + escapeHtml(error.message || error) + " If the report is missing, run the training script from the project root.</div>";
  }
}

renderFallbackForms();
addTabHandlers();
addFormHandler("#predict-form", "/api/predict", "#predict-result", "Predicting", renderPrediction);
addFormHandler("#explain-form", "/api/explain", "#explain-result", "Explaining", renderExplanation);
Promise.all([loadSchema(), loadHealth(), loadModelInfo(), loadDependency()]);
