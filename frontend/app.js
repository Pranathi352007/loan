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
  return String(value == null ? "" : value).replace(/[&<>"']/g, function (ch) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
  });
}
function niceFeature(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
}
function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? (number * 100).toFixed(1) + "%" : "—";
}
function numberFmt(value, digits) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: digits == null ? 2 : digits }) : "—";
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
        const chosen = String(option) === String(field.default) ? " selected" : "";
        return '<option value="' + escapeHtml(option) + '"' + chosen + ">" + escapeHtml(option) + "</option>";
      }).join("");
      return '<label class="field" for="' + id + '"><span>' + escapeHtml(field.label) + '</span><select id="' + id + '" name="' + escapeHtml(field.name) + '" required>' + options + "</select></label>";
    }
    const min = field.min !== undefined ? ' min="' + field.min + '"' : "";
    const max = field.max !== undefined ? ' max="' + field.max + '"' : "";
    const step = field.step !== undefined ? ' step="' + field.step + '"' : ' step="any"';
    const value = field.default !== undefined ? ' value="' + escapeHtml(field.default) + '"' : "";
    const hint = field.name === "cibil_score" ? "Use a value from 300 to 900." : "";
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
      if (!Number.isFinite(number)) throw new Error(field.label + " must be numeric.");
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
  const result = await response.json().catch(function () { return {}; });
  if (!response.ok) throw new Error(result.detail || ("Request failed (" + response.status + ")."));
  return result;
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
function addFactorCards(factors, gemini) {
  if (!Array.isArray(factors) || !factors.length) return '<p class="muted">No SHAP factors were available.</p>';
  const explanations = new Map(((gemini && gemini.factor_explanations) || []).map(function (item) { return [item.feature, item.explanation]; }));
  const maxAbs = Math.max(0.000001, ...factors.map(function (item) { return Math.abs(Number(item.shap_value) || 0); }));
  return '<ul class="factors">' + factors.map(function (item) {
    const val = Number(item.shap_value) || 0;
    const positive = val >= 0;
    const detail = explanations.get(item.feature) || item.explanation || "Contribution to the model's estimate.";
    const width = Math.max(1, Math.min(100, Math.abs(val) / maxAbs * 100));
    return '<li class="factor"><div class="factor-title"><span>' + escapeHtml(item.display_name || niceFeature(item.feature)) +
      '</span><strong class="' + (positive ? "positive" : "negative") + '">' + signed(val, 4) + "</strong></div>" +
      '<p class="muted">Applicant value: ' + escapeHtml(item.applicant_value) + " · " + escapeHtml(item.direction) + "</p>" +
      '<div class="bar"><span class="' + (positive ? "" : "negative") + '" style="width:' + width + '%"></span></div>' +
      "<p>" + escapeHtml(detail) + "</p></li>";
  }).join("") + "</ul>";
}
function renderResult(data, workflow) {
  const prediction = data.prediction || {};
  const gemini = data.gemini || {};
  const recorded = data.recorded_decision;
  const predicted = data.model_prediction || prediction.predicted_status || "Unknown";
  const statusClass = function (value) { return value === "Approved" ? "" : value === "Rejected" ? "rejected" : "neutral"; };
  let statusSection = '<p><span class="status ' + statusClass(predicted) + '">Model estimate: ' + escapeHtml(predicted) + "</span></p>";
  if (workflow === "lookup") {
    statusSection = '<div class="metrics"><div class="metric"><span>Status stored in dataset</span><strong><span class="status ' + statusClass(recorded) + '">' + escapeHtml(recorded || "Unknown") + '</span></strong></div>' +
      '<div class="metric"><span>Gradient Boosting estimate</span><strong><span class="status ' + statusClass(predicted) + '">' + escapeHtml(predicted) + "</span></strong></div></div>";
  }
  const mismatch = workflow === "lookup" && data.decision_matches_record === false;
  const mismatchNote = mismatch
    ? '<div class="warning"><strong>The stored status and model estimate differ.</strong> SHAP explains the model estimate, not the reason the recorded application received its stored status.</div>'
    : "";
  const geminiLine = gemini.available
    ? '<p class="gemini-note">Plain-language explanation generated by Gemini (' + escapeHtml(gemini.model || "configured model") + ").</p>"
    : '<p class="gemini-note">' + escapeHtml(gemini.overview || "Gemini is unavailable. The computed SHAP evidence remains below.") + "</p>";
  const overview = gemini.available ? '<p>' + escapeHtml(gemini.overview) + "</p>" : "";
  const probability = prediction.approval_probability !== undefined ? prediction.approval_probability : data.model_explanation && data.model_explanation.approval_probability;
  const rejectionProbability = prediction.rejection_probability !== undefined ? prediction.rejection_probability : (probability === undefined ? undefined : 1 - Number(probability));
  const suggestions = Array.isArray(data.suggestions) ? data.suggestions : [];
  const suggestionSection = suggestions.length
    ? "<h3>Suggestions to consider</h3>" + suggestions.map(function (item) { return '<div class="suggestion">' + escapeHtml(item) + "</div>"; }).join("")
    : "";
  const sourceText = workflow === "lookup"
    ? "These are factors that influenced the Gradient Boosting model's estimate for this row. The dataset does not contain official bank reasons."
    : "These are factors that influenced this project's model estimate. They are not actual bank policy.";
  return '<article class="result-card"><div class="result-header"><div><h3>' + (workflow === "lookup" ? "Application " + escapeHtml(data.application_id) : "Prediction result") +
    "</h3><p class=\"muted\">" + escapeHtml(sourceText) + "</p></div></div>" +
    statusSection + mismatchNote +
    '<div class="metrics"><div class="metric"><span>Estimated approval probability</span><strong>' + percent(probability) + '</strong></div>' +
    '<div class="metric"><span>Estimated rejection probability</span><strong>' + percent(rejectionProbability) + "</strong></div></div>" +
    geminiLine + overview +
    "<h3>Main factors affecting the model estimate</h3>" + addFactorCards(data.factors, gemini) +
    suggestionSection +
    '<p class="warning">' + escapeHtml(data.disclaimer || "This is an educational estimate based on synthetic data.") + "</p></article>";
}
function showWorkflow(which) {
  const isNew = which === "new";
  $("#options").hidden = true;
  $("#new-application-panel").hidden = !isNew;
  $("#lookup-application-panel").hidden = isNew;
  (isNew ? $("#new-application-panel") : $("#lookup-application-panel")).scrollIntoView({ behavior: "smooth", block: "start" });
}
function returnToOptions(event) {
  event.preventDefault();
  $("#new-application-panel").hidden = true;
  $("#lookup-application-panel").hidden = true;
  $("#options").hidden = false;
  $("#options").scrollIntoView({ behavior: "smooth", block: "start" });
}
async function loadSchema() {
  try {
    const response = await apiFetch("/api/schema");
    const incoming = response.features || [];
    if (!incoming.length) throw new Error("No feature schema was returned.");
    const values = captureValues($("#predict-form"));
    featureFields = incoming;
    renderFormFields($("#predict-fields"));
    restoreValues($("#predict-form"), values);
  } catch (_) {
    renderFormFields($("#predict-fields"));
  }
}
async function loadHealth() {
  const el = $("#api-status");
  try {
    const health = await apiFetch("/api/health");
    const modelText = health.model_loaded ? "Model loaded" : "Model needs setup";
    const geminiText = health.gemini_configured ? "Gemini configured" : "Gemini key missing";
    el.textContent = modelText + " · " + geminiText;
    if (!health.model_loaded || !health.gemini_configured) el.classList.add("error-text");
  } catch (_) {
    el.textContent = "API unavailable";
    el.classList.add("error-text");
  }
}
$("#new-application-button").addEventListener("click", function () { showWorkflow("new"); });
$("#lookup-application-button").addEventListener("click", function () { showWorkflow("lookup"); });
$$("[data-back]").forEach(function (link) { link.addEventListener("click", returnToOptions); });

$("#predict-form").addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  const target = $("#predict-result");
  try {
    const payload = collectForm(event.currentTarget);
    setBusy(button, true, "Predicting");
    target.innerHTML = '<div class="result-card"><p>Making the estimate and preparing its explanation…</p></div>';
    const result = await apiFetch("/api/predict-and-explain", { method: "POST", body: JSON.stringify(payload) });
    target.innerHTML = renderResult(result, "predict");
  } catch (error) {
    showError(target, error);
  } finally {
    setBusy(button, false);
  }
});
$("#lookup-form").addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  const target = $("#lookup-result");
  const rawId = $("#application-id").value.trim();
  try {
    if (!rawId || !Number.isInteger(Number(rawId)) || Number(rawId) < 1) throw new Error("Enter a positive whole-number application ID.");
    setBusy(button, true, "Searching");
    target.innerHTML = '<div class="result-card"><p>Finding the application and preparing its explanation…</p></div>';
    const result = await apiFetch("/api/application/" + encodeURIComponent(rawId));
    target.innerHTML = renderResult(result, "lookup");
  } catch (error) {
    showError(target, error);
  } finally {
    setBusy(button, false);
  }
});
renderFormFields($("#predict-fields"));
Promise.all([loadSchema(), loadHealth()]);
