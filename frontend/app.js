const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const integerFields = new Set(["no_of_dependents", "loan_term", "cibil_score"]);

const FALLBACK_FIELDS = [
  { name: "no_of_dependents", label: "Number of dependents", kind: "number", min: 0, max: 10, step: 1, default: 2 },
  { name: "education", label: "Education level", kind: "select", options: ["Graduate", "Not Graduate"], default: "Graduate" },
  { name: "self_employed", label: "Self-employed", kind: "select", options: ["No", "Yes"], default: "No" },
  { name: "income_annum", label: "Annual income (₹)", kind: "number", min: 1, step: 1, default: 6000000 },
  { name: "loan_amount", label: "Requested loan amount (₹)", kind: "number", min: 1, step: 1, default: 15000000 },
  { name: "loan_term", label: "Loan term (years)", kind: "number", min: 1, max: 30, step: 1, default: 10 },
  { name: "cibil_score", label: "CIBIL score", kind: "number", min: 300, max: 900, step: 1, default: 750 },
  { name: "residential_assets_value", label: "Residential assets value (₹)", kind: "number", min: 0, step: 1, default: 12000000 },
  { name: "commercial_assets_value", label: "Commercial assets value (₹)", kind: "number", min: 0, step: 1, default: 4000000 },
  { name: "luxury_assets_value", label: "Luxury assets value (₹)", kind: "number", min: 0, step: 1, default: 18000000 },
  { name: "bank_asset_value", label: "Bank asset value (₹)", kind: "number", min: 0, step: 1, default: 8000000 }
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
    const hint = "";
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
function renderFactorCards(factors) {
  if (!Array.isArray(factors) || !factors.length) {
    return '<p class="muted">No key details were available for this result.</p>';
  }
  return '<div class="factor-grid">' + factors.slice(0, 3).map(function (item) {
    const name = item.display_name || niceFeature(item.feature);
    return '<article class="factor-card"><h4>' + escapeHtml(name) +
      '</h4><span class="value-label">Your value</span><p class="factor-value">' +
      escapeHtml(item.applicant_value == null ? "Not available" : item.applicant_value) +
      '</p></article>';
  }).join("") + '</div>';
}
function renderReasonCards(factors, gemini) {
  if (!Array.isArray(factors) || !factors.length) {
    return '<p class="muted">A simple explanation is not available for this result.</p>';
  }
  const explanations = new Map(((gemini && gemini.factor_explanations) || []).map(function (item) {
    return [item.feature, item.explanation];
  }));
  return '<div class="reason-list">' + factors.slice(0, 3).map(function (item) {
    const name = item.display_name || niceFeature(item.feature);
    const reason = item.reason || explanations.get(item.feature) ||
      "This detail was considered as part of the result.";
    return '<article class="reason-card"><h4>' + escapeHtml(name) +
      '</h4><p>' + escapeHtml(reason) + '</p></article>';
  }).join("") + '</div>';
}
function renderSuggestionCards(suggestions, factors) {
  if (!Array.isArray(suggestions) || !suggestions.length) return "";
  const names = new Map((Array.isArray(factors) ? factors : []).map(function (factor) {
    return [factor.feature, factor.display_name || niceFeature(factor.feature)];
  }));
  return '<section class="suggestions-block"><h3>Suggestions to consider</h3><ul class="suggestion-list">' +
    suggestions.slice(0, 3).map(function (item) {
      const text = typeof item === "string" ? item : (item && item.suggestion) || "";
      const feature = item && typeof item === "object" ? item.feature : "";
      const name = names.get(feature) || (feature ? niceFeature(feature) : "");
      return text ? '<li class="suggestion">' +
        (name ? '<strong>' + escapeHtml(name) + '</strong>' : '') +
        '<p>' + escapeHtml(text) + '</p></li>' : "";
    }).join("") + '</ul></section>';
}
function renderResult(data, workflow) {
  const prediction = data.prediction || {};
  const gemini = data.gemini || {};
  const factors = (Array.isArray(data.factors) ? data.factors : []).slice(0, 3);
  const recorded = data.recorded_decision;
  const predicted = data.model_prediction || prediction.predicted_status || "Unknown";
  const statusClass = function (value) {
    return value === "Approved" ? "" : value === "Rejected" ? "rejected" : "neutral";
  };

  let statusSection = '<div class="decision-grid single-decision"><div class="decision-item">' +
    '<span>Estimated outcome</span><strong><span class="status ' + statusClass(predicted) + '">' +
    escapeHtml(predicted) + '</span></strong></div></div>';
  if (workflow === "lookup") {
    statusSection = '<div class="decision-grid">' +
      '<div class="decision-item"><span>Saved outcome</span><strong><span class="status ' +
      statusClass(recorded) + '">' + escapeHtml(recorded || "Unknown") + '</span></strong></div>' +
      '<div class="decision-item"><span>Demo estimate</span><strong><span class="status ' +
      statusClass(predicted) + '">' + escapeHtml(predicted) + '</span></strong></div></div>';
  }

  const mismatch = workflow === "lookup" && data.decision_matches_record === false;
  const mismatchNote = mismatch
    ? '<div class="warning"><strong>The saved outcome and demo estimate are different.</strong> The reasons below explain this demo estimate only. The sample record does not include the lender’s actual reasons.</div>'
    : "";
  const overview = gemini.overview
    ? '<section class="result-overview"><h3>Summary</h3><p>' + escapeHtml(gemini.overview) + '</p></section>'
    : "";
  const suggestions = renderSuggestionCards(data.suggestions, factors);
  const disclaimer = data.disclaimer ||
    "This demo uses sample data. The estimate is not a loan offer or a guarantee of approval.";
  return '<article class="result-card"><div class="result-header"><div><h3>' +
    (workflow === "lookup" ? "Application " + escapeHtml(data.application_id) : "Your result") +
    '</h3></div></div>' +
    statusSection + mismatchNote + overview +
    '<h3>Top 3 factors affecting this result</h3>' + renderFactorCards(factors) +
    '<h3>Why these factors matter</h3>' + renderReasonCards(factors, gemini) +
    suggestions +
    '<p class="warning">' + escapeHtml(disclaimer) + '</p></article>';
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
    const predictionText = health.model_loaded ? "Prediction service ready" : "Prediction setup needed";
    const explanationText = health.gemini_configured ? "Explanations ready" : "AI explanations need setup";
    el.textContent = predictionText + " · " + explanationText;
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
