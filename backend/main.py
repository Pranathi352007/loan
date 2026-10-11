"""FastAPI application for the Loan Explainable AI demonstration."""
from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from backend.explanation_service import ExplanationService
from backend.gemini_service import GeminiExplanationService
from backend.model_service import ModelService
from backend.suggestion_service import SuggestionService
from backend.schemas import (
    ExplanationResponse,
    LoanApplicationInput,
    ModelInfoResponse,
    PredictionResponse,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
EVALUATION_PATH = ROOT_DIR / "reports" / "model_evaluation.json"
DEPENDENCY_PATH = ROOT_DIR / "reports" / "dependency_analysis.json"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "reports" / "feature_importance.csv"
APPLICATION_DATA_PATHS = (ROOT_DIR / "synthetic_loan_dataset (1).csv", ROOT_DIR / "data" / "loan_dataset.csv")

app = FastAPI(
    title="Loan Explainable AI",
    description=(
        "Educational demo for Gradient Boosting loan predictions, individual SHAP explanations, "
        "responsible suggestions, and feature-dependency analysis. Not a real bank decision system."
    ),
    version="1.0.0",
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _require_metadata() -> dict[str, Any]:
    metadata = _read_json(MODEL_METADATA_PATH)
    if not metadata or not metadata.get("feature_schema"):
        raise HTTPException(
            status_code=503,
            detail="Model metadata is missing or invalid. Run `python train_model.py` from the project root.",
        )
    return metadata


def _require_model_service() -> ModelService:
    try:
        return ModelService.get_instance()
    except Exception as exc:
        # Serialized-model incompatibilities can raise different exception types
        # across Python, NumPy, joblib, and scikit-learn versions.
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model could not be loaded. Rebuild the model with "
                "`python train_model.py` from the project root using this environment. "
                f"Details: {type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend files are missing.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Report artifact presence and whether the serialized model really loads."""
    model_error = None
    model_loaded = False
    try:
        ModelService.get_instance()
        model_loaded = True
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"

    dataset_present = (
        (ROOT_DIR / "synthetic_loan_dataset (1).csv").exists()
        or (ROOT_DIR / "data" / "loan_dataset.csv").exists()
    )
    artifact_present = (ROOT_DIR / "models" / "gradient_boosting_pipeline.pkl").exists()
    metadata_present = MODEL_METADATA_PATH.exists()
    return {
        "status": "ready" if model_loaded else "setup_required",
        "model_loaded": model_loaded,
        "model_error": model_error,
        "model_artifact_present": artifact_present,
        "metadata_present": metadata_present,
        "dataset_present": dataset_present,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "note": "A loaded model is not evidence of external validity. This project uses synthetic data.",
    }

@app.get("/api/schema")
def feature_schema() -> dict[str, Any]:
    """Expose the actual feature names and categorical options from training metadata."""
    metadata = _require_metadata()
    schema = metadata["feature_schema"]
    features = schema.get("features", [])
    numerical = set(schema.get("numerical_features", []))
    categorical = set(schema.get("categorical_features", []))
    options = schema.get("categorical_options", {})

    labels = {
        "no_of_dependents": "Number of Dependents",
        "education": "Education Level",
        "self_employed": "Self-employed",
        "income_annum": "Annual Income (₹)",
        "loan_amount": "Requested Loan Amount (₹)",
        "loan_term": "Loan Term (years)",
        "cibil_score": "CIBIL Score",
        "residential_assets_value": "Residential Assets Value (₹)",
        "commercial_assets_value": "Commercial Assets Value (₹)",
        "luxury_assets_value": "Luxury Assets Value (₹)",
        "bank_asset_value": "Bank Asset Value (₹)",
    }
    hints = {
        "no_of_dependents": {"min": 0, "max": 10, "step": 1, "default": 2},
        "education": {"default": "Graduate"},
        "self_employed": {"default": "No"},
        "income_annum": {"min": 1, "step": 1000, "default": 6000000},
        "loan_amount": {"min": 1, "step": 1000, "default": 15000000},
        "loan_term": {"min": 1, "max": 30, "step": 1, "default": 10},
        "cibil_score": {"min": 300, "max": 900, "step": 1, "default": 750},
        "residential_assets_value": {"min": 0, "step": 1000, "default": 12000000},
        "commercial_assets_value": {"min": 0, "step": 1000, "default": 4000000},
        "luxury_assets_value": {"min": 0, "step": 1000, "default": 18000000},
        "bank_asset_value": {"min": 0, "step": 1000, "default": 8000000},
    }

    fields = []
    for feature in features:
        field: dict[str, Any] = {
            "name": feature,
            "label": labels.get(feature, feature.replace("_", " ").title()),
            "kind": "number" if feature in numerical else "select" if feature in categorical else "text",
            **hints.get(feature, {}),
        }
        if feature in categorical:
            field["options"] = options.get(feature, [])
            if field.get("default") not in field["options"] and field["options"]:
                field["default"] = field["options"][0]
        fields.append(field)

    return {"features": fields, "target_mapping": metadata.get("target_mapping", {})}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(application: LoanApplicationInput) -> PredictionResponse:
    service = _require_model_service()
    return service.predict(application)


@app.post("/api/explain", response_model=ExplanationResponse)
def explain(application: LoanApplicationInput) -> ExplanationResponse:
    # Ensure artifact-loading failures are converted to helpful API errors.
    _require_model_service()
    try:
        return ExplanationService.get_instance().explain(application)
    except HTTPException:
        raise
    except Exception as exc:  # Return a useful server response rather than leaking a traceback.
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed. Check the installed package versions and model artifact. Details: {exc}",
        ) from exc



def _select_decision_factors(explanation: ExplanationResponse) -> list[dict[str, Any]]:
    """Return up to three SHAP factors that push in the model's predicted direction."""
    contributions = [item.model_dump() for item in explanation.feature_contributions]
    if explanation.is_approved:
        selected = [item for item in contributions if float(item.get("shap_value", 0)) > 0]
    else:
        selected = [item for item in contributions if float(item.get("shap_value", 0)) < 0]
    selected.sort(key=lambda item: abs(float(item.get("shap_value", 0))), reverse=True)
    if not selected:
        selected = sorted(
            contributions,
            key=lambda item: abs(float(item.get("shap_value", 0))),
            reverse=True,
        )
    return selected[:3]


@lru_cache(maxsize=1)
def _load_application_dataset() -> pd.DataFrame:
    dataset_path = next((path for path in APPLICATION_DATA_PATHS if path.exists()), None)
    if dataset_path is None:
        raise HTTPException(
            status_code=503,
            detail="The application dataset was not found. Expected synthetic_loan_dataset (1).csv in the project root or data/loan_dataset.csv.",
        )
    try:
        frame = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not read the application dataset: {exc}") from exc

    frame.columns = frame.columns.astype(str).str.strip()
    required = {"loan_id", "loan_status"}
    if not required.issubset(frame.columns):
        raise HTTPException(
            status_code=503,
            detail="The application dataset must contain loan_id and loan_status columns.",
        )
    frame["loan_status"] = frame["loan_status"].astype(str).str.strip().str.title()
    return frame


def _run_explanation(
    application: LoanApplicationInput,
    *,
    mode: str,
    recorded_decision: str | None = None,
) -> dict[str, Any]:
    model_service = _require_model_service()
    try:
        prediction = model_service.predict(application)
        explanation = ExplanationService.get_instance().explain(application)
        factors = _select_decision_factors(explanation)

        # The new-app workflow bases suggestions on the model estimate. The existing-record
        # workflow bases whether suggestions are shown on the stored dataset status.
        if mode == "application_lookup":
            rejected_for_suggestions = recorded_decision == "Rejected"
        else:
            rejected_for_suggestions = prediction.predicted_status == "Rejected"

        if rejected_for_suggestions:
            suggestions = SuggestionService.generate_suggestions(
                input_data=application,
                is_approved=False,
                top_negative_factors=[item["feature"] for item in factors if item.get("shap_value", 0) < 0],
                top_positive_factors=[item["feature"] for item in factors if item.get("shap_value", 0) > 0],
            )
        else:
            suggestions = []

        decision_matches_record = (
            prediction.predicted_status == recorded_decision
            if recorded_decision is not None
            else None
        )
        gemini = GeminiExplanationService.explain(
            mode="prediction" if mode == "prediction" else "application_lookup",
            model_prediction=prediction.predicted_status,
            approval_probability=prediction.approval_probability,
            decision_threshold=prediction.decision_threshold,
            factors=factors,
            safe_suggestions=suggestions,
            recorded_decision=recorded_decision,
            decision_matches_record=decision_matches_record,
        )
        return {
            "mode": mode,
            "prediction": prediction.model_dump(),
            "model_explanation": explanation.model_dump(),
            "recorded_decision": recorded_decision,
            "decision_matches_record": decision_matches_record,
            "model_prediction": prediction.predicted_status,
            "factors": factors,
            "gemini": gemini,
            "suggestions": gemini.get("suggestions", suggestions) if rejected_for_suggestions else [],
            "disclaimer": (
                "This project uses synthetic data. SHAP explains this Gradient Boosting model's estimate, "
                "not a real bank's internal reasoning. A dataset's stored decision is an outcome label, not an explanation."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate the explanation. Check the model and SHAP packages. Details: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/predict-and-explain")
def predict_and_explain(application: LoanApplicationInput) -> dict[str, Any]:
    """Predict a new application and explain the factors behind the model estimate."""
    return _run_explanation(application, mode="prediction")


@app.get("/api/application/{application_id}")
def explain_existing_application(application_id: int) -> dict[str, Any]:
    """Look up a stored application ID, then explain the model's view of that row."""
    if application_id < 1:
        raise HTTPException(status_code=400, detail="Application ID must be a positive integer.")

    frame = _load_application_dataset()
    if pd.api.types.is_numeric_dtype(frame["loan_id"]):
        matches = frame[pd.to_numeric(frame["loan_id"], errors="coerce") == application_id]
    else:
        matches = frame[frame["loan_id"].astype(str).str.strip() == str(application_id)]

    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Application ID {application_id} was not found in the checked-in dataset.")

    metadata = _require_metadata()
    row = matches.iloc[0]
    stored_status = str(row["loan_status"]).strip().title()
    if stored_status not in {"Approved", "Rejected"}:
        raise HTTPException(status_code=503, detail=f"Unsupported stored loan_status value: {stored_status!r}.")

    feature_names = metadata["feature_schema"]["features"]
    missing_features = [name for name in feature_names if name not in frame.columns]
    if missing_features:
        raise HTTPException(
            status_code=503,
            detail=f"The dataset is missing model feature columns: {', '.join(missing_features)}",
        )

    values: dict[str, Any] = {}
    for name in feature_names:
        value = row[name]
        values[name] = value.item() if hasattr(value, "item") else value
    try:
        application = LoanApplicationInput.model_validate(values)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Application ID {application_id} contains values that do not match the model input schema: {exc}",
        ) from exc

    result = _run_explanation(
        application,
        mode="application_lookup",
        recorded_decision=stored_status,
    )
    result["application_id"] = application_id
    return result


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metadata = _require_metadata()
    schema = metadata.get("feature_schema", {})
    return ModelInfoResponse(
        model_name=metadata.get("model_name", "GradientBoostingClassifier"),
        pipeline_steps=metadata.get("pipeline_steps", []),
        features=schema.get("features", []),
        numerical_features=schema.get("numerical_features", []),
        categorical_features=schema.get("categorical_features", []),
        categorical_options=schema.get("categorical_options", {}),
        target_mapping=metadata.get("target_mapping", {"0": "Rejected", "1": "Approved"}),
        selected_threshold=float(metadata.get("selected_threshold", 0.5)),
        metrics_summary=metadata.get("metrics_summary", {}),
        most_influential_feature=metadata.get("most_influential_feature", "Not available"),
        synthetic_disclaimer=metadata.get(
            "synthetic_disclaimer",
            "This is a synthetic-data educational demonstration, not a bank underwriting model.",
        ),
    )


@app.get("/api/dependency-analysis")
def dependency_analysis() -> dict[str, Any]:
    """Return dependency report together with global feature importance rows."""
    report = _read_json(DEPENDENCY_PATH)
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Dependency analysis report was not found. Run `python train_model.py` first.",
        )

    # The CSV is the source of the individual feature-level SHAP and permutation values.
    rows: list[dict[str, Any]] = []
    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            with FEATURE_IMPORTANCE_PATH.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    parsed: dict[str, Any] = {"feature": row.get("feature", "")}
                    for key in ("mean_abs_shap", "permutation_importance_mean", "permutation_importance_std"):
                        try:
                            parsed[key] = float(row[key]) if row.get(key) not in (None, "") else None
                        except (TypeError, ValueError):
                            parsed[key] = None
                    rows.append(parsed)
            rows.sort(
                key=lambda row: row.get("mean_abs_shap") if row.get("mean_abs_shap") is not None else float("-inf"),
                reverse=True,
            )
        except OSError:
            rows = []
    report["global_importance"] = rows
    return report

@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    report = _read_json(EVALUATION_PATH)
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Model evaluation report was not found. Run `python train_model.py` first.",
        )
    return report
