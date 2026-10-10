"""FastAPI application for the Loan Explainable AI demonstration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.explanation_service import ExplanationService
from backend.model_service import ModelService
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
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model could not be loaded. Run `python train_model.py` in the project root "
                f"using the installed requirements. Details: {exc}"
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
    """Lightweight health check that does not load the model into memory."""
    return {
        "status": "ready" if MODEL_METADATA_PATH.exists() else "setup_required",
        "model_artifact_present": (ROOT_DIR / "models" / "gradient_boosting_pipeline.pkl").exists(),
        "metadata_present": MODEL_METADATA_PATH.exists(),
        "dataset_present": (ROOT_DIR / "synthetic_loan_dataset (1).csv").exists()
        or (ROOT_DIR / "data" / "loan_dataset.csv").exists(),
        "note": "A ready response does not mean the model has been externally validated.",
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
        "income_annum": {"min": 0, "step": 1000, "default": 6000000},
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
    report = _read_json(DEPENDENCY_PATH)
    if not report:
        raise HTTPException(
            status_code=404,
            detail="Dependency analysis report was not found. Run `python train_model.py` first.",
        )
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
