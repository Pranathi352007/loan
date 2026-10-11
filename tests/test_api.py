import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_homepage_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Loan Explainable AI" in response.text


def test_health_endpoint_shape():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "setup_required"}
    assert "model_artifact_present" in payload
    assert "metadata_present" in payload
    assert "gemini_configured" in payload
    assert "model_loaded" in payload
    assert "model_error" in payload


def test_schema_contains_expected_fields():
    response = client.get("/api/schema")
    # Metadata is tracked in this repository. If a user has not trained yet, the route
    # deliberately returns 503 and explains how to generate metadata.
    if response.status_code == 503:
        assert "train_model.py" in response.json()["detail"]
        return
    assert response.status_code == 200
    names = {field["name"] for field in response.json()["features"]}
    assert "loan_amount" in names
    assert "income_annum" in names
    assert "cibil_score" in names
    fields = {field["name"]: field for field in response.json()["features"]}
    assert fields["income_annum"]["step"] == 1
    assert fields["loan_amount"]["step"] == 1


def test_frontend_assets_are_served():
    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")
    assert css_response.status_code == 200
    assert ".field input" in css_response.text
    assert js_response.status_code == 200
    assert '/api/predict-and-explain' in js_response.text
    assert '/api/application/' in js_response.text
    html_response = client.get("/")
    assert "SHAP" not in html_response.text
    assert "Gradient Boosting" not in html_response.text
    assert "Gemini" not in html_response.text
    assert "SHAP" not in js_response.text
    assert "Gradient Boosting" not in js_response.text
    assert "Gemini" not in js_response.text
    assert "step: 1000" not in js_response.text
    assert "shap_value" not in js_response.text
    assert "approval_probability" not in js_response.text


def test_homepage_has_two_workflow_buttons():
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="new-application-button"' in response.text
    assert 'id="lookup-application-button"' in response.text


def test_application_lookup_returns_404_for_missing_id():
    response = client.get("/api/application/99999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_prediction_explanation_route_is_registered():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/predict-and-explain" in paths
    assert "/api/application/{application_id}" in paths



SAMPLE_APPLICATION = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 6000000,
    "loan_amount": 15000000,
    "loan_term": 10,
    "cibil_score": 750,
    "residential_assets_value": 12000000,
    "commercial_assets_value": 4000000,
    "luxury_assets_value": 18000000,
    "bank_asset_value": 8000000,
}


def _skip_if_serialized_model_cannot_load():
    health = client.get("/api/health").json()
    if not health.get("model_loaded"):
        pytest.skip("Bundled model artifact cannot load in this dependency environment; run train_model.py.")


def test_new_application_prediction_and_plain_language_flow():
    _skip_if_serialized_model_cannot_load()
    response = client.post("/api/predict-and-explain", json=SAMPLE_APPLICATION)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction"]["predicted_status"] in {"Approved", "Rejected"}
    assert "approval_probability" not in payload["prediction"]
    assert "model_explanation" not in payload
    assert payload["factors"]
    assert len(payload["factors"]) <= 3
    assert "overview" in payload["gemini"]
    for factor in payload["factors"]:
        assert {"feature", "display_name", "applicant_value", "reason"} <= set(factor)
        assert "shap_value" not in factor
    assert "shap_value" not in str(payload).lower()


def test_application_id_lookup_explains_existing_row():
    _skip_if_serialized_model_cannot_load()
    response = client.get("/api/application/1")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["application_id"] == 1
    assert payload["recorded_decision"] in {"Approved", "Rejected"}
    assert payload["model_prediction"] in {"Approved", "Rejected"}
    assert isinstance(payload["decision_matches_record"], bool)
    assert len(payload["factors"]) <= 3
    assert "model_explanation" not in payload
    assert "shap_value" not in str(payload).lower()
    for factor in payload["factors"]:
        assert {"feature", "display_name", "applicant_value", "reason"} <= set(factor)


def test_unknown_path_is_not_accidentally_served():
    response = client.get("/does-not-exist")
    assert response.status_code == 404
