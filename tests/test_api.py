from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_homepage_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "LoanLens" in response.text


def test_health_endpoint_shape():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "setup_required"}
    assert "model_artifact_present" in payload
    assert "metadata_present" in payload


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


def test_unknown_path_is_not_accidentally_served():
    response = client.get("/does-not-exist")
    assert response.status_code == 404
