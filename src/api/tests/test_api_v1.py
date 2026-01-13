from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    models = r.json()
    assert isinstance(models, list)
    assert any(m["disease"] == "t2d" for m in models)


def test_predict_t2d_normal():
    payload = {
        "request_id": "unit-normal",
        "age_years": 25,
        "sex_at_birth": "female",
        "waist_circumference_cm": 70,
        "systolic_bp_mmHg": 110,
        "diastolic_bp_mmHg": 70,
        "fasting_glucose_mg_dL": 85,
        "triglycerides_mg_dL": 80,
        "hdl_mg_dL": 60,
        "total_cholesterol_mg_dL": 170,
        "hba1c_percent": 5.1,
        "height_cm": 165,
        "weight_kg": 55,
        "race_ethnicity": "unknown",
        "pregnancy_status": "unknown",
    }
    r = client.post("/v1/predict/t2d", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["disease"] == "t2d"
    assert "probabilities" in out
    probs = out["probabilities"]
    s = probs["p_normal"] + probs["p_prediabetes"] + probs["p_diabetes"]
    assert abs(s - 1.0) < 1e-6
    assert out["predicted_label"] in ["normal", "prediabetes", "diabetes"]


def test_predict_missing_required_field():
    payload = {"age_years": 40}  # missing required fields
    r = client.post("/v1/predict/t2d", json=payload)
    assert r.status_code == 422
