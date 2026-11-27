
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.mp272.tracking import log_experiment

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "v2"


def test_prediction_endpoint():
    payload = {
        "section": "Upper Garden",
        "rain_mm_week": 20.0,
        "slope_deg": 15.0,
        "temp_c": 25.0,
        "bush_age_yr": 50.0,
        "hours_worked": 8.0,
        "worker_exp_yr": 5.0,
        "flush": "First"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["unit"] == "kg_per_worker"
    assert response.json()["model_version"] == "v2"


def test_prediction_rejects_invalid_temperature():
    payload = {
        "section": "Upper Garden",
        "rain_mm_week": 20.0,
        "slope_deg": 15.0,
        "temp_c": 2.0,
        "bush_age_yr": 50.0,
        "hours_worked": 8.0,
        "worker_exp_yr": 5.0,
        "flush": "First"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_tracking_rejects_invalid_hash():
    with pytest.raises(ValueError, match="SHA-256"):
        log_experiment(
            model=None,
            params={},
            metrics={"mae": 2.5},
            data_sha256="dummy-test-hash"
        )


def test_tracking_requires_mae():
    with pytest.raises(ValueError, match="MAE"):
        log_experiment(
            model=None,
            params={},
            metrics={"accuracy": 0.90},
            data_sha256="a" * 64
        )
