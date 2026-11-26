"""Tests for the MP272 prediction API (app/main.py).

Run from the repo root:
    python3 -m pytest -v
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app, get_model

client = TestClient(app)

VALID_INPUT = {
    "section": "Upper Garden",
    "rain_mm_week": 45.0,
    "slope_deg": 12.0,
    "temp_c": 24.0,
    "bush_age_yr": 30.0,
    "hours_worked": 7.5,
    "worker_exp_yr": 8.0,
    "flush": "first",
}

V1_FEATURES = ["worker_exp_yr", "rain_mm_week", "slope_deg", "temp_c",
               "bush_age_yr", "hours_worked", "section"]


class FakeModel:
    """Stands in for the real model: records its input, returns 25.0."""

    feature_names_in_ = np.array(V1_FEATURES, dtype=object)

    def __init__(self):
        self.last_input = None

    def predict(self, X):
        self.last_input = X
        return np.array([25.0])


@pytest.fixture
def fake_model():
    model = FakeModel()
    app.dependency_overrides[get_model] = lambda: model
    yield model
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# --------------------------------------------------------------------------
# Valid requests
# --------------------------------------------------------------------------
def test_predict_valid_request_returns_json(fake_model):
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["prediction"] == 25.0
    assert body["target"] == "kg_per_worker"
    assert body["model"] == "model.joblib"


def test_model_receives_only_its_training_columns(fake_model):
    client.post("/predict", json=VALID_INPUT)

    sent = fake_model.last_input
    assert list(sent.columns) == V1_FEATURES      # `flush` dropped for v1
    assert sent.iloc[0]["section"] == "Upper Garden"
    assert sent.iloc[0]["hours_worked"] == 7.5


def test_sentinel_experience_is_treated_as_missing(fake_model):
    payload = {**VALID_INPUT, "worker_exp_yr": 9999}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert np.isnan(fake_model.last_input.iloc[0]["worker_exp_yr"])


@pytest.mark.skipif(not main.MODEL_PATH.exists(),
                    reason="models/model.joblib not present")
def test_predict_with_real_champion_model():
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert isinstance(prediction, float)
    assert np.isfinite(prediction)


# --------------------------------------------------------------------------
# Invalid requests (rejected by validation with 422)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", list(VALID_INPUT))
def test_missing_field_is_rejected(fake_model, field):
    payload = {k: v for k, v in VALID_INPUT.items() if k != field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_temperature_below_minimum_is_rejected(fake_model):
    response = client.post("/predict", json={**VALID_INPUT, "temp_c": 2})
    assert response.status_code == 422


def test_bush_age_above_maximum_is_rejected(fake_model):
    response = client.post("/predict",
                           json={**VALID_INPUT, "bush_age_yr": 200})
    assert response.status_code == 422


def test_wrong_type_is_rejected(fake_model):
    response = client.post("/predict",
                           json={**VALID_INPUT, "rain_mm_week": "heavy"})
    assert response.status_code == 422


def test_empty_body_is_rejected(fake_model):
    response = client.post("/predict", json={})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Missing model file
# --------------------------------------------------------------------------
def test_missing_model_file_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "model.joblib")
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 503
