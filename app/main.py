from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MP272 Tea Leaf Plucking API")

# Champion model location. Override with the MODEL_PATH environment
# variable (e.g. inside Docker / Kubernetes).
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))

TARGET = "kg_per_worker"
WORKER_EXP_SENTINEL = 9999   # "unknown" code in the raw data -> missing


# Pydantic schema: all 8 fields are required
class PredictionInput(BaseModel):
    section: str
    rain_mm_week: float
    slope_deg: float
    temp_c: float = Field(ge=5)
    bush_age_yr: float = Field(le=150)
    hours_worked: float
    worker_exp_yr: float
    flush: str


class PredictionOutput(BaseModel):
    prediction: float
    target: str
    model: str


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model(path: str):
    """Load the champion model once and keep it in memory."""
    return joblib.load(path)


def get_model():
    """FastAPI dependency: the loaded champion model (503 if missing)."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model file not found: {MODEL_PATH.name}. "
                   "Train the model and save it to models/model.joblib.",
        )
    return load_model(str(MODEL_PATH))


def make_prediction(model, data: dict) -> float:
    """Turn one validated request into a model prediction."""
    row = pd.DataFrame([data])

    # Same treatment as training: 9999 means "unknown experience".
    row["worker_exp_yr"] = row["worker_exp_yr"].replace(
        WORKER_EXP_SENTINEL, np.nan)

    # Give the model exactly the columns it was trained on (v1 does not
    # use `flush`; a later model may).
    columns = getattr(model, "feature_names_in_", None)
    if columns is not None:
        row = row[list(columns)]

    return float(model.predict(row)[0])


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.exists(),
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, model=Depends(get_model)):
    try:
        value = make_prediction(model, data.model_dump())
    except Exception as exc:  # e.g. model/feature mismatch
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {exc}") from exc

    return PredictionOutput(
        prediction=round(value, 4),
        target=TARGET,
        model=MODEL_PATH.name,
    )
