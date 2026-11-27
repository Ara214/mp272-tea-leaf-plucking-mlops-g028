import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.mp272.logging_config import (
    get_logger,
    log_prediction,
    log_prediction_error,
)
from src.mp272.train import clean

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_v1.joblib"

app = FastAPI(title="MP272 Tea Leaf Plucking API")
logger = get_logger("mp272.api")

MODEL_VERSION = "v2"
model = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    section: str
    rain_mm_week: float
    slope_deg: float
    temp_c: float = Field(ge=5)
    bush_age_yr: float = Field(le=150)
    hours_worked: float
    worker_exp_yr: float
    flush: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    start_time = time.perf_counter()

    try:
        input_data = data.model_dump()

        # Apply the same unit and sentinel cleaning used during training.
        input_df = pd.DataFrame([input_data])
        cleaned_df = clean(input_df)

        # Predict using the trained V1 pipeline.
        prediction = float(model.predict(cleaned_df)[0])

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000, 3
        )

        log_prediction(
            logger,
            model_version=MODEL_VERSION,
            status="success",
            duration_ms=duration_ms,
        )

        return {
            "prediction": prediction,
            "unit": "kg_per_worker",
            "model_version": MODEL_VERSION,
        }

    except Exception as error:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000, 3
        )

        log_prediction_error(
            logger,
            error=error,
            model_version=MODEL_VERSION,
        )
        logger.error(
            "Prediction processing duration: %.3f ms",
            duration_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from error
