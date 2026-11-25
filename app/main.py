
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="MP272 Tea Leaf Plucking API")


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


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    # Temporary mock response until model integration
    return {
        "message": "Input validation successful",
        "validated_input": data.model_dump()
    }
