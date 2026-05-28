"""
Swiss ED Predictor — FastAPI Backend
REST API for ED admission predictions.

MVP: Basic endpoints + JWT auth scaffold
V2: Full RBAC, multi-canton, FHIR integration
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import numpy as np
from loguru import logger

app = FastAPI(
    title="Swiss ED Predictor API",
    description="Predicts emergency department admission peaks using Swiss open data.",
    version="0.1.0-mvp",
    contact={"name": "Albert Deutou Ngodji", "email": "a.deutou@gmail.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# ── Schemas ──────────────────────────────────────────────

class PredictionRequest(BaseModel):
    canton: str = "BE"
    hospital_id: Optional[str] = None
    horizon_hours: int = 24  # 24, 48, or 72

    model_config = {"json_schema_extra": {"example": {
        "canton": "BE",
        "hospital_id": "INSEL",
        "horizon_hours": 48,
    }}}


class PredictionResponse(BaseModel):
    canton: str
    hospital_id: Optional[str]
    horizon_hours: int
    predicted_admissions: int
    confidence: float
    alert_level: str  # LOW | MEDIUM | HIGH
    generated_at: datetime
    model_version: str
    data_sources: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    timestamp: datetime


# ── Helpers ──────────────────────────────────────────────

def _alert_level(admissions: int) -> str:
    if admissions > 75:
        return "HIGH"
    elif admissions > 55:
        return "MEDIUM"
    return "LOW"


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    MVP: stub auth — always passes.
    V2: validate JWT, check canton-level RBAC.
    """
    # TODO V2: validate JWT token against canton roles
    return True


# ── Endpoints ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0-mvp",
        model_loaded=True,  # TODO: check actual model file
        timestamp=datetime.utcnow(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    req: PredictionRequest,
    _auth: bool = Depends(_verify_token),
):
    """
    Predict ED admissions for a given canton and time horizon.

    - **canton**: Swiss canton code (BE, ZH, GE, VD, ...)
    - **horizon_hours**: Prediction horizon in hours (24, 48, 72)
    - **hospital_id**: Optional specific hospital identifier
    """
    logger.info(f"Prediction request: canton={req.canton} horizon={req.horizon_hours}h")

    if req.horizon_hours not in [24, 48, 72]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="horizon_hours must be 24, 48, or 72",
        )

    # TODO: replace with real model inference
    # from src.model.train import predict, load_model
    # from src.features.engineering import build_feature_matrix
    # features = build_feature_matrix(weather_df, transport_df, patterns, datetime.now(), req.canton)
    # admissions = int(predict(features)[0])

    # MVP stub — simulated prediction
    base = {"BE": 58, "ZH": 72, "GE": 65, "VD": 61}.get(req.canton, 55)
    predicted = base + int(np.random.randint(-8, 15))
    confidence = round(np.random.uniform(0.82, 0.93), 3)

    return PredictionResponse(
        canton=req.canton,
        hospital_id=req.hospital_id,
        horizon_hours=req.horizon_hours,
        predicted_admissions=predicted,
        confidence=confidence,
        alert_level=_alert_level(predicted),
        generated_at=datetime.utcnow(),
        model_version="xgboost-mvp-v0.1",
        data_sources=["SpiGes/OFSP", "MétéoSuisse", "opentransportdata.swiss", "OFS"],
    )


@app.get("/cantons", tags=["Reference"])
async def list_cantons():
    """Return list of supported cantons."""
    return {
        "cantons": [
            {"code": "BE", "name": "Bern"},
            {"code": "ZH", "name": "Zürich"},
            {"code": "GE", "name": "Genève"},
            {"code": "VD", "name": "Vaud"},
            {"code": "BS", "name": "Basel-Stadt"},
            {"code": "AG", "name": "Aargau"},
            {"code": "SG", "name": "St. Gallen"},
            {"code": "TI", "name": "Ticino"},
            {"code": "VS", "name": "Valais"},
            {"code": "NE", "name": "Neuchâtel"},
        ]
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return current model metadata."""
    return {
        "model_type": "XGBoost Regressor",
        "version": "0.1.0-mvp",
        "features": 18,
        "training_data": "SpiGes 2018-2023",
        "cv_strategy": "TimeSeriesSplit (5 folds)",
        "reference_mae": 4.0,
        "reference": "King et al., Nature npj Digital Medicine 2022",
        "roadmap_v2": "LSTM + MLflow tracking",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
