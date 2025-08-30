import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List

# Predict helpers
from model.predict import predict_game_success

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"
lgb_trained = joblib.load(ARTIFACTS_DIR / "lgb_model.pkl")
features_used: List[str] = joblib.load(ARTIFACTS_DIR / "features_used.pkl")
known_dev_cols = [c for c in features_used if c.startswith("dev__")]
known_pub_cols = [c for c in features_used if c.startswith("pub__")]


# Schemas
class GameInput(BaseModel):
    price: float
    is_free: bool
    required_age: int
    achievements: int
    english: bool

    windows: bool
    mac: bool
    linux: bool

    release_date: str = Field(..., description="ISO date, e.g. 2019-01-24")
    extract_date: str = Field(..., description="ISO date, e.g. 2025-02-20")

    developer: str
    publisher: str
    publisherClass_encoded: int

    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    owners: int
    players: int
    copiesSold: int
    revenue: float


# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the artifacts once
try:
    lgb_trained = joblib.load(ARTIFACTS_DIR / "lgb_model.pkl")
    features_used: List[str] = joblib.load(ARTIFACTS_DIR / "features_used.pkl")
    known_dev_cols = [c for c in features_used if c.startswith("dev__")]
    known_pub_cols = [c for c in features_used if c.startswith("pub__")]
except Exception as e:
    raise RuntimeError(f"Failed to load artifacts: {e}")


# Health check
@app.get("/api/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/api/predict", response_model=PredictionResponse)
def predict_game(input_data: GameInput):
    """
    Accepts a full game payload and returns predictions.
    """
    try:
        # Turn the BaseModel into a plain dict (includes extra fields)
        # payload: Dict[str, Any] = input_data.dict()
        payload: Dict[str, Any] = input_data.model_dump()

        preds = predict_game_success(
            user_input=payload,
            model=lgb_trained,
            features_used=features_used,
            known_devs=known_dev_cols,
            known_pubs=known_pub_cols,
        )
        return PredictionResponse(**preds)
    except KeyError as ke:
        # Often caused by a missing key used in preprocessing
        raise HTTPException(status_code=400, detail=f"Missing or invalid field: {ke}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# # /api/home
# @app.get("/api/home")
# def return_home():
#     return {"message": "Hello from FastAPI!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
