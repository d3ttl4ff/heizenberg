import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any, List
import joblib

# universal predictor you already have
from model.predict import predict_game_success

# ---------------------------------------------------------------------
# Paths / Artifacts
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"

# Model file
MODEL_FILE = os.getenv("MODEL_FILE", "cb_model.pkl")
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILE

FEATURES_FILE = ARTIFACTS_DIR / "features_used.pkl"


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_features_used() -> List[str]:
    if FEATURES_FILE.exists():
        return joblib.load(FEATURES_FILE)
    raise FileNotFoundError(
        "Could not load features list. Ensure features_used.pkl exists."
    )


try:
    MODEL_OBJECT = load_model()
    FEATURES_USED: List[str] = load_features_used()
except Exception as e:
    raise RuntimeError(f"Failed to load artifacts: {e}")


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class GameInput(BaseModel):
    # Core numeric/boolean inputs
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

    # keep if used in training
    publisherClass_encoded: int = 0

    # Allow extra boolean flags like 'Single_player', 'Action', etc.
    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    owners: int
    players: int
    copiesSold: int
    revenue: float


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_file": MODEL_FILE,
        "features_count": len(FEATURES_USED),
        "estimator_type": (
            "dict" if isinstance(MODEL_OBJECT, dict) else type(MODEL_OBJECT).__name__
        ),
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_game(input_data: GameInput):
    """
    Accepts a full game payload and returns predictions.
    Any additional boolean flags (e.g., 'Single_player', 'Action') can be included in the JSON body.
    """
    try:
        payload: Dict[str, Any] = input_data.model_dump()

        preds = predict_game_success(
            user_input=payload,
            model=MODEL_OBJECT,  # <-- fixed: use 'model', not 'estimator'
            features_used=FEATURES_USED,
        )
        return PredictionResponse(**preds)

    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing or invalid field: {ke}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: override model filename via env: MODEL_FILE=catboost_per_target_models.pkl
    uvicorn.run(app, host="0.0.0.0", port=8000)
