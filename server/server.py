# import os
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from pathlib import Path
# from typing import Dict, Any, List
# import joblib

# # universal predictor you already have
# from model.predict import predict_game_success

# # ---------------------------------------------------------------------
# # Paths / Artifacts
# # ---------------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent
# ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"

# # Model file
# MODEL_FILE = os.getenv("MODEL_FILE", "cb_model.pkl")
# MODEL_PATH = ARTIFACTS_DIR / MODEL_FILE

# FEATURES_FILE = ARTIFACTS_DIR / "features_used.pkl"


# def load_model() -> Any:
#     if not MODEL_PATH.exists():
#         raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
#     return joblib.load(MODEL_PATH)


# def load_features_used() -> List[str]:
#     if FEATURES_FILE.exists():
#         return joblib.load(FEATURES_FILE)
#     raise FileNotFoundError(
#         "Could not load features list. Ensure features_used.pkl exists."
#     )


# try:
#     MODEL_OBJECT = load_model()
#     FEATURES_USED: List[str] = load_features_used()
# except Exception as e:
#     raise RuntimeError(f"Failed to load artifacts: {e}")


# # ---------------------------------------------------------------------
# # Schemas
# # ---------------------------------------------------------------------
# class GameInput(BaseModel):
#     # Core numeric/boolean inputs
#     price: float
#     is_free: bool
#     required_age: int
#     achievements: int
#     english: bool

#     windows: bool
#     mac: bool
#     linux: bool

#     release_date: str = Field(..., description="ISO date, e.g. 2019-01-24")
#     extract_date: str = Field(..., description="ISO date, e.g. 2025-02-20")

#     # keep if used in training
#     publisherClass_encoded: int = 0

#     # Allow extra boolean flags like 'Single_player', 'Action', etc.
#     class Config:
#         extra = "allow"


# class PredictionResponse(BaseModel):
#     owners: int
#     players: int
#     copiesSold: int
#     revenue: float


# # ---------------------------------------------------------------------
# # App
# # ---------------------------------------------------------------------
# app = FastAPI()

# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------------------------------------------------------------------
# # Routes
# # ---------------------------------------------------------------------
# @app.get("/api/health")
# def health():
#     return {
#         "status": "ok",
#         "model_file": MODEL_FILE,
#         "features_count": len(FEATURES_USED),
#         "estimator_type": (
#             "dict" if isinstance(MODEL_OBJECT, dict) else type(MODEL_OBJECT).__name__
#         ),
#     }


# @app.post("/api/predict", response_model=PredictionResponse)
# def predict_game(input_data: GameInput):
#     """
#     Accepts a full game payload and returns predictions.
#     Any additional boolean flags (e.g., 'Single_player', 'Action') can be included in the JSON body.
#     """
#     try:
#         payload: Dict[str, Any] = input_data.model_dump()

#         preds = predict_game_success(
#             user_input=payload,
#             model=MODEL_OBJECT,  # <-- fixed: use 'model', not 'estimator'
#             features_used=FEATURES_USED,
#         )
#         return PredictionResponse(**preds)

#     except KeyError as ke:
#         raise HTTPException(status_code=400, detail=f"Missing or invalid field: {ke}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# # ---------------------------------------------------------------------
# # Entrypoint
# # ---------------------------------------------------------------------
# if __name__ == "__main__":
#     # Optional: override model filename via env: MODEL_FILE=catboost_per_target_models.pkl
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# ENSEMBLE
# server.py (your FastAPI file)
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
from typing import Union
import pandas as pd

from model.xai import explain_prediction_ensemble
from model.predict import predict_game_success  # dispatcher with toggle

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"
ARTIFACTS_DIR_ENSEMBLE = BASE_DIR / "model" / "artifacts_ensemble"

# ===== Toggle (hardcoded or env var) =====
USE_NEW_PIPELINE = os.getenv("USE_NEW_PIPELINE", "true").lower() in {
    "1",
    "true",
    "yes",
}

# Old (single model) artifacts
MODEL_FILE = os.getenv("MODEL_FILE", "cb_model.pkl")
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILE
FEATURES_FILE_OLD = ARTIFACTS_DIR / "features_used.pkl"

# New (ensemble) artifacts
PER_TARGET_MODELS_PATH = ARTIFACTS_DIR_ENSEMBLE / "per_target_models.pkl"
ENSEMBLE_WEIGHTS_PATH = ARTIFACTS_DIR_ENSEMBLE / "ensemble_weights.pkl"
FEATURES_FILE_NEW = ARTIFACTS_DIR_ENSEMBLE / "feature_order.pkl"


def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def load_artifacts():
    """
    Returns a dict with keys depending on pipeline:
      Old: {"mode":"old", "model":..., "features": [...]}
      New: {"mode":"new", "per_target_models":..., "ensemble_weights":..., "features":[...]}
    """
    if USE_NEW_PIPELINE:
        if not _exists(PER_TARGET_MODELS_PATH):
            raise FileNotFoundError(
                f"Missing per-target models: {PER_TARGET_MODELS_PATH}"
            )
        if not _exists(ENSEMBLE_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Missing ensemble weights: {ENSEMBLE_WEIGHTS_PATH}"
            )
        if not _exists(FEATURES_FILE_NEW):
            raise FileNotFoundError(f"Missing feature order: {FEATURES_FILE_NEW}")

        per_target_models = joblib.load(PER_TARGET_MODELS_PATH)
        ensemble_weights = joblib.load(ENSEMBLE_WEIGHTS_PATH)
        features_used = joblib.load(FEATURES_FILE_NEW)

        return {
            "mode": "new",
            "per_target_models": per_target_models,
            "ensemble_weights": ensemble_weights,
            "features": list(features_used),
        }

    # old
    if not _exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not _exists(FEATURES_FILE_OLD):
        raise FileNotFoundError(f"Missing features_used.pkl: {FEATURES_FILE_OLD}")

    model = joblib.load(MODEL_PATH)
    features_used = joblib.load(FEATURES_FILE_OLD)
    return {"mode": "old", "model": model, "features": list(features_used)}


ART = load_artifacts()

XAI_BG_PATH = ARTIFACTS_DIR_ENSEMBLE / "xai_background.pkl"
XAI_BACKGROUND = joblib.load(XAI_BG_PATH) if XAI_BG_PATH.exists() else None


# ---------- Helpers ----------
def _df_to_contrib_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "feature": str(r["feature"]),
                "value": (
                    None
                    if pd.isna(r["value"])
                    else (
                        bool(r["value"])
                        if r["value"] in (0, 1) and str(r["value"]).isdigit()
                        else r["value"]
                    )
                ),
                "contrib_normal": float(r["contrib_normal"]),
                "abs_contrib": float(r["abs_contrib"]),
            }
        )
    return out


# ---------- Schemas ----------
class GameInput(BaseModel):
    price: float
    is_free: bool
    required_age: int
    achievements: int
    english: bool

    windows: bool
    mac: bool
    linux: bool

    release_date: str = Field(..., description="YYYY-MM-DD")
    extract_date: str = Field(..., description="YYYY-MM-DD")

    publisherClass_encoded: int = 0

    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    owners: int
    players: int
    copiesSold: int
    revenue: float


class FeatureContribution(BaseModel):
    feature: str
    value: Union[int, float, bool, str, None] = None
    contrib_normal: float
    abs_contrib: float


class ExplainPayload(BaseModel):
    price: float
    is_free: bool
    required_age: int
    achievements: int
    english: bool
    windows: bool
    mac: bool
    linux: bool
    release_date: str
    extract_date: str
    publisherClass_encoded: int = 0

    class Config:
        extra = "allow"


class ExplainResponse(BaseModel):
    predictions: PredictionResponse
    xai: Dict[str, List[FeatureContribution]]


# ---------- App ----------
app = FastAPI()

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    base = {
        "status": "ok",
        "mode": ART["mode"],
        "features_count": len(ART["features"]),
        "use_new_pipeline": USE_NEW_PIPELINE,
    }
    if ART["mode"] == "old":
        base["model_file"] = MODEL_FILE
        base["estimator_type"] = type(ART["model"]).__name__
    else:
        base["per_target_families"] = list(ART["per_target_models"].keys())
        base["weights_targets"] = list(ART["ensemble_weights"].keys())
    return base


@app.post("/api/predict", response_model=PredictionResponse)
def predict_game(input_data: GameInput):
    try:
        payload: Dict[str, Any] = input_data.model_dump()

        if ART["mode"] == "old":
            preds = predict_game_success(
                user_input=payload,
                model_or_models=ART["model"],
                features_used=ART["features"],
                use_new_pipeline=False,
            )
        else:
            preds = predict_game_success(
                user_input=payload,
                model_or_models=ART["per_target_models"],
                features_used=ART["features"],
                use_new_pipeline=True,
                ensemble_weights=ART["ensemble_weights"],
            )

        return PredictionResponse(**preds)

    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing or invalid field: {ke}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/explain", response_model=ExplainResponse)
def explain_game(input_data: ExplainPayload):
    if ART["mode"] != "new":
        raise HTTPException(
            status_code=400, detail="Explanations require the per-target ensemble mode."
        )
    if XAI_BACKGROUND is None:
        raise HTTPException(
            status_code=500,
            detail="Missing xai_background.pkl. Retrain to generate it.",
        )

    payload: Dict[str, Any] = input_data.model_dump()

    preds = predict_game_success(
        user_input=payload,
        model_or_models=ART["per_target_models"],
        features_used=ART["features"],
        use_new_pipeline=True,
        ensemble_weights=ART["ensemble_weights"],
    )

    xai_top = explain_prediction_ensemble(
        user_input=payload,
        per_target_models=ART["per_target_models"],
        ensemble_weights=ART["ensemble_weights"],
        features_used=ART["features"],
        background_df=XAI_BACKGROUND,
        top_k=8,
    )

    xai_json = {t: _df_to_contrib_list(df) for t, df in xai_top.items()}

    return ExplainResponse(predictions=PredictionResponse(**preds), xai=xai_json)


if __name__ == "__main__":
    # Set USE_NEW_PIPELINE via env or leave hardcoded
    uvicorn.run(app, host="0.0.0.0", port=8000)
