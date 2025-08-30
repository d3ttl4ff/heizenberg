# import numpy as np
# import pandas as pd
# from pathlib import Path
# import joblib
# from typing import Dict, List
# from tabulate import tabulate

# target_cols = ["owners", "players", "copiesSold", "revenue"]
# post_release = ["wishlists", "avgPlaytime", "followers", "reviews", "reviewScore"]

# # BASE_DIR = Path(__file__).resolve().parent
# # ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"
# # lgb_trained = joblib.load(ARTIFACTS_DIR / "lgb_model.pkl")
# # features_used = joblib.load(ARTIFACTS_DIR / "features_used.pkl")

# PLATFORM_FLAGS = ["windows", "mac", "linux"]
# TAG_FLAGS = [
#     "Single-player",
#     "Family Sharing",
#     "Steam Achievements",
#     "Steam Cloud",
#     "Full controller support",
#     "Multi-player",
#     "Partial Controller Support",
#     "Steam Trading Cards",
#     "PvP",
#     "Co-op",
#     "Steam Leaderboards",
#     "Remote Play Together",
#     "Online PvP",
#     "Shared/Split Screen",
#     "Tracked Controller Support",
#     "VR Only",
#     "Shared/Split Screen PvP",
#     "Online Co-op",
#     "Stats",
#     "Shared/Split Screen Co-op",
# ]
# GENRE_FLAGS = [
#     "Indie",
#     "Casual",
#     "Adventure",
#     "Action",
#     "Simulation",
#     "Strategy",
#     "RPG",
#     "Free To Play",
#     "Sports",
#     "Racing",
# ]


# def predict_game_success(
#     user_input: Dict,
#     model,
#     features_used: List[str],
# ):
#     # 1) Base vector (zeros)
#     input_data = {feature: 0 for feature in features_used}

#     # 2) Direct fields
#     input_data["price"] = user_input.get("price", 0)
#     input_data["is_free"] = int(user_input.get("is_free", False))
#     input_data["required_age"] = user_input.get("required_age", 0)
#     input_data["achievements"] = user_input.get("achievements", 0)
#     input_data["english"] = int(user_input.get("english", True))

#     # 3) Flags
#     for flag in PLATFORM_FLAGS + TAG_FLAGS + GENRE_FLAGS:
#         input_data[flag] = int(user_input.get(flag, False))

#     # 4) Publisher class
#     input_data["publisherClass_encoded"] = user_input.get("publisherClass_encoded", 0)

#     # 5) Derived fields
#     release_date = pd.to_datetime(user_input["release_date"])
#     extract_date = pd.to_datetime(user_input["extract_date"])
#     input_data["days_since_release"] = (extract_date - release_date).days

#     # 7) DF in correct column order
#     input_df = pd.DataFrame([input_data])[features_used]

#     # 8) Predict
#     y_pred_log = model.predict(input_df)
#     y_pred = np.expm1(y_pred_log)

#     return {
#         "owners": int(y_pred[0][0]),
#         "players": int(y_pred[0][1]),
#         "copiesSold": int(y_pred[0][2]),
#         "revenue": float(y_pred[0][3]),
#     }


# ENSEMBLE
# model/predict.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# ===== Toggle (hardcoded) =====
# Set True to use the new per-target (optionally ensemble) pipeline.
USE_NEW_PIPELINE = os.getenv("USE_NEW_PIPELINE", "true").lower() in {
    "1",
    "true",
    "yes",
}

TARGET_COLS = ["owners", "players", "copiesSold", "revenue"]

PLATFORM_FLAGS = ["windows", "mac", "linux"]
TAG_FLAGS = [
    "Single-player",
    "Family Sharing",
    "Steam Achievements",
    "Steam Cloud",
    "Full controller support",
    "Multi-player",
    "Partial Controller Support",
    "Steam Trading Cards",
    "PvP",
    "Co-op",
    "Steam Leaderboards",
    "Remote Play Together",
    "Online PvP",
    "Shared/Split Screen",
    "Tracked Controller Support",
    "VR Only",
    "Shared/Split Screen PvP",
    "Online Co-op",
    "Stats",
    "Shared/Split Screen Co-op",
]
GENRE_FLAGS = [
    "Indie",
    "Casual",
    "Adventure",
    "Action",
    "Simulation",
    "Strategy",
    "RPG",
    "Free To Play",
    "Sports",
    "Racing",
]


def _build_input_df(
    user_input: Dict[str, Any], features_used: List[str]
) -> pd.DataFrame:
    """1-row DF in the exact training feature order (zeros by default)."""
    x = {f: 0 for f in features_used}

    # Core fields
    x["price"] = user_input.get("price", 0)
    x["is_free"] = int(user_input.get("is_free", False))
    x["required_age"] = user_input.get("required_age", 0)
    x["achievements"] = user_input.get("achievements", 0)
    x["english"] = int(user_input.get("english", True))

    # Flags that exist in features
    for flag in PLATFORM_FLAGS + TAG_FLAGS + GENRE_FLAGS:
        if flag in x:
            x[flag] = int(user_input.get(flag, False))

    # Encoded publisher
    if "publisherClass_encoded" in x:
        x["publisherClass_encoded"] = user_input.get("publisherClass_encoded", 0)

    # days_since_release if present
    if (
        "days_since_release" in x
        and "release_date" in user_input
        and "extract_date" in user_input
    ):
        rd = pd.to_datetime(user_input["release_date"])
        ed = pd.to_datetime(user_input["extract_date"])
        x["days_since_release"] = (ed - rd).days

    return pd.DataFrame([x])[features_used]


# ------------------------------
# OLD pipeline (single model)
# ------------------------------
def _predict_single_model(
    user_input: Dict, model, features_used: List[str]
) -> Dict[str, Any]:
    X1 = _build_input_df(user_input, features_used)
    # single multi-output model returns log-space predictions (your trainer uses log1p)
    y_log = model.predict(X1)
    y = np.expm1(y_log)
    return {
        "owners": int(y[0][0]),
        "players": int(y[0][1]),
        "copiesSold": int(y[0][2]),
        "revenue": float(y[0][3]),
    }


# ------------------------------
# NEW pipeline (per-target models + weights)
# Expected artifacts:
#   per_target_models: {'lgb': {target: est}, 'xgb': {...}, 'cb': {...}}
#   ensemble_weights:  {target: {'lgb': w, 'xgb': w, 'cb': w}}
# ------------------------------
def _predict_ensemble(
    user_input: Dict,
    per_target_models: Dict[str, Dict[str, Any]],
    ensemble_weights: Dict[str, Dict[str, float]],
    features_used: List[str],
) -> Dict[str, Any]:
    X1 = _build_input_df(user_input, features_used)
    out = {}
    for t in TARGET_COLS:
        blend = 0.0
        w = ensemble_weights[t]
        for alg_name, fam in per_target_models.items():
            est = fam[t]
            yhat_log = float(est.predict(X1)[0])
            yhat = float(np.expm1(yhat_log))
            blend += w[alg_name] * yhat
        out[t] = float(blend) if t == "revenue" else int(round(blend))
    return out


# ------------------------------
# Public dispatcher
# ------------------------------
def predict_game_success(
    user_input: Dict,
    model_or_models,  # either single estimator OR dict of per-target models
    features_used: List[str],
    *,
    use_new_pipeline: bool = None,
    ensemble_weights: Dict[str, Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    If use_new_pipeline == False -> use old single multi-output model.
    If use_new_pipeline == True  -> require per_target_models dict (+ weights) and run weighted ensemble.
    If use_new_pipeline is None  -> fall back to module-level USE_NEW_PIPELINE.
    """
    if use_new_pipeline is None:
        use_new_pipeline = USE_NEW_PIPELINE

    if not use_new_pipeline:
        # old path
        return _predict_single_model(user_input, model_or_models, features_used)

    # new path (ensemble)
    if not isinstance(model_or_models, dict):
        raise ValueError(
            "New pipeline expects per_target_models dict, got a non-dict model."
        )
    if ensemble_weights is None:
        raise ValueError("New pipeline expects ensemble_weights dict.")
    return _predict_ensemble(
        user_input, model_or_models, ensemble_weights, features_used
    )
