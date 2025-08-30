# model/xai.py
import numpy as np
import pandas as pd

# SHAP is optional — install if you haven't:
# pip install shap
try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

TARGET_COLS = ["owners", "players", "copiesSold", "revenue"]


def _ensure_shap():
    if not HAS_SHAP:
        raise RuntimeError("SHAP not available. Install with: pip install shap")


def _make_tree_explainer(estimator, background_df: pd.DataFrame):
    """
    Builds a TreeExplainer for a single per-target estimator (LGBM/XGB/CatBoost).
    background_df should be a modest sample (e.g., 500–1000 rows).
    """
    _ensure_shap()
    # For stability/perf, we pass background as ndarray
    return shap.TreeExplainer(
        estimator, data=background_df.values, feature_names=list(background_df.columns)
    )


def _shap_for_single_target(estimator, X1: pd.DataFrame, background_df: pd.DataFrame):
    """
    Returns (base_log, shap_log, pred_log) for ONE target model on ONE row.
    All in LOG space (because models were trained on log1p).
    """
    expl = _make_tree_explainer(estimator, background_df)
    # shap_values returns (values, base_values) with shape (1, n_features)
    sv = expl(X1.values)  # SHAP 0.46+: callable returns Explanation
    shap_vals = np.array(sv.values).reshape(1, -1)
    base_val = float(np.array(sv.base_values).reshape(-1)[0])
    # Model prediction in log space for sanity
    pred_log = float(estimator.predict(X1)[0])
    return base_val, shap_vals[0], pred_log


def _map_log_shap_to_normal(base_log: float, shap_log_vec: np.ndarray, pred_log: float):
    """
    Your models predict log1p(y). To express contributions in NORMAL space y (≈ exp(logy)-1),
    we use first-order approximation: dy ≈ exp(pred_log) * dlogy.
    That is, normal-space contribution ≈ exp(pred_log) * shap_log.
    """
    scale = float(np.exp(pred_log))  # derivative of exp at pred_log
    return scale * shap_log_vec


def explain_prediction_family(
    user_input: dict,
    per_target_family: dict,  # {'owners': est, ...} (all in log space)
    features_used: list,
    background_df: pd.DataFrame,
    top_k: int = 10,
):
    """
    Explain a SINGLE algorithm family (e.g., LGB OR XGB OR CB) across all four targets.
    Returns a dict[target] = DataFrame with columns:
      feature, value, shap_log, contrib_normal
    """
    X1 = _build_single_row(user_input, features_used)
    out = {}
    for t in TARGET_COLS:
        est = per_target_family[t]
        base_log, shap_log, pred_log = _shap_for_single_target(est, X1, background_df)
        contrib_norm = _map_log_shap_to_normal(base_log, shap_log, pred_log)
        df = pd.DataFrame(
            {
                "feature": features_used,
                "value": X1.iloc[0].values,
                "shap_log": shap_log,
                "contrib_normal": contrib_norm,
            }
        )
        df["abs_contrib"] = df["contrib_normal"].abs()
        out[t] = (
            df.sort_values("abs_contrib", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    return out


def explain_prediction_ensemble(
    user_input: dict,
    per_target_models: dict,  # {'lgb': {...}, 'xgb': {...}, 'cb': {...}}
    ensemble_weights: dict,  # {target: {'lgb': w, 'xgb': w, 'cb': w}}
    features_used: list,
    background_df: pd.DataFrame,
    top_k: int = 10,
):
    """
    Approximates per-feature contributions for the ENSEMBLE by:
      1) computing per-family normal-space contributions via SHAP (log→normal mapping)
      2) averaging them with the ensemble target-wise weights.
    Returns dict[target] = DataFrame (feature, value, contrib_normal, abs_contrib) top_k.
    """
    _ensure_shap()
    X1 = _build_single_row(user_input, features_used)
    # Accumulate weighted contributions per target
    results = {}
    for t in TARGET_COLS:
        agg = np.zeros(len(features_used), dtype=float)
        for fam_name, fam_models in per_target_models.items():
            if fam_name not in ensemble_weights[t]:
                continue
            w = float(ensemble_weights[t][fam_name])
            est = fam_models[t]
            _, shap_log, pred_log = _shap_for_single_target(est, X1, background_df)
            contrib_norm = _map_log_shap_to_normal(
                0.0, shap_log, pred_log
            )  # base not needed for diffs
            agg += w * contrib_norm
        df = pd.DataFrame(
            {
                "feature": features_used,
                "value": X1.iloc[0].values,
                "contrib_normal": agg,
            }
        )
        df["abs_contrib"] = df["contrib_normal"].abs()
        results[t] = (
            df.sort_values("abs_contrib", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    return results


def global_importance_from_models(
    per_target_models: dict, features_used: list, ensemble_weights: dict = None
):
    """
    Global importance via built-in tree importances, optionally weighted by ensemble weights per target,
    and then averaged across targets.
    Returns DataFrame with columns: feature, importance
    """
    # First, per-family per-target normalized importances
    fam_target_imps = {fam: {} for fam in per_target_models}
    for fam, models in per_target_models.items():
        for t in TARGET_COLS:
            est = models[t]
            # Works for LGBM/XGB/CB
            if hasattr(est, "feature_importances_"):
                imp = np.array(est.feature_importances_, dtype=float)
            else:
                try:
                    imp = np.array(est.get_feature_importance(), dtype=float)
                except Exception:
                    imp = np.zeros(len(features_used), dtype=float)
            s = imp.sum() or 1.0
            fam_target_imps[fam][t] = imp / s

    # Combine over families with ensemble weights per target
    combined = np.zeros(len(features_used), dtype=float)
    for t_idx, t in enumerate(TARGET_COLS):
        fam_sum = np.zeros(len(features_used), dtype=float)
        for fam, t2imp in fam_target_imps.items():
            w = (
                float(ensemble_weights[t][fam])
                if (ensemble_weights is not None and fam in ensemble_weights[t])
                else 1.0
            )
            fam_sum += w * t2imp[t]
        combined += fam_sum
    combined /= len(TARGET_COLS)

    out = pd.DataFrame({"feature": features_used, "importance": combined})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# ---- tiny helper duplicated here to avoid circular import ----
def _build_single_row(user_input: dict, features_used: list) -> pd.DataFrame:
    x = {f: 0 for f in features_used}
    # Basic fields
    x["price"] = user_input.get("price", 0)
    x["is_free"] = int(user_input.get("is_free", False))
    x["required_age"] = user_input.get("required_age", 0)
    x["achievements"] = user_input.get("achievements", 0)
    x["english"] = int(user_input.get("english", True))
    # Flags (only if present)
    platform_flags = ["windows", "mac", "linux"]
    tag_flags = [
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
    genre_flags = [
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
    for flag in platform_flags + tag_flags + genre_flags:
        if flag in x:
            x[flag] = int(user_input.get(flag, False))
    if "publisherClass_encoded" in x:
        x["publisherClass_encoded"] = user_input.get("publisherClass_encoded", 0)
    if (
        "days_since_release" in x
        and "release_date" in user_input
        and "extract_date" in user_input
    ):
        rd = pd.to_datetime(user_input["release_date"])
        ed = pd.to_datetime(user_input["extract_date"])
        x["days_since_release"] = (ed - rd).days
    return pd.DataFrame([x])[features_used]
