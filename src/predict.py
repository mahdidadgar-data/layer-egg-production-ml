"""Load the trained models and make predictions for a flock plan.

Used by both the FastAPI service (api/main.py) and the Streamlit app (app.py),
so the two always give the same answer.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import FIRST_WEEK, MODELS_DIR
from .curves import age_at_50pct, fit_yang
from .features import FEATURES, flock_planning_features
from .forecast import FC_FEATURES, HISTORY_COLS, cutoff_features
from .modeling import halfwidth_for_ages

TARGETS = ["hen_day_pct", "egg_weight_g", "breaking_strength_n", "haugh_units"]
DIET_KEYS = ["me_kcal", "cp_pct", "dlys_pct", "dmc_pct", "ca_pct", "avp_pct", "coarse_ca_share", "la_pct"]
RANGE_LABELS = {
    "pullet_bw_pct": "Pullet body weight", "photostim_age_wk": "Age at light stimulation",
    "me_kcal": "Dietary energy", "r_lys": "Digestible lysine supply", "r_mc": "Digestible Met+Cys supply",
    "r_ca": "Calcium supply", "r_avp": "Available phosphorus supply", "la_gap_g": "Linoleic acid supply",
    "coarse_ca_share": "Coarse limestone share", "expected_temp_c": "Expected house temperature",
}


def flatten_plan(plan: dict) -> dict:
    """Turn a nested plan ({..., 'diets': {'phase1': {...}}}) into the flat
    form used internally (me_kcal_p1, ...). Flat plans pass through."""
    if "diets" not in plan:
        return dict(plan)
    flat = {k: v for k, v in plan.items() if k != "diets"}
    for k in (1, 2, 3):
        for key, val in plan["diets"][f"phase{k}"].items():
            flat[f"{key}_p{k}"] = val
    flat.setdefault("flock_id", "plan")
    flat.setdefault("farm_id", "farm")
    return flat


class Predictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.planning = {t: joblib.load(models_dir / f"planning_{t}.joblib") for t in TARGETS}
        self.forecaster = joblib.load(models_dir / "forecast_hen_day.joblib")
        with open(models_dir / "model_card.json") as f:
            self.card = json.load(f)
        self.version = self.card["model_version"]

    # ---------- planning ----------
    def plan_weeks(self, plan: dict) -> pd.DataFrame:
        fl = flatten_plan(plan)
        feats = flock_planning_features(fl)
        out = pd.DataFrame({"age_wk": feats["age_wk"], "phase": feats["phase"],
                            "breed_standard_hen_day_pct": feats["std_hd"],
                            "breed_standard_egg_weight_g": feats["std_ew"]})
        for t in TARGETS:
            p = self.planning[t].predict(feats[FEATURES[t]])
            if t == "hen_day_pct":
                p = np.clip(p, 0, 100)
            hw = halfwidth_for_ages(feats["age_wk"], self.card["planning_models"][t]["interval_90_halfwidth_by_age"])
            out[t] = p
            out[f"{t}_lower"] = np.clip(p - hw, 0, None)
            out[f"{t}_upper"] = p + hw
        out["hen_day_pct_upper"] = np.minimum(out["hen_day_pct_upper"], 100)
        out["egg_mass_g"] = out["hen_day_pct"] / 100 * out["egg_weight_g"]
        return out, feats

    def summary(self, weeks: pd.DataFrame) -> dict:
        w = weeks.set_index("age_wk")
        last = int(w.index.max())
        to72 = w.loc[:min(72, last)]
        params, _ = fit_yang(w.index.values, w["hen_day_pct"].values)
        out = {
            "eggs_per_hen_day_to_72wk": round(float((to72["hen_day_pct"] / 100 * 7).sum()), 1) if last >= 72 else None,
            "eggs_per_hen_day_to_end": round(float((w["hen_day_pct"] / 100 * 7).sum()), 1),
            "egg_mass_kg_per_hen_to_end": round(float((w["egg_mass_g"] * 7).sum() / 1000), 2),
            "mean_egg_weight_g": round(float(np.average(w["egg_weight_g"], weights=w["hen_day_pct"])), 1),
            "peak_hen_day_pct": round(float(w["hen_day_pct"].max()), 1),
            "age_at_peak_wk": int(w["hen_day_pct"].idxmax()),
            "age_at_50pct_wk": round(age_at_50pct(params), 1) if params is not None else None,
            "breaking_strength_n_at_70wk": round(float(w["breaking_strength_n"].get(70, np.nan)), 1) if last >= 70 else None,
            "haugh_units_at_70wk": round(float(w["haugh_units"].get(70, np.nan)), 1) if last >= 70 else None,
            "laying_curve": None if params is None else {
                "model": "Yang et al. (1989)", "a": round(float(params[0]), 3), "b": round(float(params[1]), 5),
                "c": round(float(params[2]), 3), "d": round(float(params[3]), 2)},
        }
        return out

    def range_warnings(self, feats: pd.DataFrame) -> list:
        msgs = []
        for col, (lo, hi) in self.card["input_ranges_p01_p99"].items():
            if col not in feats:
                continue
            v = feats[col].values
            n_out = int(((v < lo) | (v > hi)).sum())
            if n_out:
                msgs.append(f"{RANGE_LABELS.get(col, col)} is outside the range seen in training "
                            f"({lo:.2f} to {hi:.2f}) in {n_out} of {len(v)} weeks. Treat those weeks with caution.")
        return msgs

    def predict_plan(self, plan: dict) -> dict:
        weeks, feats = self.plan_weeks(plan)
        return {"model_version": self.version, "summary": self.summary(weeks),
                "warnings": self.range_warnings(feats), "weeks": weeks}

    # ---------- in-cycle forecast ----------
    def forecast(self, plan: dict, history: pd.DataFrame, horizons=range(1, 13)) -> dict:
        fl = flatten_plan(plan)
        hist = history.copy()
        for c in HISTORY_COLS:
            if c not in hist:
                hist[c] = np.nan
        hist = hist[HISTORY_COLS].apply(pd.to_numeric, errors="coerce").astype(float).sort_values("age_wk")
        cutoff = int(hist["age_wk"].max())
        horizons = [h for h in horizons if cutoff + h <= max(int(fl["end_age_wk"]), cutoff + 1)]
        plan_feats = flock_planning_features(fl, np.arange(FIRST_WEEK, cutoff + max(horizons) + 1))
        f = cutoff_features(fl, hist, horizons, plan_feats)
        dev = self.forecaster.predict(f[FC_FEATURES])
        hd = np.clip(f["std_hd_target"].values + dev, 0, 100)
        hw = np.array([self.card["forecast_model"]["interval_90_halfwidth_by_horizon"][str(h)] for h in horizons])
        out = pd.DataFrame({"age_wk": f["target_age"].values, "horizon": horizons, "hen_day_pct": hd,
                            "lower": np.clip(hd - hw, 0, 100), "upper": np.clip(hd + hw, 0, 100),
                            "breed_standard": f["std_hd_target"].values})
        warnings = []
        n_obs = int(hist["hen_day_pct"].notna().sum())
        if n_obs < 4:
            warnings.append("Fewer than 4 weeks of hen-day records; the forecast leans on the breed standard.")
        if cutoff < 24:
            warnings.append("The model was trained on cutoffs from 24 weeks; forecasts before peak are less reliable.")
        return {"model_version": self.version, "cutoff_age_wk": cutoff, "forecast": out, "warnings": warnings}


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
