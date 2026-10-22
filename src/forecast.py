"""In-cycle forecast of hen-day production.

At a cutoff age the farm has its weekly records up to that week. The model
forecasts hen-day production 1 to 12 weeks ahead. It predicts the deviation
from the breed standard, using the recent deviation and its trend, feed intake,
mortality, egg weight and house temperature, plus what is known about the
coming weeks: the planned diet and the normal temperature for the time of year.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from .config import FIRST_WEEK, RANDOM_STATE
from .curves import fit_yang, yang
from .features import flock_planning_features

HORIZONS = list(range(1, 13))
CUTOFFS = list(range(24, 81, 4))

FC_FEATURES = [
    "cutoff_age", "horizon", "strain_w80", "housing_barn", "housing_free_range", "zone_warm",
    "std_hd_target", "std_hd_cutoff", "dev_last", "dev_mean4", "dev_mean8", "dev_slope4",
    "recovery6", "weeks_since_min6", "hd_ratio4", "fi_ratio2", "fi_change", "mort4",
    "ew_dev2", "temp_anom2", "expected_temp_target", "r_aa_target", "r_ca_target",
    "r_avp_target", "phase_change",
]

FC_PARAMS = dict(n_estimators=500, learning_rate=0.04, max_depth=5, min_child_weight=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                 random_state=RANDOM_STATE, n_jobs=4)

HISTORY_COLS = ["age_wk", "hen_day_pct", "feed_intake_g", "egg_weight_g",
                "mortality_pct", "house_temp_max_c"]


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _slope(ages, y):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return 0.0
    return float(np.polyfit(ages[ok], y[ok], 1)[0])


def _history_arrays(history, plan):
    """Align a flock's history (ages FIRST_WEEK..cutoff) with its plan."""
    h = history.set_index("age_wk").reindex(range(FIRST_WEEK, int(history["age_wk"].max()) + 1))
    ages = h.index.values
    p = plan.set_index("age_wk").reindex(ages)
    return ages, h, p


def cutoff_features(fl, history: pd.DataFrame, horizons=HORIZONS, plan=None) -> pd.DataFrame:
    """Forecast features for one flock at the last age in `history`.

    `fl` holds the flock plan (strain, housing, diets ...). `history` has
    HISTORY_COLS for weeks up to the cutoff; missing values are allowed.
    """
    c = int(history["age_wk"].max())
    target_ages = np.array([c + h for h in horizons])
    if plan is None:
        plan = flock_planning_features(fl, np.arange(FIRST_WEEK, max(target_ages.max(), c) + 1))
    ages, h, p = _history_arrays(history, plan)
    std = p["std_hd"].values
    hd = h["hen_day_pct"].values.astype(float)
    dev = hd - std

    last_ok = np.where(np.isfinite(dev))[0]
    dev_last = dev[last_ok[-1]] if len(last_ok) else 0.0
    w4, w6, w8 = ages > c - 4, ages > c - 6, ages > c - 8
    d6 = np.where(w6 & np.isfinite(dev), dev, np.inf)
    i_min = int(np.argmin(d6)) if np.isfinite(d6).any() else len(ages) - 1
    fi_ratio = h["feed_intake_g"].values / p["std_fi"].values
    temp_anom = h["house_temp_max_c"].values - p["expected_temp_c"].values

    base = {
        "cutoff_age": c,
        "strain_w80": float(p["strain_w80"].iloc[-1]),
        "housing_barn": float(p["housing_barn"].iloc[-1]),
        "housing_free_range": float(p["housing_free_range"].iloc[-1]),
        "zone_warm": float(p["zone_warm"].iloc[-1]),
        "std_hd_cutoff": float(std[-1]),
        "dev_last": float(dev_last),
        "dev_mean4": _nanmean(dev[w4]),
        "dev_mean8": _nanmean(dev[w8]),
        "dev_slope4": _slope(ages[w4], dev[w4]),
        "recovery6": float(dev_last - d6[i_min]) if np.isfinite(d6[i_min]) else 0.0,
        "weeks_since_min6": float(c - ages[i_min]),
        "hd_ratio4": _nanmean(hd[w4] / std[w4]),
        "fi_ratio2": _nanmean(fi_ratio[ages > c - 2]),
        "fi_change": _nanmean(fi_ratio[ages > c - 2]) - _nanmean(fi_ratio[(ages > c - 8) & (ages <= c - 4)]),
        "mort4": float(np.nanmean(h["mortality_pct"].values[w4]) * 4) if np.isfinite(h["mortality_pct"].values[w4]).any() else np.nan,
        "ew_dev2": _nanmean((h["egg_weight_g"].values - p["std_ew"].values)[ages > c - 2]),
        "temp_anom2": _nanmean(temp_anom[ages > c - 2]),
    }
    pt = plan.set_index("age_wk").reindex(target_ages)
    phase_c = plan.set_index("age_wk")["phase"].get(c, np.nan)
    rows = []
    for k, hz in enumerate(horizons):
        r = dict(base)
        r.update(horizon=hz, target_age=int(target_ages[k]),
                 std_hd_target=float(pt["std_hd"].iloc[k]),
                 expected_temp_target=float(pt["expected_temp_c"].iloc[k]),
                 r_aa_target=float(pt["r_aa"].iloc[k]), r_ca_target=float(pt["r_ca"].iloc[k]),
                 r_avp_target=float(pt["r_avp"].iloc[k]),
                 phase_change=float(pt["phase"].iloc[k] != phase_c))
        rows.append(r)
    return pd.DataFrame(rows)


def build_forecast_frame(flocks, weekly, cutoffs=CUTOFFS, horizons=HORIZONS):
    """All (flock, cutoff, horizon) rows with the observed target where available."""
    frames = []
    wk = weekly.sort_values(["flock_id", "age_wk"])
    by_flock = dict(tuple(wk.groupby("flock_id")))
    for _, fl in flocks.iterrows():
        g = by_flock.get(fl["flock_id"])
        if g is None:
            continue
        last = int(g["age_wk"].max())
        plan = flock_planning_features(fl, np.arange(FIRST_WEEK, last + max(horizons) + 1))
        obs = g.set_index("age_wk")["hen_day_pct"]
        for c in cutoffs:
            if c >= last:
                break
            hz = [h for h in horizons if c + h <= last]
            hist = g[g["age_wk"] <= c][HISTORY_COLS]
            f = cutoff_features(fl, hist, hz, plan)
            f.insert(0, "flock_id", fl["flock_id"])
            f.insert(1, "farm_id", fl["farm_id"])
            f["hd_observed"] = obs.reindex(f["target_age"]).values
            f["hd_last"] = f["std_hd_cutoff"] + f["dev_last"]
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["dev_target"] = out["hd_observed"] - out["std_hd_target"]
    return out


def make_forecaster():
    return XGBRegressor(**FC_PARAMS)


def fit_forecaster(frame):
    d = frame.dropna(subset=["dev_target"])
    m = make_forecaster()
    m.fit(d[FC_FEATURES], d["dev_target"])
    return m


def predict_hd(model, frame):
    return np.clip(frame["std_hd_target"].values + model.predict(frame[FC_FEATURES]), 0, 100)


def baseline_predictions(frame):
    return {
        "Breed standard": frame["std_hd_target"].values,
        "Last week carried forward": frame["hd_last"].values,
        "Standard x recent ratio": np.clip(frame["std_hd_target"].values * frame["hd_ratio4"].values, 0, 100),
    }


def yang_refit_predictions(frame, weekly, min_cutoff=32):
    """Fit the Yang curve to the weeks seen so far and extrapolate it."""
    pred = np.full(len(frame), np.nan)
    wk = dict(tuple(weekly.groupby("flock_id")))
    for (fid, c), idx in frame.groupby(["flock_id", "cutoff_age"]).groups.items():
        if c < min_cutoff:
            continue
        g = wk[fid]
        g = g[g["age_wk"] <= c]
        p, _ = fit_yang(g["age_wk"].values, g["hen_day_pct"].values)
        if p is not None:
            pred[frame.index.get_indexer(idx)] = yang(frame.loc[idx, "target_age"].values, *p)
    return pred


def oof_horizon_halfwidths(frame, n_splits=5, q=0.90):
    """90 % interval half-width per horizon from out-of-fold errors."""
    d = frame.dropna(subset=["dev_target"]).reset_index(drop=True)
    oof = np.full(len(d), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(d, groups=d["farm_id"]):
        m = make_forecaster().fit(d.loc[tr, FC_FEATURES], d.loc[tr, "dev_target"])
        oof[va] = m.predict(d.loc[va, FC_FEATURES])
    err = np.abs(oof - d["dev_target"].values)
    return {int(h): float(np.quantile(err[d["horizon"] == h], q)) for h in sorted(d["horizon"].unique())}
