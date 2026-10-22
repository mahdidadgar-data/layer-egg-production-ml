"""Shared model building and evaluation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from .config import RANDOM_STATE
from .features import FEATURES, INTERACTIONS, monotone_vector

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.04, max_depth=5, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                  random_state=RANDOM_STATE, n_jobs=4)

AGE_BANDS = [(19, 23), (24, 40), (41, 60), (61, 90)]


def split_farms(flocks: pd.DataFrame, test_size=0.2, seed=RANDOM_STATE):
    """Hold out whole farms, so the test set measures performance on new farms."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    farms = flocks["farm_id"].values
    tr, te = next(gss.split(flocks, groups=farms))
    return sorted(set(farms[tr])), sorted(set(farms[te]))


def make_xgb(target, monotone=True, **overrides):
    params = {**XGB_PARAMS, **overrides}
    if monotone:
        params["monotone_constraints"] = monotone_vector(target)
        if target in INTERACTIONS:
            params["interaction_constraints"] = INTERACTIONS[target]
    return XGBRegressor(**params)


def make_ridge(target, alpha=1.0):
    cols = FEATURES[target]
    other = [c for c in cols if c != "age_wk"]
    pre = ColumnTransformer([
        ("age", make_pipeline(SimpleImputer(strategy="median"),
                              SplineTransformer(n_knots=8, degree=3)), ["age_wk"]),
        ("rest", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def strain_age_mean(train: pd.DataFrame, target: str):
    """Baseline: average of the training flocks for each strain and age."""
    table = train.groupby(["strain_w80", "age_wk"])[target].mean()

    def predict(df):
        idx = pd.MultiIndex.from_arrays([df["strain_w80"], df["age_wk"]])
        out = table.reindex(idx).values
        # ages missing in training: fall back to the nearest age of the same strain
        if np.isnan(out).any():
            for i in np.where(np.isnan(out))[0]:
                s, a = df["strain_w80"].iloc[i], df["age_wk"].iloc[i]
                sub = table.loc[s]
                out[i] = sub.iloc[np.argmin(np.abs(sub.index.values - a))]
        return out
    return predict


def oof_residuals(model_fn, df, target, groups, n_splits=5):
    """Out-of-fold predictions on the training farms (grouped by farm)."""
    X, y = df[FEATURES[target]], df[target].values
    oof = np.full(len(df), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    return oof


def interval_halfwidths(ages, residuals, q=0.90):
    """Half-width of a 90 % interval for each age band, from out-of-fold errors."""
    ages, res = np.asarray(ages), np.abs(np.asarray(residuals))
    out = {}
    for lo, hi in AGE_BANDS:
        sel = (ages >= lo) & (ages <= hi) & ~np.isnan(res)
        out[f"{lo}-{hi}"] = float(np.quantile(res[sel], q)) if sel.sum() > 20 else float(np.nanquantile(res, q))
    return out


def halfwidth_for_ages(ages, table):
    ages = np.asarray(ages)
    out = np.empty(len(ages))
    for lo, hi in AGE_BANDS:
        sel = (ages >= lo) & (ages <= hi)
        out[sel] = table[f"{lo}-{hi}"]
    out[ages > AGE_BANDS[-1][1]] = table[f"{AGE_BANDS[-1][0]}-{AGE_BANDS[-1][1]}"]
    return out


def eggs_to_age(df, col, age=72):
    """Eggs per hen (hen-day basis) from week 19 to `age`, per flock.

    Missing weeks are filled by linear interpolation within the flock. Flocks
    that did not reach `age` are left out.
    """
    out = {}
    for fid, g in df.groupby("flock_id"):
        g = g.sort_values("age_wk")
        if g["age_wk"].max() < age:
            continue
        s = g.set_index("age_wk")[col].reindex(range(19, age + 1)).interpolate(limit_direction="both")
        out[fid] = float((s / 100 * 7).sum())
    return pd.Series(out)
