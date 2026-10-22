"""Compare candidate models for the four planning targets.

For each target the candidates are scored on two things:
  1. accuracy on farms held out from training;
  2. how well their predicted response to diet changes matches the known
     true response of the synthetic data (src/truth.py).

Run:  python -m src.compare_models
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV

from .config import FLOCKS_FILE, QUALITY_FILE, REPORTS_DIR, WEEKLY_FILE
from .curves import PARAMS, fit_yang, yang
from .features import (FEATURES, FLOCK_COLS, flock_planning_features, planning_features,
                       quality_training_frame, weekly_training_frame)
from .modeling import eggs_to_age, make_ridge, make_xgb, metrics, split_farms, strain_age_mean
from .scenarios import SCENARIOS, apply_scenario
from .truth import true_flock

TARGETS = {
    "hen_day_pct": ("std_hd", "weekly"),
    "egg_weight_g": ("std_ew", "weekly"),
    "breaking_strength_n": ("std_bs", "quality"),
    "haugh_units": ("std_hu", "quality"),
}
CANDIDATES = {
    "Ridge, broken-line features": lambda t: make_ridge(t),
    "XGBoost, monotonic": lambda t: make_xgb(t),
    "XGBoost, unconstrained": lambda t: make_xgb(t, monotone=False),
}

# Summary of each target used for the response check
RESPONSE_WINDOWS = {
    "hen_day_pct": ("eggs to 72 wk", 19, 72),
    "egg_weight_g": ("mean g, 30-72 wk", 30, 72),
    "breaking_strength_n": ("mean N, 50-72 wk", 50, 72),
    "haugh_units": ("mean HU, 30-72 wk", 30, 72),
}


def summarise(target, ages, values):
    _, lo, hi = RESPONSE_WINDOWS[target]
    sel = (ages >= lo) & (ages <= hi)
    if target == "hen_day_pct":
        return float(np.sum(values[sel]) / 100 * 7)
    return float(np.mean(values[sel]))


def response_table(models, flocks_sample):
    """Mean change versus the flock's own diet, truth and each model."""
    rows = []
    ages = np.arange(19, 73)
    base_feats = {fid: flock_planning_features(fl, ages) for fid, fl in flocks_sample.iterrows()}
    base_true = {fid: true_flock(fl, ages) for fid, fl in flocks_sample.iterrows()}
    for scen, (kind, value) in SCENARIOS.items():
        deltas = {}
        for fid, fl in flocks_sample.iterrows():
            mod = apply_scenario(fl, kind, value)
            f_mod = flock_planning_features(mod, ages)
            t_mod = true_flock(mod, ages)
            for target in TARGETS:
                d_true = summarise(target, ages, t_mod[target].values) - summarise(target, ages, base_true[fid][target].values)
                deltas.setdefault((target, "Truth"), []).append(d_true)
                for name, m in models[target].items():
                    cols = FEATURES[target]
                    d = summarise(target, ages, m.predict(f_mod[cols])) - summarise(target, ages, m.predict(base_feats[fid][cols]))
                    deltas.setdefault((target, name), []).append(d)
        for (target, name), v in deltas.items():
            rows.append({"scenario": scen, "target": target, "model": name, "mean_change": float(np.mean(v))})
    return pd.DataFrame(rows)


YANG_FLOCK_FEATURES = FLOCK_COLS + ["expected_temp_c", "me_kcal", "r_lys", "r_mc", "r_aa", "r_avp",
                                    "def_aa", "def_avp"]


def two_stage_yang(flocks, weekly, train_farms, test_farms):
    """Fit the Yang curve to each training flock, predict its four parameters
    from the flock plan (ridge regression), and rebuild the curve for new flocks.
    """
    fits = []
    for fid, g in weekly.groupby("flock_id"):
        p, rmse = fit_yang(g["age_wk"].values, g["hen_day_pct"].values)
        if p is not None:
            fits.append({"flock_id": fid, "fit_rmse": rmse, **dict(zip(PARAMS, p))})
    fits = pd.DataFrame(fits).merge(flocks[["flock_id", "farm_id", "last_recorded_age_wk"]], on="flock_id")
    # flock-level summary of the plan: average over weeks 19-72
    feats = planning_features(flocks, {fid: np.arange(19, 73) for fid in flocks.flock_id})
    summary = feats.groupby("flock_id")[YANG_FLOCK_FEATURES].mean().reset_index()
    d = fits.merge(summary, on="flock_id")
    tr = d[d.farm_id.isin(train_farms) & (d.last_recorded_age_wk >= 60)]
    te_ids = set(flocks.loc[flocks.farm_id.isin(test_farms), "flock_id"])
    models = {k: RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(tr[YANG_FLOCK_FEATURES], tr[k]) for k in PARAMS}
    te_feats = summary[summary.flock_id.isin(te_ids)].set_index("flock_id")
    pred_params = pd.DataFrame({k: models[k].predict(te_feats[YANG_FLOCK_FEATURES]) for k in PARAMS},
                               index=te_feats.index)
    wk = weekly[weekly.flock_id.isin(te_ids)].dropna(subset=["hen_day_pct"]).copy()
    pp = pred_params.reindex(wk.flock_id)
    wk["pred"] = yang(wk.age_wk.values, pp.a.values, pp.b.values, pp.c.values, pp.d.values)
    return wk, fits, pred_params


def run(n_response_flocks=60, seed=7):
    flocks = pd.read_csv(FLOCKS_FILE)
    weekly = pd.read_csv(WEEKLY_FILE)
    quality = pd.read_csv(QUALITY_FILE)
    train_farms, test_farms = split_farms(flocks)
    frames = {"weekly": weekly_training_frame(flocks, weekly),
              "quality": quality_training_frame(flocks, quality)}

    acc_rows, models = [], {}
    for target, (std_col, src) in TARGETS.items():
        d = frames[src].dropna(subset=[target])
        tr, te = d[d.farm_id.isin(train_farms)], d[d.farm_id.isin(test_farms)]
        acc_rows.append({"target": target, "model": "Breed standard", **metrics(te[target], te[std_col])})
        acc_rows.append({"target": target, "model": "Training average by strain and age",
                         **metrics(te[target], strain_age_mean(tr, target)(te))})
        models[target] = {}
        for name, make in CANDIDATES.items():
            m = make(target).fit(tr[FEATURES[target]], tr[target])
            models[target][name] = m
            acc_rows.append({"target": target, "model": name, **metrics(te[target], m.predict(te[FEATURES[target]]))})
    # Laying curve with predicted parameters (two-stage Yang model)
    wk_yang, fits, yang_params = two_stage_yang(flocks, weekly, train_farms, test_farms)
    acc_rows.append({"target": "hen_day_pct", "model": "Yang curve, parameters from flock plan",
                     **metrics(wk_yang["hen_day_pct"], wk_yang["pred"])})
    acc = pd.DataFrame(acc_rows)

    # Eggs per hen to 72 weeks on the held-out flocks
    d = frames["weekly"]
    te_all = d[d.farm_id.isin(test_farms)].copy()
    tr_all = d[d.farm_id.isin(train_farms)].dropna(subset=["hen_day_pct"])
    obs = eggs_to_age(te_all, "hen_day_pct")
    preds = {"Breed standard": te_all["std_hd"].values,
             "Training average by strain and age": strain_age_mean(tr_all, "hen_day_pct")(te_all)}
    for name, m in models["hen_day_pct"].items():
        preds[name] = m.predict(te_all[FEATURES["hen_day_pct"]])
    tot_rows = []
    for name, p in preds.items():
        est = eggs_to_age(te_all.assign(hen_day_pct=p), "hen_day_pct").reindex(obs.index)
        err = est - obs
        tot_rows.append({"model": name, "mae_eggs": float(err.abs().mean()), "bias_eggs": float(err.mean()),
                         "n_flocks": int(err.notna().sum())})
    ages = np.arange(19, 73)
    yang_tot = pd.Series({fid: float(yang(ages, *yang_params.loc[fid, PARAMS].values).sum() / 100 * 7)
                          for fid in obs.index if fid in yang_params.index})
    err = yang_tot.reindex(obs.index) - obs
    tot_rows.append({"model": "Yang curve, parameters from flock plan", "mae_eggs": float(err.abs().mean()),
                     "bias_eggs": float(err.mean()), "n_flocks": int(err.notna().sum())})
    totals = pd.DataFrame(tot_rows)

    sample = flocks[flocks.farm_id.isin(test_farms) & (flocks.end_age_wk >= 72)]
    sample = sample.sample(min(n_response_flocks, len(sample)), random_state=seed).set_index("flock_id", drop=False)
    resp = response_table(models, sample)

    # Response error: mean absolute gap to the true change across scenarios,
    # counting only scenarios where the true change is meaningful.
    truth = resp[resp.model == "Truth"].set_index(["scenario", "target"])["mean_change"]
    err_rows = []
    for (target, name), g in resp[resp.model != "Truth"].groupby(["target", "model"]):
        g = g.set_index(["scenario", "target"])["mean_change"]
        gap = (g - truth.reindex(g.index)).abs()
        err_rows.append({"target": target, "model": name, "response_error": float(gap.mean())})
    err = pd.DataFrame(err_rows)
    summary = acc.merge(err, on=["target", "model"], how="left")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(REPORTS_DIR / "planning_model_comparison.csv", index=False)
    totals.round(2).to_csv(REPORTS_DIR / "eggs_to_72_weeks.csv", index=False)
    fits.round(5).to_csv(REPORTS_DIR / "yang_fits.csv", index=False)
    resp.round(4).to_csv(REPORTS_DIR / "diet_response_check.csv", index=False)
    return summary, resp, totals


if __name__ == "__main__":
    s, r, t = run()
    pd.set_option("display.width", 160)
    print(s.round(3).to_string(index=False))
    print(t.round(2).to_string(index=False))
    print(r.pivot_table(index=["target", "scenario"], columns="model", values="mean_change").round(2).to_string())
