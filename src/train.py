"""Train, validate and save the models used by the API and the app.

  1. Planning models (hen-day production, egg weight, shell breaking strength,
     Haugh units): validated on held-out farms, 90 % intervals from
     out-of-fold errors, then refitted on all farms.
  2. In-cycle forecast of hen-day production 1-12 weeks ahead: compared with
     simple forecasting rules on held-out farms, then refitted on all farms.

The model choice per target follows the comparison in src/compare_models.py.

Run:  python -m src.train     (about 4 minutes)
"""
import json
import platform
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from .config import FLOCKS_FILE, MODELS_DIR, QUALITY_FILE, REPORTS_DIR, TRUTH_DIR, WEEKLY_FILE
from .features import FEATURES, quality_training_frame, weekly_training_frame
from .forecast import (FC_FEATURES, baseline_predictions, build_forecast_frame, fit_forecaster,
                       oof_horizon_halfwidths, predict_hd, yang_refit_predictions)
from .modeling import (halfwidth_for_ages, interval_halfwidths, make_ridge, make_xgb, metrics,
                       oof_residuals, split_farms)

MODEL_VERSION = "1.0.0"

CHOSEN = {
    "hen_day_pct": ("XGBoost, monotonic", lambda: make_xgb("hen_day_pct"),
                    "Most accurate on held-out farms with the right direction for every diet effect. "
                    "The unconstrained version was slightly more accurate but missed the effect of available phosphorus."),
    "egg_weight_g": ("Ridge, broken-line features", lambda: make_ridge("egg_weight_g"),
                     "Most accurate on held-out farms and closest to the true response to amino acids and linoleic acid."),
    "breaking_strength_n": ("Ridge, broken-line features", lambda: make_ridge("breaking_strength_n"),
                            "Most accurate on held-out farms; response to calcium and coarse limestone close to the truth."),
    "haugh_units": ("Ridge, broken-line features", lambda: make_ridge("haugh_units"),
                    "Only slightly better than the breed curve: in this data Haugh units depend mostly on age."),
}
SOURCE = {"hen_day_pct": "weekly", "egg_weight_g": "weekly",
          "breaking_strength_n": "quality", "haugh_units": "quality"}
INPUT_RANGE_COLS = ["pullet_bw_pct", "photostim_age_wk", "me_kcal", "r_lys", "r_mc", "r_ca", "r_avp",
                    "la_gap_g", "coarse_ca_share", "expected_temp_c"]


def train_planning(flocks, weekly, quality, train_farms, test_farms):
    frames = {"weekly": weekly_training_frame(flocks, weekly),
              "quality": quality_training_frame(flocks, quality)}
    card, final = {}, {}
    for target, (name, make, reason) in CHOSEN.items():
        d = frames[SOURCE[target]].dropna(subset=[target])
        tr, te = d[d.farm_id.isin(train_farms)], d[d.farm_id.isin(test_farms)]
        cols = FEATURES[target]
        m = make().fit(tr[cols], tr[target])
        pred = m.predict(te[cols])
        oof = oof_residuals(make, tr, target, tr["farm_id"])
        hw = interval_halfwidths(tr["age_wk"], tr[target].values - oof)
        half = halfwidth_for_ages(te["age_wk"], hw)
        coverage = float(np.mean(np.abs(te[target].values - pred) <= half))
        final[target] = make().fit(d[cols], d[target])
        card[target] = {"model": name, "why": reason, "features": cols,
                        "test": metrics(te[target], pred), "interval_90_halfwidth_by_age": hw,
                        "interval_90_coverage_test": coverage,
                        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te))}
        print(f"{target:22s} {name:30s} test RMSE {card[target]['test']['rmse']:.3f}  "
              f"R2 {card[target]['test']['r2']:.3f}  coverage {coverage:.3f}")
    ranges = frames["weekly"][INPUT_RANGE_COLS].quantile([0.01, 0.99]).T
    ranges.columns = ["p01", "p99"]
    return final, card, {k: [float(v.p01), float(v.p99)] for k, v in ranges.iterrows()}, frames


def train_forecast(flocks, weekly, train_farms, test_farms, planning_hd_model):
    print("Building the forecast data set (about 90 seconds) ...")
    fc = build_forecast_frame(flocks, weekly)
    tr = fc[fc.farm_id.isin(train_farms)]
    te = fc[fc.farm_id.isin(test_farms)].dropna(subset=["hd_observed"]).copy()
    m = fit_forecaster(tr)

    preds = baseline_predictions(te)
    preds["Yang curve refitted to weeks so far"] = yang_refit_predictions(te, weekly)
    # planning model: what the flock plan alone predicts for the target week
    plan = weekly_training_frame(flocks[flocks.farm_id.isin(test_farms)], weekly)
    plan["p"] = planning_hd_model.predict(plan[FEATURES["hen_day_pct"]])
    preds["Planning model (no records)"] = plan.set_index(["flock_id", "age_wk"])["p"].reindex(
        pd.MultiIndex.from_arrays([te.flock_id, te.target_age])).values
    preds["XGBoost forecast"] = predict_hd(m, te)

    truth = pd.read_csv(TRUTH_DIR / "weekly_truth.csv.gz", usecols=["flock_id", "age_wk", "disease_severity"])
    sev = truth.set_index(["flock_id", "age_wk"])["disease_severity"]
    te["episode_at_cutoff"] = sev.reindex(pd.MultiIndex.from_arrays([te.flock_id, te.cutoff_age])).values > 0.01

    rows = []
    for name, p in preds.items():
        err = np.abs(p - te["hd_observed"].values)
        for h in sorted(te.horizon.unique()):
            sel = (te.horizon == h).values & np.isfinite(err)
            rows.append({"method": name, "horizon": int(h), "mae": float(err[sel].mean()), "n": int(sel.sum())})
        ok32 = (te.cutoff_age >= 32).values & np.isfinite(err)
        rows.append({"method": name, "horizon": "all (cutoff >= 32 wk)", "mae": float(err[ok32].mean()), "n": int(ok32.sum())})
        ep = te["episode_at_cutoff"].values & np.isfinite(err)
        rows.append({"method": name, "horizon": "all, during a disease episode", "mae": float(err[ep].mean()), "n": int(ep.sum())})
    by_h = pd.DataFrame(rows)

    hw = oof_horizon_halfwidths(tr)
    half = te["horizon"].map(hw).values
    coverage = float(np.mean(np.abs(te["hd_observed"].values - preds["XGBoost forecast"]) <= half))
    final = fit_forecaster(fc)
    card = {"model": "XGBoost on deviation from breed standard", "features": FC_FEATURES,
            "cutoffs_wk": sorted(int(c) for c in fc.cutoff_age.unique()),
            "test_mae_by_horizon": {int(r.horizon): r.mae for r in by_h[(by_h.method == "XGBoost forecast")
                                                                          & by_h.horizon.apply(lambda x: isinstance(x, int))].itertuples()},
            "interval_90_halfwidth_by_horizon": hw, "interval_90_coverage_test": coverage,
            "n_train_rows": int(tr.dev_target.notna().sum()), "n_test_rows": int(len(te))}
    print(f"forecast: coverage {coverage:.3f}")
    return final, card, by_h


def main():
    flocks = pd.read_csv(FLOCKS_FILE)
    weekly = pd.read_csv(WEEKLY_FILE)
    quality = pd.read_csv(QUALITY_FILE)
    train_farms, test_farms = split_farms(flocks)

    planning, p_card, ranges, frames = train_planning(flocks, weekly, quality, train_farms, test_farms)
    # forecast comparison uses a planning model fitted on the training farms only
    d = frames["weekly"].dropna(subset=["hen_day_pct"])
    hd_train_only = CHOSEN["hen_day_pct"][1]().fit(d[d.farm_id.isin(train_farms)][FEATURES["hen_day_pct"]],
                                                   d[d.farm_id.isin(train_farms)]["hen_day_pct"])
    forecaster, f_card, by_h = train_forecast(flocks, weekly, train_farms, test_farms, hd_train_only)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for target, m in planning.items():
        joblib.dump(m, MODELS_DIR / f"planning_{target}.joblib")
    joblib.dump(forecaster, MODELS_DIR / "forecast_hen_day.joblib")
    by_h.round(4).to_csv(REPORTS_DIR / "forecast_by_horizon.csv", index=False)

    card = {
        "model_version": MODEL_VERSION,
        "trained_on": str(date.today()),
        "data": {"description": "Synthetic layer flocks built on published Lohmann Brown-Classic and Hy-Line W-80 "
                                "standards; effects of diet, housing, climate and disease are assumptions.",
                 "n_flocks": int(len(flocks)), "n_farms": int(flocks.farm_id.nunique()),
                 "n_weekly_records": int(len(weekly)), "n_quality_samples": int(len(quality)),
                 "validation": f"{len(test_farms)} farms held out ({int(flocks.farm_id.isin(test_farms).sum())} flocks); "
                               f"final models refitted on all {flocks.farm_id.nunique()} farms"},
        "planning_models": p_card,
        "forecast_model": f_card,
        "input_ranges_p01_p99": ranges,
        "strains": sorted(flocks.strain.unique().tolist()),
        "housing": sorted(flocks.housing.unique().tolist()),
        "climate_zones": sorted(flocks.climate_zone.unique().tolist()),
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "intended_use": "Portfolio demonstration of planning and forecasting layer flock performance. "
                        "Not for real feeding or management decisions.",
    }
    with open(MODELS_DIR / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print("Saved models and model card to", MODELS_DIR)


if __name__ == "__main__":
    main()
