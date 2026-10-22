"""Feature engineering, laying curve and the true response."""
import numpy as np
import pytest

from src.curves import age_at_50pct, fit_yang, yang
from src.features import FEATURES, flock_planning_features, monotone_vector
from src.forecast import HISTORY_COLS, cutoff_features
from src.truth import true_flock
from tests.conftest import make_plan


def test_diet_built_at_target_gives_supply_ratio_near_one(plan):
    f = flock_planning_features(plan).set_index("age_wk")
    # after peak intake, a diet built at 100 % of the daily target supplies about 100 %
    assert f.loc[30:45, "r_lys"].mean() == pytest.approx(1.0, abs=0.03)
    assert f.loc[30:45, "r_ca"].mean() == pytest.approx(1.0, abs=0.03)
    assert f.loc[30:45, "def_aa"].max() < 0.03


def test_monotone_vectors_match_feature_lists():
    for target, cols in FEATURES.items():
        assert len(monotone_vector(target)) == len(cols)


def test_yang_fit_recovers_known_parameters():
    ages = np.arange(19, 81)
    true = (105.0, 0.004, 1.1, 20.6)
    rng = np.random.default_rng(0)
    y = yang(ages, *true) + rng.normal(0, 0.5, len(ages))
    p, rmse = fit_yang(ages, y)
    assert p == pytest.approx(true, rel=0.1)
    assert rmse < 1.0
    assert 19 < age_at_50pct(p) < 22


def test_forecast_features_ignore_records_after_the_cutoff(flocks, weekly):
    fl = flocks.iloc[0].to_dict()
    rec = weekly[weekly.flock_id == fl["flock_id"]].sort_values("age_wk")
    hist = rec[rec.age_wk <= 36][HISTORY_COLS]
    a = cutoff_features(fl, hist)
    future = rec.copy()
    future.loc[future.age_wk > 36, "hen_day_pct"] = 0.0  # change the future
    b = cutoff_features(fl, future[future.age_wk <= 36][HISTORY_COLS])
    assert a.equals(b)


def test_true_response_directions():
    base = true_flock(make_plan())
    low_aa = true_flock(make_plan(amino_acids=0.9))
    more_ca = true_flock(make_plan(calcium=1.1))
    assert low_aa.hen_day_pct.sum() < base.hen_day_pct.sum()
    assert low_aa.egg_weight_g.mean() < base.egg_weight_g.mean()
    assert more_ca.breaking_strength_n.mean() > base.breaking_strength_n.mean()
    # coarse limestone matters more in old hens
    coarse = true_flock(make_plan(coarse=(0.9, 0.9, 0.9))).set_index("age_wk")
    fine = true_flock(make_plan(coarse=(0.1, 0.1, 0.1))).set_index("age_wk")
    gap = coarse.breaking_strength_n - fine.breaking_strength_n
    assert gap.loc[75] > gap.loc[30] > 0
