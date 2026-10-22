"""Published standards are read correctly and the generated data are sane."""
import numpy as np
import pandas as pd
import pytest

from src import generate_data as G
from src.config import TRUTH_DIR
from src.standards import breaking_strength_curve, haugh_curve, requirements, strain_standard


def test_lohmann_cumulative_eggs_match_the_guide():
    # Lohmann Brown-Classic guide: 323.8 eggs per hen housed to 72 weeks, 366.8 to 80
    std = strain_standard("Lohmann Brown-Classic")
    assert (std.hen_day_pct.loc[19:72] * 7 / 100).sum() == pytest.approx(323.8, abs=0.5)
    assert (std.hen_day_pct.loc[19:80] * 7 / 100).sum() == pytest.approx(366.8, abs=0.5)


def test_w80_midpoint_matches_cumulative_hen_day_eggs():
    # Hy-Line W-80 guide: 333.7-347.4 hen-day eggs to 72 weeks
    std = strain_standard("Hy-Line W-80")
    total = (std.hen_day_pct.loc[19:72] * 7 / 100).sum()
    assert 333.7 <= total <= 347.4


def test_egg_quality_curves_pass_through_published_points():
    hu = haugh_curve("Hy-Line W-80")
    assert hu([38, 56, 70]) == pytest.approx([88.1, 85.6, 83.7], abs=1e-6)
    assert breaking_strength_curve([33, 45, 67]) == pytest.approx([57.1, 52.1, 43.1], abs=1e-6)


def test_requirements_follow_the_phase_table():
    req = requirements()
    assert req.loc[30, "dig_lysine_g"] == pytest.approx(0.82)
    assert req.loc[60, "calcium_g"] == pytest.approx(4.40)
    assert req.loc[85, "available_p_g"] == pytest.approx(0.38)


def test_raw_data_ranges(flocks, weekly):
    assert weekly.hen_day_pct.dropna().between(0, 100).all()
    assert weekly.egg_weight_g.dropna().between(35, 80).all()
    assert weekly.age_wk.between(19, 90).all()
    assert set(flocks.strain) == {"Lohmann Brown-Classic", "Hy-Line W-80"}
    # every flock's records stop at its planned end or at the data cut-off
    last = weekly.groupby("flock_id").age_wk.max()
    assert (last <= flocks.set_index("flock_id").end_age_wk.reindex(last.index)).all()


def test_published_files_do_not_contain_hidden_effects(flocks, weekly):
    hidden = [c for c in flocks.columns if c.endswith("_re") or c.startswith("delivered") or "shift" in c]
    assert hidden == []
    assert "hen_day_true" not in weekly.columns
    assert (TRUTH_DIR / "weekly_truth.csv.gz").exists()


def test_generator_is_reproducible(monkeypatch):
    monkeypatch.setattr(G, "N_FARMS", 4)
    a, b = G.generate(seed=1), G.generate(seed=1)
    pd.testing.assert_frame_equal(a["weekly"], b["weekly"])
    c = G.generate(seed=2)
    assert not np.allclose(a["weekly"].hen_day_pct.fillna(0).values[:200],
                           c["weekly"].hen_day_pct.fillna(0).values[:200])
