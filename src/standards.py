"""Published breed standards and nutrient requirements on a weekly grid.

Everything that comes from a breeder guide or a paper is read from the CSV files
in data/standards. Where a value had to be assumed, the function says so.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import FIRST_WEEK, LAST_WEEK, STANDARDS_DIR

WEEKS = np.arange(FIRST_WEEK, LAST_WEEK + 1)

# Lohmann does not publish weekly feed intake or mortality for the Brown-Classic.
# I use the shape of the Hy-Line Brown curves instead, scaled to Lohmann's
# summary figures: 110-120 g/day at 2725 kcal and 20 C, and 90-91 % liveability
# to 100 weeks (Hy-Line Brown: 91.4 %).
LB_FEED_PLATEAU = 115.0
LB_MORTALITY_SCALE = 1.10


@lru_cache(maxsize=None)
def _tables():
    lb = pd.read_csv(STANDARDS_DIR / "lohmann_brown_classic_performance.csv")
    w80 = pd.read_csv(STANDARDS_DIR / "hyline_w80_performance.csv")
    hlb = pd.read_csv(STANDARDS_DIR / "hyline_brown_feed_mortality.csv")
    eq = pd.read_csv(STANDARDS_DIR / "egg_quality_by_age.csv")
    req = pd.read_csv(STANDARDS_DIR / "nutrient_requirements.csv")
    return lb, w80, hlb, eq, req


@lru_cache(maxsize=None)
def strain_standard(strain: str) -> pd.DataFrame:
    """Weekly standard for one strain, weeks 19-90.

    Columns: hen_day_pct, egg_weight_g, feed_g, mortality_cum_pct,
    egg_mass_g, haugh_units, breaking_strength_n.
    For Hy-Line, the midpoint of the published range is used.
    """
    lb, w80, hlb, _, _ = _tables()
    if strain == "Lohmann Brown-Classic":
        d = lb.set_index("week").reindex(WEEKS)
        hlb_i = hlb.set_index("week").reindex(WEEKS)
        feed_mid = (hlb_i["feed_low_g"] + hlb_i["feed_high_g"]) / 2
        plateau = feed_mid.loc[40:60].mean()
        out = pd.DataFrame({
            "hen_day_pct": d["hen_day_pct"],
            "egg_weight_g": d["egg_weight_g"],
            "feed_g": feed_mid * LB_FEED_PLATEAU / plateau,
            "mortality_cum_pct": hlb_i["mortality_cum_pct"] * LB_MORTALITY_SCALE,
        }, index=WEEKS)
        hu_strain = "Hy-Line Brown"  # Lohmann publishes no Haugh unit curve
    elif strain == "Hy-Line W-80":
        d = w80.set_index("week").reindex(WEEKS)
        out = pd.DataFrame({
            "hen_day_pct": (d["hen_day_low_pct"] + d["hen_day_high_pct"]) / 2,
            "egg_weight_g": (d["egg_weight_low_g"] + d["egg_weight_high_g"]) / 2,
            "feed_g": (d["feed_low_g"] + d["feed_high_g"]) / 2,
            "mortality_cum_pct": d["mortality_cum_pct"],
        }, index=WEEKS)
        hu_strain = "Hy-Line W-80"
    else:
        raise ValueError(f"Unknown strain: {strain}")
    out.index.name = "week"
    out["egg_mass_g"] = out["hen_day_pct"] / 100 * out["egg_weight_g"]
    out["haugh_units"] = haugh_curve(hu_strain)(WEEKS)
    out["breaking_strength_n"] = breaking_strength_curve(WEEKS)
    return out


def haugh_curve(hu_strain: str):
    """Haugh units by age from the Hy-Line guides (4 published ages).

    Assumption: at 24 weeks Haugh units are 2 HU above the 38-week value, and
    they stay flat after the last published age.
    """
    eq = _tables()[3]
    pts = eq[(eq.measure == "haugh_units") & (eq.strain == hu_strain)].sort_values("week")
    x = np.r_[24, pts.week.values]
    y = np.r_[pts.value.values[0] + 2.0, pts.value.values]
    f = PchipInterpolator(x, y, extrapolate=False)

    def curve(weeks):
        w = np.clip(np.asarray(weeks, dtype=float), x[0], x[-1])
        return f(w)
    return curve


def breaking_strength_curve(weeks):
    """Shell breaking strength (N) by age.

    Published points: 57.1, 52.1 and 43.1 N at 33, 45 and 67 weeks
    (Benavides-Reyes et al., 2021). Assumptions: 58.5 N up to 28 weeks, a
    straight line to the 33-week value, and half the 45-67 week slope after
    67 weeks.
    """
    eq = _tables()[3]
    pts = eq[eq.measure == "breaking_strength"].sort_values("week")
    x = np.r_[28, pts.week.values]
    y = np.r_[58.5, pts.value.values]
    w = np.asarray(weeks, dtype=float)
    out = np.interp(w, x, y)
    slope = 0.5 * (y[-1] - y[-2]) / (x[-1] - x[-2])
    late = w > x[-1]
    out[late] = y[-1] + slope * (w[late] - x[-1])
    return out


@lru_cache(maxsize=None)
def requirements() -> pd.DataFrame:
    """Daily nutrient requirement per hen (g/day) on the weekly grid.

    Phase values are the Lohmann Brown-Classic recommendations. The guide gives
    three steps; I interpolate linearly between the phase midpoints (34.5, 60
    and 80 weeks) so the requirement changes smoothly with age. The same daily
    intakes are used for both strains.
    """
    req = _tables()[4]
    cols = ["crude_protein_g", "calcium_g", "available_p_g", "dig_lysine_g",
            "dig_met_cys_g", "linoleic_acid_g"]
    mids = np.array([34.5, 60.0, 80.0])
    out = pd.DataFrame(index=WEEKS)
    for c in cols:
        out[c] = np.interp(WEEKS, mids, req[c].values)
    out.index.name = "week"
    return out


def requirement_at(weeks, nutrient: str, strain: str) -> np.ndarray:
    """Requirement scaled down during the rise to peak.

    Before peak intake the hen eats less and lays less, so the daily need is
    scaled by standard feed intake relative to its plateau (the guide keeps
    the diet concentration constant while intake rises).
    """
    w = np.clip(np.asarray(weeks, dtype=int), FIRST_WEEK, LAST_WEEK)
    base = requirements()[nutrient].reindex(w).values
    std = strain_standard(strain)
    plateau = std["feed_g"].loc[40:60].mean()
    scale = np.minimum(1.0, std["feed_g"].reindex(w).values / plateau)
    return base * scale


def standard_at(strain: str, column: str, weeks) -> np.ndarray:
    """Standard value at (possibly fractional) ages, clipped to 19-90 weeks.

    For hen-day production before week 19 the curve falls linearly to zero
    at 17.5 weeks.
    """
    std = strain_standard(strain)[column]
    w = np.asarray(weeks, dtype=float)
    x = std.index.values.astype(float)
    y = std.values
    if column == "hen_day_pct":
        x = np.r_[17.5, x]
        y = np.r_[0.0, y]
    return np.interp(w, x, y)


def feed_plateau(strain: str) -> float:
    return float(strain_standard(strain)["feed_g"].loc[40:60].mean())
