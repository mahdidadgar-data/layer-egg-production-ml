"""Feature engineering.

Planning features use only what is known before the flock starts laying: the
strain, the house, the pullet weight at transfer, the lighting programme, the
planned diets and the time of year. Weekly production records are never used
here, so the planning models cannot leak information from the cycle they
predict.

The diets enter as supply relative to requirement. For each week I estimate
the daily intake of each nutrient from the diet concentration and the
standard feed intake (corrected for dietary energy, because hens eat partly
to energy), and divide it by the daily requirement for that age.
"""
import numpy as np
import pandas as pd

from .climate import climate_normal
from .config import FIRST_WEEK, REFERENCE_ENERGY
from .standards import requirement_at, standard_at

STRAIN_W80 = "Hy-Line W-80"

RATIO_SPECS = {  # feature: (diet column, requirement column)
    "r_lys": ("dlys_pct", "dig_lysine_g"),
    "r_mc": ("dmc_pct", "dig_met_cys_g"),
    "r_ca": ("ca_pct", "calcium_g"),
    "r_avp": ("avp_pct", "available_p_g"),
    "r_cp": ("cp_pct", "crude_protein_g"),
}

FLOCK_COLS = ["strain_w80", "housing_barn", "housing_free_range", "zone_warm",
              "pullet_bw_pct", "photostim_age_wk"]

# Calcium is left out of the laying-rate model on purpose: in the normal range
# it acts on the shell, and in the data its supply moves together with amino
# acid supply, so a model would credit calcium with amino acid effects.
HD_DIET = ["me_kcal", "r_lys", "r_mc", "r_aa", "r_avp", "def_aa", "def_avp", "cum_aa_deficit"]
HD_FEATURES = ["age_wk", "std_hd"] + FLOCK_COLS + ["expected_temp_c", "heat_load"] + HD_DIET
EW_FEATURES = ["age_wk", "std_ew"] + FLOCK_COLS + [
    "expected_temp_c", "heat_load", "me_kcal", "r_lys", "r_mc", "r_aa", "def_aa", "la_gap_g"]
BS_FEATURES = ["age_wk", "std_bs"] + FLOCK_COLS + [
    "expected_temp_c", "heat_load", "r_ca", "def_ca", "r_avp", "def_avp", "coarse_ca_share",
    "coarse_x_age"]
HU_FEATURES = ["age_wk", "std_hu"] + FLOCK_COLS + [
    "expected_temp_c", "heat_load", "r_lys", "r_aa", "def_aa"]

# Monotonic constraints (+1 increasing, -1 decreasing); others unconstrained
MONOTONE = {
    "hen_day_pct": {"std_hd": 1, "r_lys": 1, "r_mc": 1, "r_aa": 1, "r_avp": 1,
                    "def_aa": -1, "def_avp": -1, "cum_aa_deficit": -1, "heat_load": -1},
    "egg_weight_g": {"std_ew": 1, "r_lys": 1, "r_mc": 1, "r_aa": 1, "def_aa": -1, "la_gap_g": 1,
                     "pullet_bw_pct": 1, "heat_load": -1},
    "breaking_strength_n": {"std_bs": 1, "r_ca": 1, "def_ca": -1, "r_avp": 1, "def_avp": -1,
                            "coarse_ca_share": 1, "coarse_x_age": 1, "heat_load": -1},
    "haugh_units": {"std_hu": 1, "r_lys": 1, "r_aa": 1, "def_aa": -1, "heat_load": -1},
}
# In the laying-rate model the diet features may interact with each other but
# not with age, house or season. This keeps the predicted diet response stable.
INTERACTIONS = {"hen_day_pct": [HD_DIET, [c for c in HD_FEATURES if c not in HD_DIET]]}

FEATURES = {"hen_day_pct": HD_FEATURES, "egg_weight_g": EW_FEATURES,
            "breaking_strength_n": BS_FEATURES, "haugh_units": HU_FEATURES}


def monotone_vector(target):
    return tuple(MONOTONE[target].get(c, 0) for c in FEATURES[target])


def _phase(ages, p2, p3):
    return np.where(ages < p2, 1, np.where(ages < p3, 2, 3))


def flock_planning_features(fl, ages=None) -> pd.DataFrame:
    """Planning features for one flock (a dict or Series) at the given ages."""
    if ages is None:
        ages = np.arange(FIRST_WEEK, int(fl["end_age_wk"]) + 1)
    ages = np.asarray(ages, dtype=int)
    strain = fl["strain"]
    phase = _phase(ages, fl["phase2_start_wk"], fl["phase3_start_wk"])
    diet = lambda c: np.array([fl[f"{c}_p{p}"] for p in phase], dtype=float)

    start = pd.Timestamp(fl["first_record_date"])
    woy = np.array([(start + pd.Timedelta(weeks=int(a - FIRST_WEEK))).isocalendar()[1] for a in ages])

    me = diet("me_kcal")
    fi_exp = standard_at(strain, "feed_g", ages) * REFERENCE_ENERGY / me

    out = pd.DataFrame({
        "age_wk": ages,
        "std_hd": standard_at(strain, "hen_day_pct", ages),
        "std_ew": standard_at(strain, "egg_weight_g", ages),
        "std_bs": standard_at(strain, "breaking_strength_n", ages),
        "std_hu": standard_at(strain, "haugh_units", ages),
        "std_fi": standard_at(strain, "feed_g", ages),
        "strain_w80": float(strain == STRAIN_W80),
        "housing_barn": float(fl["housing"] == "barn"),
        "housing_free_range": float(fl["housing"] == "free range"),
        "zone_warm": float(fl["climate_zone"] == "warm"),
        "pullet_bw_pct": float(fl["pullet_bw_pct"]),
        "photostim_age_wk": float(fl["photostim_age_wk"]),
        "week_of_year": woy,
        "expected_temp_c": climate_normal(fl["climate_zone"], fl["housing"], woy),
        "phase": phase,
        "me_kcal": me,
        "coarse_ca_share": diet("coarse_ca_share"),
    })
    for name, (dcol, rcol) in RATIO_SPECS.items():
        out[name] = diet(dcol) / 100 * fi_exp / requirement_at(ages, rcol, strain)
    out["r_aa"] = np.minimum(out["r_lys"], out["r_mc"])
    # broken-line terms: shortfall below requirement, as in requirement studies
    out["def_aa"] = np.maximum(0.0, 1 - out["r_aa"])
    out["def_ca"] = np.maximum(0.0, 1 - out["r_ca"])
    out["def_avp"] = np.maximum(0.0, 1 - out["r_avp"])
    out["heat_load"] = np.maximum(0.0, out["expected_temp_c"] - 25.0)
    out["coarse_x_age"] = out["coarse_ca_share"] * out["age_wk"] / 50.0
    out["la_gap_g"] = diet("la_pct") / 100 * fi_exp - requirement_at(ages, "linoleic_acid_g", strain)
    out["cum_aa_deficit"] = np.r_[0.0, np.cumsum(np.maximum(0.0, 1 - out["r_aa"].values))[:-1]]
    return out


def planning_features(flocks: pd.DataFrame, ages_by_flock=None) -> pd.DataFrame:
    """Planning features for many flocks; one row per flock and week."""
    frames = []
    for _, fl in flocks.iterrows():
        ages = None if ages_by_flock is None else ages_by_flock.get(fl["flock_id"])
        if ages is not None and len(ages) == 0:
            continue
        f = flock_planning_features(fl, ages)
        f.insert(0, "flock_id", fl["flock_id"])
        f.insert(1, "farm_id", fl["farm_id"])
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def weekly_training_frame(flocks, weekly):
    """Weekly records joined to the planning features of the same flock and age."""
    feats = planning_features(flocks)
    return weekly.drop(columns=["farm_id"]).merge(feats, on=["flock_id", "age_wk"], how="inner")


def quality_training_frame(flocks, quality):
    feats = planning_features(flocks)
    return quality.drop(columns=["farm_id"]).merge(feats, on=["flock_id", "age_wk"], how="inner")
