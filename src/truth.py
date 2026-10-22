"""The known true response of the synthetic data, for validation only.

Because I generated the data, I can compute what a flock would truly do under
a different diet, without noise, disease or farm effects. The models never see
this module; it is used to check whether their predicted response to a diet
change is right.
"""
import numpy as np
import pandas as pd

from . import responses as R
from .climate import climate_normal
from .config import FIRST_WEEK, REFERENCE_ENERGY
from .standards import requirement_at, standard_at, strain_standard


def true_flock(fl, ages=None) -> pd.DataFrame:
    """True weekly performance of an average flock on an average farm.

    Diets are taken as delivered exactly as formulated, the house temperature
    is the climate normal, and there are no disease episodes.
    """
    if ages is None:
        ages = np.arange(FIRST_WEEK, int(fl["end_age_wk"]) + 1)
    ages = np.asarray(ages, dtype=int)
    strain, housing = fl["strain"], fl["housing"]
    start = pd.Timestamp(fl["first_record_date"])
    woy = np.array([(start + pd.Timedelta(weeks=int(a - FIRST_WEEK))).isocalendar()[1] for a in ages])
    temp = climate_normal(fl["climate_zone"], housing, woy)
    phase = np.where(ages < fl["phase2_start_wk"], 1, np.where(ages < fl["phase3_start_wk"], 2, 3))
    diet = lambda c: np.array([fl[f"{c}_p{p}"] for p in phase], dtype=float)

    fi_std = standard_at(strain, "feed_g", ages)
    fi = R.feed_intake(fi_std, diet("me_kcal"), temp, housing, fl["pullet_bw_pct"])

    def ratio(dcol, rcol):
        return diet(dcol) / 100 * fi / requirement_at(ages, rcol, strain)
    r_lys, r_mc = ratio("dlys_pct", "dig_lysine_g"), ratio("dmc_pct", "dig_met_cys_g")
    r_ca, r_avp = ratio("ca_pct", "calcium_g"), ratio("avp_pct", "available_p_g")
    la_gap = diet("la_pct") / 100 * fi - requirement_at(ages, "linoleic_acid_g", strain)
    e_req = REFERENCE_ENERGY * fi_std * R.HOUSING_INTAKE[housing] * (1 - 0.010 * np.maximum(0, temp - 21))
    r_e = diet("me_kcal") * fi / e_req
    r_aa = np.minimum(r_lys, r_mc)
    cum_def = np.r_[0.0, np.cumsum(np.maximum(0.0, 1 - r_aa))[:-1]]

    shift = R.onset_shift(fl["photostim_age_wk"], fl["pullet_bw_pct"])
    hd = (standard_at(strain, "hen_day_pct", ages - shift) * R.HOUSING_PEAK[housing]
          * R.production_multiplier(r_lys, r_mc, r_e, r_ca, r_avp, temp)
          * (1 - R.persistency_loss(ages, 0.0, housing, cum_def)))
    hd = np.minimum(hd, 98.5)
    ew = standard_at(strain, "egg_weight_g", ages - 0.5 * shift) + R.egg_weight_effect(
        r_aa, la_gap, fl["pullet_bw_pct"], fl["photostim_age_wk"], ages, temp, r_e)
    std = strain_standard(strain).reindex(np.clip(ages, 19, 90))
    bs = std["breaking_strength_n"].values + R.breaking_strength_effect(r_ca, r_avp, diet("coarse_ca_share"), ages, temp)
    hu = std["haugh_units"].values + R.haugh_effect(r_lys, temp)
    return pd.DataFrame({"age_wk": ages, "hen_day_pct": hd, "egg_weight_g": ew,
                         "breaking_strength_n": bs, "haugh_units": hu, "feed_intake_g": fi})
