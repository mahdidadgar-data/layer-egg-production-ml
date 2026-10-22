"""The response of layers to diet and environment in the synthetic data.

These equations generate the data, and the same functions are used later to
check whether the models learned the right response. Every effect size here is
an assumption I chose to be plausible and in the direction reported in the
literature. They are not estimates from real flocks.
"""
import numpy as np

# Feed intake
ENERGY_ELASTICITY = 0.8      # hens eat partly to energy (Coertze et al. 2025: +150 kcal -> about -4.5 % intake)
HEAT_INTAKE_DROP = 0.015     # fraction of intake lost per C above 21 C
HEAT_INTAKE_START = 21.0
HOUSING_INTAKE = {"enriched cage": 1.00, "barn": 1.03, "free range": 1.06}

# Hen-day production
HOUSING_PEAK = {"enriched cage": 1.000, "barn": 0.990, "free range": 0.975}
HOUSING_PERSISTENCY = {"enriched cage": 0.0, "barn": 0.0001, "free range": 0.00025}  # extra loss per week after 32 wk
AA_DEFICIT_SLOPE = 0.12      # production loss per unit deficit in lysine or Met+Cys
ENERGY_DEFICIT_SLOPE = 0.6
CA_THRESHOLD, CA_SLOPE = 0.85, 0.6
AVP_THRESHOLD, AVP_SLOPE = 0.80, 0.5
HEAT_PRODUCTION_START, HEAT_PRODUCTION_DROP = 28.0, 0.008
CUMULATIVE_AA_PERSISTENCY = 0.006  # loss per unit of accumulated amino acid deficit (weeks x fraction)


def deficit(ratio, threshold=1.0):
    """Relative shortfall below a threshold, zero above it (broken line)."""
    r = np.asarray(ratio, dtype=float)
    return np.maximum(0.0, threshold - r) / threshold


def feed_intake(fi_std, me_kcal, temp_c, housing, bw_pct, reference_energy=2725.0):
    """Expected daily feed intake (g) before farm effects, noise and disease."""
    energy = (reference_energy / np.asarray(me_kcal, dtype=float)) ** ENERGY_ELASTICITY
    heat = 1.0 - HEAT_INTAKE_DROP * np.maximum(0.0, np.asarray(temp_c) - HEAT_INTAKE_START)
    body = 1.0 + 0.3 * (np.asarray(bw_pct) - 100.0) / 100.0
    return fi_std * energy * heat * HOUSING_INTAKE[housing] * body


def onset_shift(photostim_age_wk, pullet_bw_pct):
    """Weeks the production curve is delayed (+) or advanced (-).

    Later light stimulation delays the onset of lay; heavier pullets start a
    little earlier.
    """
    return 0.7 * (np.asarray(photostim_age_wk) - 17.0) - 0.08 * (np.asarray(pullet_bw_pct) - 100.0)


def production_multiplier(r_lys, r_mc, r_energy, r_ca, r_avp, temp_c):
    """Immediate effect of nutrient supply and heat on hen-day production."""
    m = (1 - AA_DEFICIT_SLOPE * deficit(r_lys))
    m = m * (1 - AA_DEFICIT_SLOPE * deficit(r_mc))
    m = m * (1 - ENERGY_DEFICIT_SLOPE * deficit(r_energy))
    m = m * (1 - CA_SLOPE * deficit(r_ca, CA_THRESHOLD))
    m = m * (1 - AVP_SLOPE * deficit(r_avp, AVP_THRESHOLD))
    m = m * (1 - HEAT_PRODUCTION_DROP * np.maximum(0.0, np.asarray(temp_c) - HEAT_PRODUCTION_START))
    return m


def persistency_loss(age_wk, farm_rate, housing, cum_aa_deficit):
    """Fraction of production lost through poorer persistency."""
    weeks_after = np.maximum(0.0, np.asarray(age_wk, dtype=float) - 32.0)
    rate = farm_rate + HOUSING_PERSISTENCY[housing]
    return weeks_after * rate + CUMULATIVE_AA_PERSISTENCY * np.asarray(cum_aa_deficit)


def egg_weight_effect(r_aa, la_gap_g, pullet_bw_pct, photostim_age_wk, age_wk, temp_c, r_energy):
    """Change in egg weight (g) relative to the breed standard.

    r_aa is the lower of the lysine and Met+Cys supply ratios. Below the
    requirement egg weight falls steeply; above it egg weight keeps rising a
    little up to 108 % of requirement.
    """
    r_aa = np.asarray(r_aa, dtype=float)
    age = np.asarray(age_wk, dtype=float)
    e = 8.0 * np.minimum(0.0, r_aa - 1.0) + 25.0 * np.clip(r_aa - 1.0, 0.0, 0.08)
    e = e + 0.8 * np.clip(la_gap_g, -1.2, 0.6)
    e = e + 0.10 * (np.asarray(pullet_bw_pct) - 100.0)
    e = e + 0.5 * (np.asarray(photostim_age_wk) - 17.0) * np.exp(-(age - 19.0) / 12.0)
    e = e - 0.15 * np.maximum(0.0, np.asarray(temp_c) - 26.0)
    e = e + 4.0 * np.minimum(0.0, np.asarray(r_energy) - 1.0)
    return e


def coarse_age_weight(age_wk):
    """Coarse limestone matters more as hens age (0 at 25 wk, 1 at 70 wk)."""
    return np.clip((np.asarray(age_wk, dtype=float) - 25.0) / 45.0, 0.0, 1.2)


def breaking_strength_effect(r_ca, r_avp, coarse_share, age_wk, temp_c):
    """Change in shell breaking strength (N) relative to the age curve."""
    r_ca = np.asarray(r_ca, dtype=float)
    e = 25.0 * np.minimum(0.0, r_ca - 1.0) + 20.0 * np.clip(r_ca - 1.0, 0.0, 0.10)
    e = e + 6.0 * (np.asarray(coarse_share) - 0.5) * coarse_age_weight(age_wk)
    e = e + 12.0 * np.minimum(0.0, np.asarray(r_avp) - 0.9)
    e = e - 0.8 * np.maximum(0.0, np.asarray(temp_c) - 26.0)
    return e


def haugh_effect(r_lys, temp_c):
    """Change in Haugh units relative to the breed curve."""
    return 6.0 * np.minimum(0.0, np.asarray(r_lys) - 1.0) - 0.3 * np.maximum(0.0, np.asarray(temp_c) - 26.0)


# Effects of a respiratory disease episode at full severity (g = 1)
RESP_SHELL_DROP = 8.0   # N
RESP_HAUGH_DROP = 5.0   # HU
RESP_EGG_WEIGHT_DROP = 1.0  # g


def disease_curve(age_wk, start_wk, half_life_wk):
    """Severity of an episode over time: 0.6 in the first week, 1 the week
    after, then exponential recovery."""
    age = np.asarray(age_wk, dtype=float)
    g = np.zeros_like(age)
    g[age == start_wk] = 0.6
    after = age > start_wk
    g[after] = np.exp(-np.log(2) * (age[after] - start_wk - 1) / half_life_wk)
    return g
