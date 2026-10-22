"""Generate the synthetic layer flock data set.

Each flock follows the published breed standard for its strain. On top of that
come farm and flock effects, the formulated and delivered diets, house
temperature, disease episodes, recording noise and missing records.

Run:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from . import responses as R
from .climate import HOUSING_TEMP_FACTOR, climate_normal
from .config import (FIRST_WEEK, FLOCKS_FILE, QUALITY_FILE, RANDOM_STATE, RAW_DIR,
                     REFERENCE_ENERGY, STANDARDS_DIR, STRAINS, TRUTH_DIR, WEEKLY_FILE)
from .standards import (feed_plateau, requirement_at, standard_at,
                        strain_standard)

N_FARMS = 52
START = pd.Timestamp("2018-01-01")
DATA_CUTOFF = pd.Timestamp("2026-09-28")   # records after this date do not exist yet

# phase-table values used by the farm nutritionists
PHASE_REQ = pd.read_csv(STANDARDS_DIR / "nutrient_requirements.csv").set_index("phase")

HOUSING_MORTALITY = {"enriched cage": 1.0, "barn": 1.4, "free range": 1.8}
DISEASE_RATE = {"enriched cage": 0.45, "barn": 0.6, "free range": 0.8}


def _heat_waves(rng):
    waves = {}
    for zone, lam in [("temperate", 1.2), ("warm", 2.0)]:
        for year in range(START.year, DATA_CUTOFF.year + 2):
            n = rng.poisson(lam)
            waves[(zone, year)] = [(int(rng.integers(24, 35)), int(rng.integers(1, 4)),
                                    float(rng.uniform(3, 7))) for _ in range(n)]
    return waves


def _farm_table(rng):
    rows = []
    for f in range(N_FARMS):
        z_aa = rng.normal()
        z_e = 0.5 * z_aa + np.sqrt(0.75) * rng.normal()  # richer amino acids tend to go with more energy
        rows.append(dict(
            farm_id=f"F{f + 1:03d}",
            housing=rng.choice(["enriched cage", "barn", "free range"], p=[0.35, 0.40, 0.25]),
            climate_zone=rng.choice(["temperate", "warm"], p=[0.65, 0.35]),
            main_strain=STRAINS[f % 2],
            n_houses=int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])),
            aa_density=float(np.clip(0.99 + 0.045 * z_aa, 0.88, 1.10)),
            energy=float(np.clip(REFERENCE_ENERGY + 60 * z_e, 2600, 2880)),
            ca_density=float(rng.uniform(0.90, 1.10)),
            avp_density=float(rng.uniform(0.82, 1.15)),
            coarse_p1=float(rng.uniform(0.15, 0.75)),
            coarse_step=float(rng.choice([0.0, 0.05, 0.10, 0.15])),
            la_density=float(rng.uniform(0.65, 1.35)),
            fi_factor=float(rng.uniform(0.96, 1.04)),
            # hidden farm effects (not in the published data)
            peak_re=rng.normal(0, 0.015),
            pers_re=max(-0.0002, rng.normal(0, 0.0003)),
            onset_re=rng.normal(0, 0.3),
            ew_re=rng.normal(0, 0.8),
            bs_re=rng.normal(0, 1.5),
            hu_re=rng.normal(0, 1.5),
            fi_re=rng.normal(0, 0.02),
            mort_re=float(np.exp(rng.normal(0, 0.25))),
        ))
    return pd.DataFrame(rows)


def _policy(farm, rng):
    """The farm's nutrition policy for one flock, with some drift. In 20 % of
    flocks the farm has switched feed supplier and the policy is redrawn."""
    p = {k: farm[k] for k in ["aa_density", "energy", "ca_density", "avp_density",
                              "coarse_p1", "coarse_step", "la_density", "fi_factor"]}
    if rng.random() < 0.20:
        z_aa = rng.normal()
        p["aa_density"] = float(np.clip(0.99 + 0.045 * z_aa, 0.88, 1.10))
        p["energy"] = float(np.clip(REFERENCE_ENERGY + 60 * (0.5 * z_aa + 0.87 * rng.normal()), 2600, 2880))
        p["ca_density"] = float(rng.uniform(0.90, 1.10))
        p["avp_density"] = float(rng.uniform(0.82, 1.15))
        p["coarse_p1"] = float(rng.uniform(0.15, 0.75))
        p["la_density"] = float(rng.uniform(0.65, 1.35))
    p["aa_density"] += rng.normal(0, 0.015)
    p["ca_density"] += rng.normal(0, 0.015)
    p["energy"] += rng.normal(0, 12)
    return p


def _formulate(policy, strain, rng):
    """Formulated diet per phase (% of feed, kcal/kg).

    The nutritionist divides the daily target by the feed intake expected at
    the diet's energy level, as the breeder guides suggest.
    """
    diets = {}
    for k in (1, 2, 3):
        req = PHASE_REQ.loc[k]
        me = policy["energy"] - (20 if k == 3 else 0) + rng.normal(0, 10)
        fi = feed_plateau(strain) * policy["fi_factor"] * REFERENCE_ENERGY / me
        aa = policy["aa_density"]
        d = {
            "me_kcal": round(me),
            "cp_pct": round(100 * req.crude_protein_g * (aa + rng.normal(0, 0.02)) / fi, 2),
            "dlys_pct": round(100 * req.dig_lysine_g * aa / fi, 3),
            "dmc_pct": round(100 * req.dig_met_cys_g * (aa + rng.normal(0, 0.02)) / fi, 3),
            "ca_pct": round(100 * req.calcium_g * policy["ca_density"] / fi, 2),
            "avp_pct": round(100 * req.available_p_g * policy["avp_density"] / fi, 3),
            "coarse_ca_share": round(float(np.clip(policy["coarse_p1"] + (k - 1) * policy["coarse_step"], 0.1, 0.9)) * 20) / 20,
            "la_pct": round(100 * req.linoleic_acid_g * policy["la_density"] / fi, 2),
        }
        diets[k] = d
    return diets


def _deliver(diets, rng):
    """Delivered (analysed) diet: formulation plus batch-to-batch error."""
    out = {}
    for k, d in diets.items():
        dd = dict(d)
        dd["me_kcal"] = d["me_kcal"] + rng.normal(0, 20)
        for c in ["cp_pct", "dlys_pct", "dmc_pct", "ca_pct", "avp_pct", "la_pct"]:
            dd[c] = d[c] * (1 + rng.normal(0, 0.025))
        out[k] = dd
    return out


def _flock_schedule(farms, rng):
    rows = []
    for _, farm in farms.iterrows():
        for h in range(farm.n_houses):
            start = START + pd.Timedelta(weeks=int(rng.integers(0, 70)))
            while start < DATA_CUTOFF - pd.Timedelta(weeks=8):
                end_age = int(np.clip(round(rng.triangular(72, 84, 92)), 72, 90))
                strain = farm.main_strain if rng.random() > 0.15 else [s for s in STRAINS if s != farm.main_strain][0]
                rows.append(dict(farm_id=farm.farm_id, house_id=f"{farm.farm_id}-H{h + 1}",
                                 first_record_date=start, end_age_wk=end_age, strain=strain))
                start = start + pd.Timedelta(weeks=end_age - FIRST_WEEK + 1 + int(rng.integers(4, 9)))
    flocks = pd.DataFrame(rows).sort_values(["first_record_date", "house_id"]).reset_index(drop=True)
    flocks.insert(0, "flock_id", [f"L{i + 1:04d}" for i in range(len(flocks))])
    return flocks


def simulate_flock(fl, farm, diets, delivered, waves, rng):
    """Weekly records, egg quality samples and the hidden truth for one flock."""
    strain, housing, zone = fl["strain"], farm["housing"], farm["climate_zone"]
    last_age = fl["end_age_wk"]
    max_age_by_date = FIRST_WEEK + (DATA_CUTOFF - fl["first_record_date"]).days // 7
    last_age = int(min(last_age, max_age_by_date))
    ages = np.arange(FIRST_WEEK, last_age + 1)
    n = len(ages)
    dates = fl["first_record_date"] + pd.to_timedelta((ages - FIRST_WEEK) * 7, unit="D")
    woy = np.array([d.isocalendar()[1] for d in dates])
    years = np.array([d.year for d in dates])

    # House temperature
    temp = climate_normal(zone, housing, woy) + rng.normal(0, 0.9, n)
    hf = HOUSING_TEMP_FACTOR[housing]
    for i in range(n):
        extras = [extra for (s, length, extra) in waves.get((zone, years[i]), []) if s <= woy[i] < s + length]
        if extras:
            temp[i] += max(extras) * hf
    temp = np.minimum(temp, 38.0)

    # Diet in force each week
    phase = np.where(ages < fl["phase2_start_wk"], 1, np.where(ages < fl["phase3_start_wk"], 2, 3))
    dcol = lambda c: np.array([delivered[p][c] for p in phase])

    # Disease episodes
    lam = DISEASE_RATE[housing]
    events = []
    for _ in range(rng.poisson(lam * n / 60)):
        if last_age - 3 <= 23:
            break
        events.append(dict(start_wk=int(rng.integers(23, last_age - 2)),
                           kind=rng.choice(["respiratory", "other"]),
                           depth=float(rng.uniform(0.03, 0.16)),
                           half_life_wk=float(rng.uniform(1.2, 4.0)),
                           permanent_share=float(rng.uniform(0.0, 0.35))))
    dz = np.ones(n)
    g_all = np.zeros(n)
    g_resp = np.zeros(n)
    for e in events:
        g = R.disease_curve(ages, e["start_wk"], e["half_life_wk"])
        after = (ages > e["start_wk"]).astype(float)
        dz *= 1 - e["depth"] * (g + e["permanent_share"] * (1 - g) * after)
        g_all += e["depth"] * g
        if e["kind"] == "respiratory":
            g_resp = np.maximum(g_resp, g * e["depth"] / 0.16)

    # Feed intake
    fi_std = standard_at(strain, "feed_g", ages)
    fi_ar = np.zeros(n)
    for i in range(1, n):
        fi_ar[i] = 0.5 * fi_ar[i - 1] + rng.normal(0, 0.012)
    fi_true = R.feed_intake(fi_std, dcol("me_kcal"), temp, housing, fl["pullet_bw_pct"]) \
        * (1 + farm["fi_re"]) * (1 - 0.4 * g_all) * np.exp(fi_ar)

    # Nutrient supply relative to requirement
    def ratio(conc_col, req_col):
        return dcol(conc_col) / 100 * fi_true / requirement_at(ages, req_col, strain)
    r_lys = ratio("dlys_pct", "dig_lysine_g")
    r_mc = ratio("dmc_pct", "dig_met_cys_g")
    r_ca = ratio("ca_pct", "calcium_g")
    r_avp = ratio("avp_pct", "available_p_g")
    la_gap = dcol("la_pct") / 100 * fi_true - requirement_at(ages, "linoleic_acid_g", strain)
    e_req = REFERENCE_ENERGY * fi_std * R.HOUSING_INTAKE[housing] * (1 - 0.010 * np.maximum(0, temp - 21))
    r_e = dcol("me_kcal") * fi_true / e_req
    r_aa = np.minimum(r_lys, r_mc)
    cum_def = np.r_[0.0, np.cumsum(np.maximum(0.0, 1 - r_aa))[:-1]]

    # Hen-day production
    shift = R.onset_shift(fl["photostim_age_wk"], fl["pullet_bw_pct"]) + farm["onset_re"] + fl["_onset_re"]
    hd_base = standard_at(strain, "hen_day_pct", ages - shift)
    hd_true = (hd_base * (1 + farm["peak_re"] + fl["_peak_re"]) * R.HOUSING_PEAK[housing]
               * R.production_multiplier(r_lys, r_mc, r_e, r_ca, r_avp, temp)
               * (1 - R.persistency_loss(ages, farm["pers_re"], housing, cum_def)) * dz)
    hd_true = np.minimum(hd_true, 98.5)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.55 * ar[i - 1] + rng.normal(0, 0.7)
    hd_obs = np.clip(hd_true + ar, 0, 100)

    # Egg weight
    ew_true = (standard_at(strain, "egg_weight_g", ages - 0.5 * shift) + farm["ew_re"]
               + R.egg_weight_effect(r_aa, la_gap, fl["pullet_bw_pct"], fl["photostim_age_wk"], ages, temp, r_e)
               - R.RESP_EGG_WEIGHT_DROP * g_resp)
    ew_obs = ew_true + rng.normal(0, 0.3, n)

    # Mortality (% of hens present, per week)
    cum = standard_at(strain, "mortality_cum_pct", ages)
    m_std = np.diff(cum, prepend=cum[0] - (cum[1] - cum[0]))
    m_true = (m_std * HOUSING_MORTALITY[housing] * farm["mort_re"]
              * (1 + 0.1 * np.maximum(0, temp - 28)) + 1.5 * g_all)
    m_obs = m_true * np.exp(rng.normal(0, 0.25, n))

    # Missing records
    def miss(x, p):
        x = x.astype(float).copy()
        x[rng.random(n) < p] = np.nan
        return x

    weekly = pd.DataFrame({
        "flock_id": fl["flock_id"], "farm_id": fl["farm_id"], "age_wk": ages,
        "week_start": dates.date,
        "house_temp_max_c": np.round(miss(temp, 0.02), 1),
        "hen_day_pct": np.round(miss(hd_obs, 0.02), 2),
        "feed_intake_g": np.round(miss(fi_true * (1 + rng.normal(0, 0.01, n)), 0.05), 1),
        "egg_weight_g": np.round(miss(ew_obs, 0.04), 2),
        "mortality_pct": np.round(miss(m_obs, 0.01), 3),
    })
    truth = pd.DataFrame({
        "flock_id": fl["flock_id"], "age_wk": ages, "temp_c": temp, "phase": phase,
        "feed_intake_true": fi_true, "r_lys": r_lys, "r_mc": r_mc, "r_ca": r_ca,
        "r_avp": r_avp, "r_energy": r_e, "la_gap_g": la_gap, "cum_aa_deficit": cum_def,
        "disease_severity": g_all, "hen_day_true": hd_true, "egg_weight_true": ew_true,
    })

    # Egg quality: 30 eggs every 4 weeks
    first = int(rng.integers(22, 26))
    q_ages = np.arange(first, last_age + 1, 4)
    idx = q_ages - FIRST_WEEK
    keep = rng.random(len(idx)) > 0.10
    idx, q_ages = idx[keep], q_ages[keep]
    coarse = dcol("coarse_ca_share")
    bs_true = (strain_standard(strain)["breaking_strength_n"].reindex(q_ages).values + farm["bs_re"]
               + R.breaking_strength_effect(r_ca[idx], r_avp[idx], coarse[idx], q_ages, temp[idx])
               - R.RESP_SHELL_DROP * g_resp[idx])
    hu_true = (strain_standard(strain)["haugh_units"].reindex(q_ages).values + farm["hu_re"]
               + R.haugh_effect(r_lys[idx], temp[idx]) - R.RESP_HAUGH_DROP * g_resp[idx])
    quality = pd.DataFrame({
        "flock_id": fl["flock_id"], "farm_id": fl["farm_id"], "age_wk": q_ages,
        "n_eggs": 30,
        "breaking_strength_n": np.round(bs_true + rng.normal(0, 1.5, len(idx)), 1),
        "haugh_units": np.round(hu_true + rng.normal(0, 1.1, len(idx)), 1),
    })
    quality_truth = quality[["flock_id", "age_wk"]].assign(
        breaking_strength_true=bs_true, haugh_units_true=hu_true)
    ev = pd.DataFrame(events).assign(flock_id=fl["flock_id"]) if events else None
    return weekly, quality, truth, quality_truth, ev, last_age, shift


def generate(seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    farms = _farm_table(rng)
    waves = _heat_waves(rng)
    flocks = _flock_schedule(farms, rng)
    farm_idx = farms.set_index("farm_id")

    weekly_all, quality_all, truth_all, qtruth_all, events_all, flock_rows, hidden_rows = [], [], [], [], [], [], []
    for _, fl in flocks.iterrows():
        farm = farm_idx.loc[fl.farm_id]
        flock_size_range = {"enriched cage": (30000, 80000), "barn": (15000, 40000), "free range": (6000, 24000)}[farm.housing]
        fl = fl.to_dict()
        fl.update(
            housing=farm.housing, climate_zone=farm.climate_zone,
            flock_size=int(round(rng.integers(*flock_size_range), -2)),
            pullet_bw_pct=round(float(rng.normal(100, 4)), 1),
            photostim_age_wk=int(rng.choice([16, 17, 18, 19], p=[0.2, 0.4, 0.3, 0.1])),
            phase2_start_wk=int(rng.integers(46, 56)),
            phase3_start_wk=int(rng.integers(66, 76)),
            _onset_re=rng.normal(0, 0.25), _peak_re=rng.normal(0, 0.008),
        )
        policy = _policy(farm, rng)
        diets = _formulate(policy, fl["strain"], rng)
        delivered = _deliver(diets, rng)
        weekly, quality, truth, qtruth, ev, last_age, shift = simulate_flock(fl, farm, diets, delivered, waves, rng)
        weekly_all.append(weekly)
        quality_all.append(quality)
        truth_all.append(truth)
        qtruth_all.append(qtruth)
        if ev is not None:
            events_all.append(ev)
        row = {k: fl[k] for k in ["flock_id", "farm_id", "house_id", "strain", "housing", "climate_zone",
                                  "first_record_date", "flock_size", "pullet_bw_pct", "photostim_age_wk",
                                  "phase2_start_wk", "phase3_start_wk", "end_age_wk"]}
        row["last_recorded_age_wk"] = last_age
        row["complete"] = last_age >= fl["end_age_wk"]
        for k in (1, 2, 3):
            for c, v in diets[k].items():
                row[f"{c}_p{k}"] = v
        flock_rows.append(row)
        hid = {"flock_id": fl["flock_id"], "onset_shift_wk": shift,
               "flock_peak_re": fl["_peak_re"], "flock_onset_re": fl["_onset_re"]}
        for k in (1, 2, 3):
            for c, v in delivered[k].items():
                hid[f"delivered_{c}_p{k}"] = v
        hidden_rows.append(hid)

    flocks_out = pd.DataFrame(flock_rows)
    flocks_out["first_record_date"] = flocks_out["first_record_date"].dt.date
    return dict(
        flocks=flocks_out,
        weekly=pd.concat(weekly_all, ignore_index=True),
        quality=pd.concat(quality_all, ignore_index=True),
        farms_truth=farms,
        flocks_truth=pd.DataFrame(hidden_rows),
        weekly_truth=pd.concat(truth_all, ignore_index=True),
        quality_truth=pd.concat(qtruth_all, ignore_index=True),
        events=pd.concat(events_all, ignore_index=True)[["flock_id", "start_wk", "kind", "depth", "half_life_wk", "permanent_share"]],
    )


def main():
    out = generate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    out["flocks"].to_csv(FLOCKS_FILE, index=False)
    out["weekly"].to_csv(WEEKLY_FILE, index=False)
    out["quality"].to_csv(QUALITY_FILE, index=False)
    out["farms_truth"].to_csv(TRUTH_DIR / "farm_effects.csv", index=False)
    out["flocks_truth"].round(4).to_csv(TRUTH_DIR / "flock_effects.csv", index=False)
    out["weekly_truth"].round(4).to_csv(TRUTH_DIR / "weekly_truth.csv.gz", index=False)
    out["quality_truth"].round(3).to_csv(TRUTH_DIR / "quality_truth.csv", index=False)
    out["events"].round(3).to_csv(TRUTH_DIR / "disease_events.csv", index=False)
    f, w, q = out["flocks"], out["weekly"], out["quality"]
    print(f"{len(f)} flocks on {f.farm_id.nunique()} farms, {len(w)} weekly records, {len(q)} egg quality samples")


if __name__ == "__main__":
    main()
