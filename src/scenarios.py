"""Diet scenarios: change a flock's planned diets in all three phases."""
import copy

PHASES = (1, 2, 3)

SCENARIOS = {
    "Amino acids -10 %": ("aa", 0.90),
    "Amino acids -5 %": ("aa", 0.95),
    "Amino acids +5 %": ("aa", 1.05),
    "Calcium -10 %": ("ca", 0.90),
    "Calcium +10 %": ("ca", 1.10),
    "Available P -15 %": ("avp", 0.85),
    "Coarse limestone 30 %": ("coarse", 0.30),
    "Coarse limestone 70 %": ("coarse", 0.70),
    "Linoleic acid -30 %": ("la", 0.70),
}


def apply_scenario(fl, kind, value):
    """Return a copy of the flock plan with the diet change applied."""
    out = copy.copy(dict(fl))
    for p in PHASES:
        if kind == "aa":
            for c in ("dlys_pct", "dmc_pct", "cp_pct"):
                out[f"{c}_p{p}"] = fl[f"{c}_p{p}"] * value
        elif kind == "ca":
            out[f"ca_pct_p{p}"] = fl[f"ca_pct_p{p}"] * value
        elif kind == "avp":
            out[f"avp_pct_p{p}"] = fl[f"avp_pct_p{p}"] * value
        elif kind == "coarse":
            out[f"coarse_ca_share_p{p}"] = value
        elif kind == "la":
            out[f"la_pct_p{p}"] = fl[f"la_pct_p{p}"] * value
        elif kind == "energy":
            out[f"me_kcal_p{p}"] = fl[f"me_kcal_p{p}"] + value
        else:
            raise ValueError(kind)
    return out
