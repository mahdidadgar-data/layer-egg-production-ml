"""Build a three-phase layer diet from the daily nutrient targets.

The concentration of each nutrient is the daily target divided by the feed
intake expected at the diet's energy level, which is how the breeder guides
present their recommendations. Used by the app and in examples.
"""
import pandas as pd

from .config import REFERENCE_ENERGY, STANDARDS_DIR
from .standards import feed_plateau

PHASE_REQ = pd.read_csv(STANDARDS_DIR / "nutrient_requirements.csv").set_index("phase")

NUTRIENT_LABELS = {
    "me_kcal": "Energy (kcal AMEn/kg)",
    "cp_pct": "Crude protein (%)",
    "dlys_pct": "Digestible lysine (%)",
    "dmc_pct": "Digestible Met+Cys (%)",
    "ca_pct": "Calcium (%)",
    "avp_pct": "Available phosphorus (%)",
    "coarse_ca_share": "Coarse limestone (share of limestone)",
    "la_pct": "Linoleic acid (%)",
}


def build_diets(strain, amino_acids=1.0, calcium=1.0, available_p=1.0, energy=REFERENCE_ENERGY,
                linoleic=1.0, coarse=(0.6, 0.7, 0.8)):
    """Return {1: {...}, 2: {...}, 3: {...}} with nutrient concentrations.

    amino_acids, calcium, available_p and linoleic are multipliers of the
    daily target (1.0 = exactly the Lohmann recommendation).
    """
    diets = {}
    for k in (1, 2, 3):
        req = PHASE_REQ.loc[k]
        me = energy - (25 if k == 3 else 0)
        fi = feed_plateau(strain) * REFERENCE_ENERGY / me
        diets[k] = {
            "me_kcal": float(round(me)),
            "cp_pct": round(100 * req.crude_protein_g * amino_acids / fi, 2),
            "dlys_pct": round(100 * req.dig_lysine_g * amino_acids / fi, 3),
            "dmc_pct": round(100 * req.dig_met_cys_g * amino_acids / fi, 3),
            "ca_pct": round(100 * req.calcium_g * calcium / fi, 2),
            "avp_pct": round(100 * req.available_p_g * available_p / fi, 3),
            "coarse_ca_share": float(coarse[k - 1]),
            "la_pct": round(100 * req.linoleic_acid_g * linoleic / fi, 2),
        }
    return diets


def diets_to_nested(diets):
    return {f"phase{k}": dict(v) for k, v in diets.items()}


def diets_table(diets) -> pd.DataFrame:
    df = pd.DataFrame({f"Phase {k}": pd.Series(v) for k, v in diets.items()})
    df.index = [NUTRIENT_LABELS[i] for i in df.index]
    return df


def table_to_diets(df: pd.DataFrame):
    inv = {v: k for k, v in NUTRIENT_LABELS.items()}
    return {k: {inv[i]: float(df.loc[i, f"Phase {k}"]) for i in df.index} for k in (1, 2, 3)}
