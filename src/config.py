"""Paths and project-wide settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STANDARDS_DIR = DATA_DIR / "standards"
RAW_DIR = DATA_DIR / "raw"
TRUTH_DIR = DATA_DIR / "truth"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FLOCKS_FILE = RAW_DIR / "flocks.csv"
WEEKLY_FILE = RAW_DIR / "weekly_records.csv"
QUALITY_FILE = RAW_DIR / "egg_quality.csv"

RANDOM_STATE = 42

STRAINS = ["Lohmann Brown-Classic", "Hy-Line W-80"]
HOUSING = ["enriched cage", "barn", "free range"]
CLIMATE_ZONES = ["temperate", "warm"]

FIRST_WEEK = 19          # first week of production records
LAST_WEEK = 90           # oldest age in the data
REFERENCE_ENERGY = 2725  # kcal AMEn/kg at which Lohmann states feed intake

# Diet columns, one set per phase (suffix _p1, _p2, _p3)
DIET_NUTRIENTS = ["me_kcal", "cp_pct", "dlys_pct", "dmc_pct", "ca_pct",
                  "avp_pct", "coarse_ca_share", "la_pct"]
