import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FLOCKS_FILE, WEEKLY_FILE  # noqa: E402
from src.diet_builder import build_diets  # noqa: E402


@pytest.fixture(scope="session")
def flocks():
    return pd.read_csv(FLOCKS_FILE)


@pytest.fixture(scope="session")
def weekly():
    return pd.read_csv(WEEKLY_FILE)


def make_plan(strain="Lohmann Brown-Classic", housing="barn", zone="temperate", **diet_kw):
    plan = {"flock_id": "test", "farm_id": "test", "strain": strain, "housing": housing,
            "climate_zone": zone, "first_record_date": "2026-03-02", "pullet_bw_pct": 100.0,
            "photostim_age_wk": 17, "phase2_start_wk": 50, "phase3_start_wk": 70, "end_age_wk": 80}
    for k, d in build_diets(strain, **diet_kw).items():
        for key, v in d.items():
            plan[f"{key}_p{k}"] = v
    return plan


@pytest.fixture
def plan():
    return make_plan()
