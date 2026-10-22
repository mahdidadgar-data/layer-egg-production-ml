"""Saved models, the prediction service and the API."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import FlockPlan
from src.predict import get_predictor
from tests.conftest import make_plan

EXAMPLE = FlockPlan.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def predictor():
    return get_predictor()


def eggs_to_72(weeks):
    return float((weeks.loc[weeks.age_wk <= 72, "hen_day_pct"] * 0.07).sum())


def test_plan_prediction_shape_and_intervals(predictor, plan):
    res = predictor.predict_plan(plan)
    w = res["weeks"]
    assert w.age_wk.min() == 19 and w.age_wk.max() == 80
    for t in ["hen_day_pct", "egg_weight_g", "breaking_strength_n", "haugh_units"]:
        assert (w[f"{t}_lower"] <= w[t]).all() and (w[t] <= w[f"{t}_upper"]).all()
    assert 280 < res["summary"]["eggs_per_hen_day_to_72wk"] < 345


def test_model_responds_to_diet_in_the_right_direction(predictor):
    base = predictor.predict_plan(make_plan())["weeks"]
    low_aa = predictor.predict_plan(make_plan(amino_acids=0.9))["weeks"]
    more_ca = predictor.predict_plan(make_plan(calcium=1.1))["weeks"]
    assert eggs_to_72(low_aa) < eggs_to_72(base)
    assert low_aa.egg_weight_g.mean() < base.egg_weight_g.mean()
    assert more_ca.breaking_strength_n.mean() > base.breaking_strength_n.mean()
    # calcium is not an input of the laying-rate model
    assert eggs_to_72(more_ca) == pytest.approx(eggs_to_72(base))


def test_warning_for_inputs_outside_training_range(predictor):
    res = predictor.predict_plan(make_plan(amino_acids=0.7))
    assert any("lysine" in w.lower() for w in res["warnings"])


def test_api_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert set(info["planning_models"]) == {"hen_day_pct", "egg_weight_g", "breaking_strength_n", "haugh_units"}


def test_api_predict_flock(client):
    r = client.post("/predict/flock", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert len(body["weeks"]) == EXAMPLE["end_age_wk"] - 19 + 1
    assert "X-Process-Time-ms" in r.headers


def test_api_rejects_bad_input(client):
    bad = json.loads(json.dumps(EXAMPLE))
    bad["phase3_start_wk"] = 45
    assert client.post("/predict/flock", json=bad).status_code == 422
    bad = json.loads(json.dumps(EXAMPLE))
    bad["diets"]["phase1"]["ca_pct"] = 9.0
    assert client.post("/predict/flock", json=bad).status_code == 422
    bad = json.loads(json.dumps(EXAMPLE))
    bad["strain"] = "Unknown hybrid"
    assert client.post("/predict/flock", json=bad).status_code == 422


def test_api_forecast(client, flocks, weekly):
    hd = [9, 35, 55, 72, 82, 88, 90, 92, 93, 93.5, 94, 93.5, 93, 92.5, 92, 91]
    history = [{"age_wk": 19 + i, "hen_day_pct": v} for i, v in enumerate(hd)]
    r = client.post("/forecast", json={"flock": EXAMPLE, "history": history, "horizon_weeks": 8})
    assert r.status_code == 200
    f = r.json()["forecast"]
    assert [x["horizon"] for x in f] == list(range(1, 9))
    assert all(x["lower"] <= x["hen_day_pct"] <= x["upper"] for x in f)
    assert r.json()["cutoff_age_wk"] == 34
    short = client.post("/forecast", json={"flock": EXAMPLE, "history": history[:3]})
    assert short.status_code == 422


def test_forecast_tracks_a_flock_below_standard(predictor, plan):
    # a flock running 6 points under standard should be forecast below standard
    w = predictor.plan_weeks(plan)[0]
    hist = w[w.age_wk <= 40][["age_wk"]].copy()
    hist["hen_day_pct"] = w.loc[w.age_wk <= 40, "breed_standard_hen_day_pct"].values - 6
    f = predictor.forecast(plan, hist)["forecast"]
    assert (f["hen_day_pct"] < f["breed_standard"] - 2).all()
    assert np.isfinite(f["hen_day_pct"]).all()
