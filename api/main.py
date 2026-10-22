"""REST API for the layer flock models.

Start locally:   uvicorn api.main:app --reload
Docs:            http://127.0.0.1:8000/docs
"""
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.predict import get_predictor

from .schemas import FlockPlan, ForecastRequest, ForecastResponse, PlanResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()  # load the models once at start-up
    yield


app = FastAPI(
    title="Layer flock performance API",
    description="Predicts the laying curve, egg weight and egg quality of a planned layer flock, "
                "and forecasts hen-day production of a running flock 1-12 weeks ahead. "
                "Trained on synthetic data; for demonstration only.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.get("/")
def root():
    return {"service": "layer flock performance API", "docs": "/docs",
            "endpoints": ["/health", "/model-info", "/predict/flock", "/forecast"]}


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "model_version": p.version}


@app.get("/model-info")
def model_info():
    card = get_predictor().card
    return {
        "model_version": card["model_version"],
        "trained_on": card["trained_on"],
        "data": card["data"],
        "planning_models": {t: {k: v[k] for k in ("model", "test", "interval_90_coverage_test")}
                            for t, v in card["planning_models"].items()},
        "forecast_model": {k: card["forecast_model"][k] for k in
                           ("model", "test_mae_by_horizon", "interval_90_coverage_test")},
        "intended_use": card["intended_use"],
    }


@app.post("/predict/flock", response_model=PlanResponse)
def predict_flock(plan: FlockPlan):
    result = get_predictor().predict_plan(plan.model_dump(mode="json"))
    weeks = result["weeks"].round(3).to_dict(orient="records")
    return {**result, "weeks": weeks}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    history = pd.DataFrame([r.model_dump() for r in req.history])
    try:
        result = get_predictor().forecast(req.flock.model_dump(mode="json"), history,
                                          horizons=range(1, req.horizon_weeks + 1))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {**result, "forecast": result["forecast"].round(3).to_dict(orient="records")}
