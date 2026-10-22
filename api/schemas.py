"""Request and response schemas for the API."""
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Strain = Literal["Lohmann Brown-Classic", "Hy-Line W-80"]
Housing = Literal["enriched cage", "barn", "free range"]
Zone = Literal["temperate", "warm"]


class PhaseDiet(BaseModel):
    me_kcal: float = Field(..., ge=2400, le=3100, description="AMEn, kcal/kg")
    cp_pct: float = Field(..., ge=12, le=22, description="Crude protein, %")
    dlys_pct: float = Field(..., ge=0.50, le=1.00, description="Digestible lysine, %")
    dmc_pct: float = Field(..., ge=0.45, le=0.95, description="Digestible methionine + cystine, %")
    ca_pct: float = Field(..., ge=2.5, le=5.5, description="Calcium, %")
    avp_pct: float = Field(..., ge=0.20, le=0.60, description="Available phosphorus, %")
    coarse_ca_share: float = Field(..., ge=0.0, le=1.0, description="Share of limestone fed as coarse particles")
    la_pct: float = Field(..., ge=0.8, le=3.5, description="Linoleic acid, %")


class Diets(BaseModel):
    phase1: PhaseDiet
    phase2: PhaseDiet
    phase3: PhaseDiet


class FlockPlan(BaseModel):
    strain: Strain
    housing: Housing
    climate_zone: Zone
    first_record_date: date = Field(..., description="Date the flock reaches 19 weeks of age")
    pullet_bw_pct: float = Field(100.0, ge=80, le=120, description="Body weight at transfer, % of breed standard")
    photostim_age_wk: int = Field(17, ge=15, le=21)
    phase2_start_wk: int = Field(50, ge=35, le=70)
    phase3_start_wk: int = Field(70, ge=50, le=85)
    end_age_wk: int = Field(80, ge=60, le=90)
    diets: Diets

    @model_validator(mode="after")
    def check_phase_order(self):
        if not (self.phase2_start_wk < self.phase3_start_wk <= self.end_age_wk):
            raise ValueError("Expected phase2_start_wk < phase3_start_wk <= end_age_wk")
        return self

    model_config = {"json_schema_extra": {"example": {
        "strain": "Lohmann Brown-Classic", "housing": "barn", "climate_zone": "temperate",
        "first_record_date": "2026-03-02", "pullet_bw_pct": 101.0, "photostim_age_wk": 17,
        "phase2_start_wk": 50, "phase3_start_wk": 70, "end_age_wk": 80,
        "diets": {
            "phase1": {"me_kcal": 2725, "cp_pct": 16.6, "dlys_pct": 0.73, "dmc_pct": 0.66, "ca_pct": 3.65,
                       "avp_pct": 0.37, "coarse_ca_share": 0.6, "la_pct": 1.8},
            "phase2": {"me_kcal": 2725, "cp_pct": 16.0, "dlys_pct": 0.71, "dmc_pct": 0.64, "ca_pct": 3.90,
                       "avp_pct": 0.35, "coarse_ca_share": 0.7, "la_pct": 1.45},
            "phase3": {"me_kcal": 2700, "cp_pct": 15.1, "dlys_pct": 0.68, "dmc_pct": 0.61, "ca_pct": 3.95,
                       "avp_pct": 0.33, "coarse_ca_share": 0.8, "la_pct": 1.15}}}}}


class WeekRecord(BaseModel):
    age_wk: int = Field(..., ge=19, le=90)
    hen_day_pct: Optional[float] = Field(None, ge=0, le=100)
    feed_intake_g: Optional[float] = Field(None, ge=40, le=180)
    egg_weight_g: Optional[float] = Field(None, ge=35, le=80)
    mortality_pct: Optional[float] = Field(None, ge=0, le=20)
    house_temp_max_c: Optional[float] = Field(None, ge=-10, le=50)


class ForecastRequest(BaseModel):
    flock: FlockPlan
    history: List[WeekRecord] = Field(..., min_length=4)
    horizon_weeks: int = Field(12, ge=1, le=12)

    @model_validator(mode="after")
    def check_history(self):
        ages = [r.age_wk for r in self.history]
        if len(set(ages)) != len(ages):
            raise ValueError("Duplicate ages in history")
        if max(ages) >= self.flock.end_age_wk:
            raise ValueError("History already reaches the planned end of lay")
        if sum(r.hen_day_pct is not None for r in self.history) < 4:
            raise ValueError("At least 4 weeks with hen-day production are needed")
        return self


class WeekPrediction(BaseModel):
    age_wk: int
    phase: int
    hen_day_pct: float
    hen_day_pct_lower: float
    hen_day_pct_upper: float
    breed_standard_hen_day_pct: float
    egg_weight_g: float
    egg_weight_g_lower: float
    egg_weight_g_upper: float
    breaking_strength_n: float
    breaking_strength_n_lower: float
    breaking_strength_n_upper: float
    haugh_units: float
    haugh_units_lower: float
    haugh_units_upper: float
    egg_mass_g: float


class PlanResponse(BaseModel):
    model_version: str
    summary: dict
    warnings: List[str]
    weeks: List[WeekPrediction]


class ForecastWeek(BaseModel):
    age_wk: int
    horizon: int
    hen_day_pct: float
    lower: float
    upper: float
    breed_standard: float


class ForecastResponse(BaseModel):
    model_version: str
    cutoff_age_wk: int
    warnings: List[str]
    forecast: List[ForecastWeek]
