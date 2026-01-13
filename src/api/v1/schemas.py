from __future__ import annotations
from typing import Optional, Literal, Dict, List
from pydantic import BaseModel, Field


SexAtBirth = Literal["male", "female", "unknown"]
RaceEthnicity = Literal[
    "mexican_american",
    "other_hispanic",
    "non_hispanic_white",
    "non_hispanic_black",
    "non_hispanic_asian",
    "other_or_multiracial",
    "unknown",
]
PregStatus = Literal["pregnant", "not_pregnant", "unknown"]


class PredictRequestT2D(BaseModel):
    request_id: Optional[str] = None

    age_years: float = Field(..., ge=0, le=120)
    sex_at_birth: SexAtBirth

    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=20, le=400)
    bmi: Optional[float] = Field(None, ge=10, le=80)
    waist_circumference_cm: float = Field(..., ge=30, le=200)

    systolic_bp_mmHg: float = Field(..., ge=60, le=260)
    diastolic_bp_mmHg: float = Field(..., ge=30, le=160)

    fasting_glucose_mg_dL: float = Field(..., ge=40, le=500)
    triglycerides_mg_dL: float = Field(..., ge=20, le=2000)
    hdl_mg_dL: float = Field(..., ge=5, le=200)
    total_cholesterol_mg_dL: Optional[float] = Field(None, ge=50, le=1000)
    hba1c_percent: Optional[float] = Field(None, ge=3, le=20)

    alt_U_L: Optional[float] = Field(None, ge=0, le=2000)
    creatinine_mg_dL: Optional[float] = Field(None, ge=0.1, le=20)

    race_ethnicity: Optional[RaceEthnicity] = "unknown"
    pregnancy_status: Optional[PregStatus] = "unknown"

    tg_to_hdl_ratio: Optional[float] = None
    non_hdl_chol_mg_dL: Optional[float] = None


class PredictResponse(BaseModel):
    request_id: Optional[str] = None
    disease: str

    predicted_label: str
    probabilities: Dict[str, float]

    suggested_next_steps: List[str]
    notes: List[str]

class PredictResponseStored(PredictResponse):
    run_id: str

class RunSummary(BaseModel):
    run_id: str
    created_at: str  # ISO-like string from SQLite
    disease: str
    predicted_label: str
    probabilities: Dict[str, float]
    model_version: str
    user_id: Optional[str] = None


class RunDetail(RunSummary):
    request_payload: Dict
    latency_ms: Optional[float] = None

class UserCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)


class UserOut(BaseModel):
    user_id: str
    created_at: str

class ErrorOut(BaseModel):
    error: Dict