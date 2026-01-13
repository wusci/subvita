from __future__ import annotations
from typing import Optional, Literal, Dict, Any, List
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


class PredictRequest(BaseModel):
    # ID is optional (useful for your own logging / user linking later)
    request_id: Optional[str] = None

    # Core demographics
    age_years: float = Field(..., ge=0, le=120)
    sex_at_birth: SexAtBirth

    # Anthropometrics
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=20, le=400)
    bmi: Optional[float] = Field(None, ge=10, le=80)
    waist_circumference_cm: float = Field(..., ge=30, le=200)

    # Blood pressure
    systolic_bp_mmHg: float = Field(..., ge=60, le=260)
    diastolic_bp_mmHg: float = Field(..., ge=30, le=160)

    # Labs (fasting cohort)
    fasting_glucose_mg_dL: float = Field(..., ge=40, le=500)
    triglycerides_mg_dL: float = Field(..., ge=20, le=2000)
    hdl_mg_dL: float = Field(..., ge=5, le=200)
    total_cholesterol_mg_dL: Optional[float] = Field(None, ge=50, le=1000)
    hba1c_percent: Optional[float] = Field(None, ge=3, le=20)

    # Optional biochem
    alt_U_L: Optional[float] = Field(None, ge=0, le=2000)
    creatinine_mg_dL: Optional[float] = Field(None, ge=0.1, le=20)

    # Optional fields if your feature list includes them
    race_ethnicity: Optional[RaceEthnicity] = "unknown"
    pregnancy_status: Optional[PregStatus] = "unknown"

    # Optional derived features (we can compute if absent)
    tg_to_hdl_ratio: Optional[float] = None
    non_hdl_chol_mg_dL: Optional[float] = None


class PredictResponse(BaseModel):
    request_id: Optional[str] = None

    predicted_class: Literal["normal", "prediabetes", "diabetes"]
    probabilities: Dict[str, float]  # p_normal, p_prediabetes, p_diabetes

    # Helpful outputs for app UX
    suggested_next_steps: List[str]
    notes: List[str]
