from __future__ import annotations

from typing import Dict, List
from fastapi import APIRouter, HTTPException, Depends, Header

from sqlalchemy.orm import Session
from ..app import get_registry
from src.db.deps import get_db
from src.db.models import PredictionRun, User
from src.api.services.persistence import store_prediction_run, get_or_create_user
from .schemas import PredictRequestT2D, PredictResponse, PredictResponseStored, RunSummary, RunDetail, UserCreate, UserOut
from ..services.model_registry import ModelRegistry

from sqlalchemy.orm import Session

import logging
import time

logger = logging.getLogger("risk-api")

router = APIRouter(prefix="/v1", tags=["v1"])


def compute_derived_fields(payload: Dict) -> Dict:
    # TG/HDL ratio
    if payload.get("tg_to_hdl_ratio") is None:
        hdl = payload.get("hdl_mg_dL")
        tg = payload.get("triglycerides_mg_dL")
        if hdl and hdl != 0 and tg is not None:
            payload["tg_to_hdl_ratio"] = tg / hdl

    # non-HDL cholesterol
    if payload.get("non_hdl_chol_mg_dL") is None:
        tc = payload.get("total_cholesterol_mg_dL")
        hdl = payload.get("hdl_mg_dL")
        if tc is not None and hdl is not None:
            payload["non_hdl_chol_mg_dL"] = tc - hdl

    # BMI if missing but height/weight present
    if payload.get("bmi") is None:
        ht = payload.get("height_cm")
        wt = payload.get("weight_kg")
        if ht is not None and wt is not None and ht > 0:
            h_m = ht / 100.0
            payload["bmi"] = wt / (h_m * h_m)

    return payload


def next_steps_t2d(p_diabetes: float, p_pred: float, req: PredictRequestT2D) -> List[str]:
    steps = []
    if p_diabetes >= 0.5:
        steps.append("Consider confirming with a clinician: repeat fasting glucose and/or HbA1c.")
        steps.append("Discuss lifestyle changes (nutrition, activity) and medication options if indicated.")
    elif p_pred >= 0.5:
        steps.append("Consider follow-up screening (HbA1c and fasting glucose) within 3-6 months.")
        steps.append("Lifestyle changes: increase activity, reduce refined carbs, aim for waist/BMI improvement.")
    else:
        steps.append("Maintain healthy lifestyle habits and routine screening intervals.")

    if req.hba1c_percent is not None and req.hba1c_percent >= 5.7:
        steps.append("Your HbA1c is in a higher range; consider monitoring trends over time.")
    return steps

def _run_to_summary_dict(run: PredictionRun) -> Dict:
    probs = {
        "p_normal": float(run.p_normal) if run.p_normal is not None else 0.0,
        "p_prediabetes": float(run.p_prediabetes) if run.p_prediabetes is not None else 0.0,
        "p_diabetes": float(run.p_diabetes) if run.p_diabetes is not None else 0.0,
    }
    return {
        "run_id": run.id,
        "created_at": str(run.created_at),
        "disease": run.disease,
        "predicted_label": run.predicted_label,
        "probabilities": probs,
        "model_version": run.model_version,
        "user_id": run.user_id,
    }

@router.get("/health")
def health(registry: ModelRegistry = Depends(get_registry)):
    return {
        "status": "ok",
        "models_loaded": list(registry.bundles.keys()),
    }

@router.get("/models")
def models(registry: ModelRegistry = Depends(get_registry)):
    return registry.list_models()

@router.post("/predict/t2d", response_model=PredictResponse)
def predict_t2d(req: PredictRequestT2D, registry: ModelRegistry = Depends(get_registry)):
    start = time.time()

    try:
        bundle = registry.get("t2d")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    payload = compute_derived_fields(req.model_dump())
    proba = bundle.predict_proba(payload)

    probs = {
        "p_normal": float(proba[0]),
        "p_prediabetes": float(proba[1]),
        "p_diabetes": float(proba[2]),
    }

    labels = ["normal", "prediabetes", "diabetes"]
    pred_label = labels[int(proba.argmax())]

    elapsed_ms = (time.time() - start) * 1000.0

    logger.info(
        "predict_t2d request_id=%s pred=%s p_diabetes=%.3f latency_ms=%.1f",
        req.request_id,
        pred_label,
        probs["p_diabetes"],
        elapsed_ms,
    )

    notes = [
        f"Prototype model for t2d trained on NHANES {bundle.spec.cycle} (strict fasting cohort).",
        "This output is for research/education; not a medical diagnosis.",
    ]
    if bundle.global_top_features:
        notes.append("Global top drivers (dataset-level): " + ", ".join(bundle.global_top_features[:5]))

    return PredictResponse(
        request_id=req.request_id,
        disease="t2d",
        predicted_label=pred_label,
        probabilities=probs,
        suggested_next_steps=next_steps_t2d(probs["p_diabetes"], probs["p_prediabetes"], req),
        notes=notes,
    )

@router.post("/predict-and-store/t2d", response_model=PredictResponseStored)
def predict_and_store_t2d(
    req: PredictRequestT2D,
    registry: ModelRegistry = Depends(get_registry),
    db: Session = Depends(get_db),
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
):
    start = time.time()

    try:
        bundle = registry.get("t2d")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Same inference logic as /predict/t2d
    payload = compute_derived_fields(req.model_dump())
    proba = bundle.predict_proba(payload)

    probs = {
        "p_normal": float(proba[0]),
        "p_prediabetes": float(proba[1]),
        "p_diabetes": float(proba[2]),
    }

    labels = ["normal", "prediabetes", "diabetes"]
    pred_label = labels[int(proba.argmax())]

    elapsed_ms = (time.time() - start) * 1000.0
    
    if x_user_id:
        get_or_create_user(db, x_user_id)

    # Persist run
    run = store_prediction_run(
        db,
        user_id=x_user_id,
        disease="t2d",
        model_version=f"nhanes_{bundle.spec.cycle}",
        predicted_label=pred_label,
        probabilities=probs,
        request_payload=payload,  # includes derived fields
        latency_ms=elapsed_ms,
    )

    # Logging (include run_id)
    logger.info(
        "predict_and_store_t2d request_id=%s run_id=%s pred=%s p_diabetes=%.3f latency_ms=%.1f",
        req.request_id,
        run.id,
        pred_label,
        probs["p_diabetes"],
        elapsed_ms,
    )

    notes = [
        f"Prototype model for t2d trained on NHANES {bundle.spec.cycle} (strict fasting cohort).",
        "This output is for research/education; not a medical diagnosis.",
    ]
    if bundle.global_top_features:
        notes.append("Global top drivers (dataset-level): " + ", ".join(bundle.global_top_features[:5]))

    return PredictResponseStored(
        run_id=run.id,
        request_id=req.request_id,
        disease="t2d",
        predicted_label=pred_label,
        probabilities=probs,
        suggested_next_steps=next_steps_t2d(probs["p_diabetes"], probs["p_prediabetes"], req),
        notes=notes,
    )

@router.get("/runs", response_model=List[RunSummary])
def list_runs(
    limit: int = 50,
    offset: int = 0,
    disease: str | None = None,
    user_id: str | None = None,
    db: Session = Depends(get_db),
):
    """
    List stored prediction runs (summary).
    Optional filters: disease, user_id.
    """
    # basic guardrails
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    q = db.query(PredictionRun)

    if disease is not None:
        q = q.filter(PredictionRun.disease == disease)
    if user_id is not None:
        q = q.filter(PredictionRun.user_id == user_id)

    runs = (
        q.order_by(PredictionRun.created_at.desc())
         .offset(offset)
         .limit(limit)
         .all()
    )

    return [_run_to_summary_dict(r) for r in runs]

@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(PredictionRun).filter(PredictionRun.id == run_id).one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    base = _run_to_summary_dict(run)
    base["request_payload"] = run.request_payload or {}
    base["latency_ms"] = float(run.latency_ms) if run.latency_ms is not None else None
    return base

@router.post("/users", response_model=UserOut)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    user_id = payload.user_id.strip()
    existing = db.query(User).filter(User.id == user_id).one_or_none()
    if existing is not None:
        return {"user_id": existing.id, "created_at": str(existing.created_at)}

    user = User(id=user_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user_id": user.id, "created_at": str(user.created_at)}

@router.get("/users", response_model=List[UserOut])
def list_users(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    users = (
        db.query(User)
        .order_by(User.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [{"user_id": u.id, "created_at": str(u.created_at)} for u in users]

@router.get("/users/{user_id}/runs", response_model=List[RunSummary])
def list_runs_for_user(user_id: str, limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    runs = (
        db.query(PredictionRun)
        .filter(PredictionRun.user_id == user_id)
        .order_by(PredictionRun.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_run_to_summary_dict(r) for r in runs]
