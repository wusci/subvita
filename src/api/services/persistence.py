from sqlalchemy.orm import Session
from src.db.models import PredictionRun, User
from datetime import datetime

def store_prediction_run(
    db: Session,
    *,
    user_id: str | None,
    disease: str,
    model_version: str,
    predicted_label: str,
    probabilities: dict,
    request_payload: dict,
    latency_ms: float,
) -> PredictionRun:
    run = PredictionRun(
        user_id=user_id,
        disease=disease,
        model_version=model_version,
        predicted_label=predicted_label,
        p_normal=float(probabilities.get("p_normal")) if probabilities.get("p_normal") is not None else None,
        p_prediabetes=float(probabilities.get("p_prediabetes")) if probabilities.get("p_prediabetes") is not None else None,
        p_diabetes=float(probabilities.get("p_diabetes")) if probabilities.get("p_diabetes") is not None else None,
        request_payload=request_payload,
        latency_ms=float(latency_ms),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run

def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.id == user_id).one_or_none()
    if user is not None:
        return user

    user = User(id=user_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
