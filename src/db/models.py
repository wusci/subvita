import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, JSON
from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # e.g. "alice", "sanity-user"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)

    disease = Column(String, nullable=False)
    model_version = Column(String, nullable=False)

    predicted_label = Column(String, nullable=False)
    p_normal = Column(Float)
    p_prediabetes = Column(Float)
    p_diabetes = Column(Float)

    request_payload = Column(JSON)
    latency_ms = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
