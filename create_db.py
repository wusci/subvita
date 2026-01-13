from src.db.base import Base
from src.db.session import engine
import src.db.models  # ensures models are registered

Base.metadata.create_all(bind=engine)
print("Database tables created in risk_api.db")
