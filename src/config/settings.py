import os
from dotenv import load_dotenv
load_dotenv()

ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///risk_api.db")
CYCLE = os.getenv("CYCLE", "2017-2018")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "200000"))