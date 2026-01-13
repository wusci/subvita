from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import LOG_LEVEL, CYCLE, CORS_ORIGINS, MAX_BODY_BYTES
from .services.model_registry import ModelRegistry, ModelSpec

# Ensure .env is loaded before we read settings/environment-dependent behavior
load_dotenv()

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("risk-api")

MODEL_DIR = Path("data_processed") / CYCLE / "models"
REPORT_DIR = Path("data_processed") / CYCLE / "reports"


def create_registry() -> ModelRegistry:
    specs = [
        ModelSpec(
            disease="t2d",
            cycle=CYCLE,
            model_path=MODEL_DIR / "model_a_calibrated.joblib",
            feature_list_path=MODEL_DIR / "feature_list.json",
            perm_importance_path=REPORT_DIR / "stage8_permutation_importance_test.csv",
        ),
    ]
    reg = ModelRegistry(specs)
    reg.load_all()
    return reg


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "payload_too_large",
                                "message": "Request body too large",
                            }
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


app = FastAPI(title="Multi-disease Risk API (Prototype)", version="1.0.0")

# ---- CORS (must be added BEFORE include_router) ----
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
if not origins:
    # Dev fallback so preflight never fails silently if env didn't load
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # enables OPTIONS preflight handling
    allow_headers=["*"],
)

# Optional safety: limit request body sizes
app.add_middleware(MaxBodySizeMiddleware)

# Load models once at startup
registry = create_registry()


def get_registry(request: Request) -> ModelRegistry:
    # registry is global for now; request arg makes it usable as a dependency
    return registry


# ---- Consistent error responses ----
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "http_error", "message": str(exc.detail)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Invalid request",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "server_error", "message": "Internal server error"}},
    )


# Import routes AFTER registry/get_registry exist (avoids circular import headaches)
from .v1.routes import router as v1_router  # noqa: E402

app.include_router(v1_router)


@app.get("/")
def root():
    return {"message": "Use /docs", "version": app.version}
