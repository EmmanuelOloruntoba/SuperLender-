from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    SinglePredictionRequest,
)
from src.feature_engineer import FeatureEngineer
from src.model_manager import ModelManager
from src.risk_prediction_service import RiskPredictionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "superlender_hgb_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


@lru_cache(maxsize=1)
def get_risk_service() -> RiskPredictionService:
    """Build the production inference service once per API process."""
    try:
        feature_engineer = FeatureEngineer()
        model_manager = ModelManager(
            model_path=MODEL_PATH,
            metadata_path=METADATA_PATH,
        )
        # Force artifact loading at startup/first dependency resolution so
        # missing or incompatible production artifacts fail clearly.
        _ = model_manager.model
        return RiskPredictionService(
            feature_engineer=feature_engineer,
            model_manager=model_manager,
        )
    except (FileNotFoundError, RuntimeError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Production inference service is unavailable: {exc}",
        ) from exc


app = FastAPI(
    title="SuperLender Credit Risk API",
    version="1.0.0",
    description=(
        "Application-time credit-risk inference API for SuperLender. "
        "The API reuses the production FeatureEngineer and frozen model artifact."
    ),
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "service": "SuperLender Credit Risk API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["system"])
def health(
    service: RiskPredictionService = Depends(get_risk_service),
) -> dict[str, Any]:
    """Return readiness and production model metadata."""
    return {
        "status": "ok",
        "model": service.model_manager.info(),
    }


@app.get("/model-info", tags=["system"])
def model_info(
    service: RiskPredictionService = Depends(get_risk_service),
) -> dict[str, Any]:
    return service.model_manager.info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Score one current loan application",
)
def predict(
    request: SinglePredictionRequest,
    service: RiskPredictionService = Depends(get_risk_service),
) -> dict[str, Any]:
    try:
        return service.predict_single(
            current_loan=request.current_loan.model_dump(),
            previous_loans=request.previous_loans,
            demographics=request.demographics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - final API safety boundary
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while scoring the application.",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
    summary="Score a batch of current loan applications",
)
def predict_batch(
    request: BatchPredictionRequest,
    service: RiskPredictionService = Depends(get_risk_service),
) -> BatchPredictionResponse:
    try:
        results = service.predict_batch(
            current_loans=[
                row.model_dump()
                for row in request.current_loans
            ],
            previous_loans=request.previous_loans,
            demographics=request.demographics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - final API safety boundary
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while scoring the batch.",
        ) from exc

    records = results.to_dict(orient="records")
    return BatchPredictionResponse(
        count=len(records),
        predictions=records,
    )


@app.exception_handler(HTTPException)
async def http_exception_passthrough(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )
