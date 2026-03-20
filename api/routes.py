"""
Endpoints:
  GET  /health          - liveness + loaded artifact counts
  GET  /states          - available state names
  GET  /models          - available model types
  GET  /metrics         - overall model comparison table
  GET  /metrics/{state} - per-state model metrics
  POST /forecast        - 8-week forecast for one state
  POST /forecast/batch  - 8-week forecast for multiple states
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predictor import forecast
from app.schemas import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    BatchForecastRequest, BatchForecastResponse,
    HealthResponse, MetricsResponse, StateMetricsResponse,
)

router = APIRouter()


def _registry(request: Request):
    """
    Get the ModelRegistry from app state (loaded at startup).
    """
    return request.app.state.registry


#  Health 

@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    reg = _registry(request)
    return HealthResponse(
        status="ok",
        states_loaded=len(reg.states),
        models_loaded=len(reg.models),
        model_types=reg.available_models,
    )


#  Lookup 

@router.get("/states")
def list_states(request: Request):
    return {"states": _registry(request).states}


@router.get("/models")
def list_models(request: Request):
    reg = _registry(request)
    from app.config import ALL_MODELS
    return {"models": reg.available_models or ALL_MODELS}


#  Metrics 

@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request):
    reg = _registry(request)
    comparison = reg.get_comparison()
    if not comparison:
        raise HTTPException(404, "No metrics available.")
    return MetricsResponse(
        comparison=comparison,
        best_per_state=reg.best_per_state,
    )


@router.get("/metrics/{state}", response_model=StateMetricsResponse)
def state_metrics(state: str, request: Request):
    reg = _registry(request)
    if state not in reg.states:
        raise HTTPException(404, f"State '{state}' not found")
    return StateMetricsResponse(
        state=state,
        best_model=reg.get_best_model_name(state),
        metrics=reg.get_state_metrics(state),
    )


#  Forecast 

@router.post("/forecast", response_model=ForecastResponse)
def make_forecast(req: ForecastRequest, request: Request):
    """
    Generate an N-week forecast for one state.
    """
    reg = _registry(request)

    if req.state not in reg.states:
        raise HTTPException(404, f"State '{req.state}' not found")

    try:
        result = forecast(reg, req.state, req.horizon, req.model_name)
    except Exception as e:
        raise HTTPException(500, str(e))

    points = [
        ForecastPoint(
            date=str(r["date"])[:10],
            state=r["state"],
            predicted_sales=r["predicted_sales"],
        )
        for r in result["forecasts"]
    ]
    return ForecastResponse(
        state=result["state"],
        model_name=result["model_name"],
        horizon=result["horizon"],
        forecasts=points,
    )


@router.post("/forecast/batch", response_model=BatchForecastResponse)
def batch_forecast(req: BatchForecastRequest, request: Request):
    """
    Generate forecasts for multiple states in one call.
    """
    reg = _registry(request)
    results = []

    for state in req.states:
        if state not in reg.states:
            raise HTTPException(404, f"State '{state}' not found")
        try:
            result = forecast(reg, state, req.horizon, req.model_name)
            points = [
                ForecastPoint(
                    date=str(r["date"])[:10],
                    state=r["state"],
                    predicted_sales=r["predicted_sales"],
                )
                for r in result["forecasts"]
            ]
            results.append(ForecastResponse(
                state=result["state"],
                model_name=result["model_name"],
                horizon=result["horizon"],
                forecasts=points,
            ))
        except Exception as e:
            raise HTTPException(500, f"{state}: {e}")

    return BatchForecastResponse(results=results)
