"""
Pydantic models for API request / response validation.
"""

from pydantic import BaseModel, Field   # to make sure we are accepting, what we should


#  Requests 

class ForecastRequest(BaseModel):
    """
    POST /forecast body.
    """
    state: str = Field(..., description="US state name, e.g. 'California'")
    model_name: str | None = Field(None, description="Model to use (auto-best if omitted)")
    horizon: int = Field(8, ge=1, le=52, description="Weeks to forecast")


class BatchForecastRequest(BaseModel):
    """
    POST /forecast/batch body.
    """
    states: list[str] = Field(..., description="List of state names")
    model_name: str | None = Field(None, description="Model to use (auto-best if omitted)")
    horizon: int = Field(8, ge=1, le=52, description="Weeks to forecast")


#  Responses 

class ForecastPoint(BaseModel):
    """
    Single forecast data point.
    """
    date: str
    state: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    """
    POST /forecast response.
    """
    state: str
    model_name: str
    horizon: int
    forecasts: list[ForecastPoint]


class BatchForecastResponse(BaseModel):
    """
    POST /forecast/batch response.
    """
    results: list[ForecastResponse]


class HealthResponse(BaseModel):
    """
    GET /health response.
    """
    status: str = "ok"
    states_loaded: int
    models_loaded: int
    model_types: list[str]


class MetricsResponse(BaseModel):
    """
    GET /metrics response.
    """
    comparison: dict
    best_per_state: dict


class StateMetricsResponse(BaseModel):
    """
    GET /metrics/{state} response.
    """
    state: str
    best_model: str | None
    metrics: list[dict]
