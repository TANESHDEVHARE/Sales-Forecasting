"""
Backend Code
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_loader import ModelRegistry
from api.routes import router

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s — %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models + data on startup.
    """
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    yield


app = FastAPI(
    title="Sales Forecasting API",
    description="Serves 8-week beverage sales forecasts per US state.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
