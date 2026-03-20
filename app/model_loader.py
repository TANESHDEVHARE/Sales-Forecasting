"""
app/model_loader.py
===================
Load trained models and metadata produced by the notebook.

The ModelRegistry is initialized once at API startup and cached in memory.
It provides fast lookups for: models, historical data, metrics, and
best-model-per-state mappings.
"""

import logging
import pickle
from typing import Any

import pandas as pd

from app.config import MODELS_DIR, FEATURES_CSV, RESULTS_CSV, COMPARISON_CSV, TREE_MODELS

log = logging.getLogger(__name__)


class ModelRegistry:
    """
    In-memory store for all notebook artifacts.
    """

    def __init__(self):
        self.models: dict[str, Any] = {}           # filename_stem -> model object
        self.results_df: pd.DataFrame | None = None
        self.comparison_df: pd.DataFrame | None = None
        self.best_per_state: dict[str, str] = {}    # state -> best model name
        self.history_df: pd.DataFrame | None = None
        self.states: list[str] = []
        self.available_models: list[str] = []

    #  Public API

    def load_all(self) -> None:
        """
        Load every artifact from disk. Called once at startup.
        """
        self._load_history()
        self._load_results()
        self._load_models()
        self._compute_best_per_state()
        log.info(
            f"Registry ready: {len(self.states)} states, "
            f"{len(self.models)} model files, "
            f"{len(self.available_models)} model types"
        )

    def get_model(self, state: str, model_name: str) -> Any | None:
        """
        Return a fitted model for (model_name, state), or None.
        """
        key = f"{model_name}_{state}".replace(" ", "_").lower()
        return self.models.get(key)

    def get_best_model_name(self, state: str) -> str | None:
        """
        Return the best model name for a state (by lowest RMSE).
        """
        return self.best_per_state.get(state)

    def get_state_history(self, state: str) -> pd.DataFrame:
        """
        Return historical feature data for a single state.
        """
        if self.history_df is None:
            raise ValueError("No history data loaded. Run the notebook first.")
        df = self.history_df[self.history_df["State"] == state].copy()
        if df.empty:
            raise ValueError(f"State '{state}' not found in history data")
        return df.sort_values("Date").reset_index(drop=True)

    def get_comparison(self) -> dict:
        if self.comparison_df is None:
            return {}

        comp = self.comparison_df.copy()

        # OPTIONAL: attach dummy run_id (or real if stored)
        if "run_id" not in comp.columns:
            comp["run_id"] = "N/A"

        return comp.to_dict(orient="index")

    def get_state_metrics(self, state: str) -> list[dict]:
        """
        Return all model metrics for a specific state.
        """
        if self.results_df is None:
            return []
        rows = self.results_df[self.results_df["state"] == state]
        return rows.to_dict(orient="records")

    #  Private loaders 

    def _load_history(self) -> None:
        if not FEATURES_CSV.exists():
            log.warning(f"Features file not found: {FEATURES_CSV}")
            return
        self.history_df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
        self.states = sorted(self.history_df["State"].unique().tolist())
        log.info(f"Loaded history: {self.history_df.shape[0]} rows, {len(self.states)} states")

    def _load_results(self) -> None:
        if RESULTS_CSV.exists():
            self.results_df = pd.read_csv(RESULTS_CSV)
            self.available_models = sorted(self.results_df["model"].unique().tolist())
            log.info(f"Loaded results: {self.results_df.shape[0]} entries")
        else:
            log.warning(f"Results file not found: {RESULTS_CSV}")

        if COMPARISON_CSV.exists():
            self.comparison_df = pd.read_csv(COMPARISON_CSV, index_col=0)
            log.info(f"Loaded comparison: {self.comparison_df.shape[0]} models")

    def _load_models(self) -> None:
        if not MODELS_DIR.exists():
            log.warning(f"Models directory not found: {MODELS_DIR}")
            return
        for pkl_path in sorted(MODELS_DIR.glob("*.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    self.models[pkl_path.stem] = pickle.load(f)
                log.info(f"Loaded model: {pkl_path.name}")
            except Exception as e:
                log.warning(f"Failed to load {pkl_path.name}: {e}")

    def _compute_best_per_state(self) -> None:
        if self.results_df is None or self.results_df.empty:
            # fallback: if we have models but no results CSV, pick from loaded models
            if self.models:
                for key in self.models:
                    # key format: modelname_statename
                    # best-effort: mark whatever we have
                    pass
            return
        best = self.results_df.loc[self.results_df.groupby("state")["RMSE"].idxmin()]
        self.best_per_state = dict(zip(best["state"], best["model"]))
        log.info(f"Best-per-state mapping: {len(self.best_per_state)} states")
