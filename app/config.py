"""
All project configuration: paths to notebook artifacts, feature definitions,
forecast settings. No training / tuning configs — those live in the notebook.
"""

from pathlib import Path

#  Project root 
ROOT = Path(__file__).resolve().parent.parent

#  Data paths (notebook artifacts) 
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

RAW_EXCEL      = RAW_DIR / "Forecasting Case- Study.xlsx"
FEATURES_CSV   = PROCESSED_DIR / "features_clean.csv"
RESULTS_CSV    = PROCESSED_DIR / "all_model_results.csv"
COMPARISON_CSV = PROCESSED_DIR / "model_comparison.csv"

#  Trained models (saved by notebook) 
MODELS_DIR = ROOT / "models"

#  Forecast settings 
HORIZON = 8          # default weeks to forecast
FREQ    = "W-SUN"    # weekly frequency anchor

# MLflow tracking
MLFLOW_UI = "http://127.0.0.1:5000"
EXPERIMENT_ID = "1"   # change if needed

#  Feature columns (must match notebook output) 
FEATURE_COLS = [
    # lag features
    "lag_1", "lag_7", "lag_30",
    # rolling statistics
    "rmean_4", "rmean_8", "rmean_13", "rmean_26", "rmean_52",
    "rstd_4", "rstd_8", "rstd_13", "rstd_26", "rstd_52",
    "rmin_13", "rmax_13", "expanding_mean",
    # time features
    "year", "month", "week", "day_of_week", "quarter",
    "month_sin", "month_cos", "week_sin", "week_cos",
    # other
    "state_code", "is_holiday", "pct_wow", "pct_mom", "pct_yoy",
]

#  Model categories 
TREE_MODELS  = ["XGBoost", "Random Forest", "LightGBM", "Gradient Boosting"]
REFIT_MODELS = ["SARIMA", "Prophet"]  # no pickle — re-fit from history
ALL_MODELS   = TREE_MODELS + REFIT_MODELS

#  SARIMA defaults (used when re-fitting at inference) 
SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 0, 52)

#  US holidays (month, day) for feature building 
US_HOLIDAYS = [
    (1, 1), (1, 20), (2, 17), (5, 26), (7, 4),
    (9, 1), (10, 13), (11, 11), (11, 27), (12, 25),
]

# API 
API_HOST = "0.0.0.0"
API_PORT = 8000
