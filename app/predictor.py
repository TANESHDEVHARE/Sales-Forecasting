"""
Generate N-week ahead forecasts for a given state.

Strategies:
  - Tree models (XGBoost, RF, LightGBM, GB): recursive prediction.
    Predict t+1 -> append to history -> predict t+2 ->...-> t+N.
  - SARIMA: re-fit on full history, call .forecast(N).
  - Prophet: re-fit on full history, predict future dates.
"""

import logging

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLS, TREE_MODELS, HORIZON, FREQ,
    SARIMA_ORDER, SARIMA_SEASONAL_ORDER, US_HOLIDAYS,
)

log = logging.getLogger(__name__)


#  Holiday helper 

def _near_holiday(date: pd.Timestamp, window: int = 3) -> int:
    """
    Return 1 if date is within `window` days of any US holiday.
    """
    for m, d in US_HOLIDAYS:
        try:
            h = pd.Timestamp(year=date.year, month=m, day=d)
            if abs((date - h).days) <= window:
                return 1
        except ValueError:
            pass
    return 0


#  Build one future row 

def _build_next_row(last_row: pd.Series, next_date: pd.Timestamp, pred: float) -> pd.Series:
    """
    Construct feature values for the next time step using the most recent
    row and the predicted value as the new 'Total'.
    """
    row = last_row.copy()
    row["Date"] = next_date
    row["Total"] = pred

    # time features
    row["year"] = next_date.year
    row["month"] = next_date.month
    row["week"] = int(next_date.isocalendar().week)
    row["day_of_week"] = next_date.dayofweek
    row["quarter"] = next_date.quarter
    row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)
    row["week_sin"] = np.sin(2 * np.pi * row["week"] / 52)
    row["week_cos"] = np.cos(2 * np.pi * row["week"] / 52)

    # shift lags forward
    row["lag_1"] = pred
    row["lag_7"] = last_row.get("lag_7", pred)
    row["lag_30"] = last_row.get("lag_30", pred)

    # holiday flag
    row["is_holiday"] = _near_holiday(next_date)

    return row


#  Tree model (recursive) 

def _forecast_tree(model, history: pd.DataFrame, state: str, horizon: int) -> pd.DataFrame:
    """
    Recursive forecast for any sklearn-compatible model.
    """
    last_date = history["Date"].max()
    last_row = history.iloc[-1].copy()

    results = []
    for step in range(1, horizon + 1):
        next_date = last_date + pd.DateOffset(weeks=step)
        future = _build_next_row(last_row, next_date, last_row["Total"])

        X = pd.DataFrame([future[FEATURE_COLS]])
        pred = float(model.predict(X)[0])

        results.append({"date": next_date, "state": state, "predicted_sales": round(pred, 2)})

        future["Total"] = pred
        last_row = future

    return pd.DataFrame(results)


#  SARIMA (re-fit) 

def _forecast_sarima(history: pd.DataFrame, state: str, horizon: int) -> pd.DataFrame:
    """
    Fit SARIMA on full history and forecast ahead.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = history.sort_values("Date")["Total"].values
    model = SARIMAX(
        y,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=300)
    preds = fit.forecast(steps=horizon)

    last_date = history["Date"].max()
    dates = [last_date + pd.DateOffset(weeks=i) for i in range(1, horizon + 1)]

    return pd.DataFrame({
        "date": dates,
        "state": state,
        "predicted_sales": [round(float(p), 2) for p in preds],
    })


#  Prophet (re-fit) 

def _forecast_prophet(history: pd.DataFrame, state: str, horizon: int) -> pd.DataFrame:
    """
    Fit Prophet on full history and forecast ahead.
    """
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
    from prophet import Prophet

    train = history[["Date", "Total"]].rename(columns={"Date": "ds", "Total": "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train)

    future = m.make_future_dataframe(periods=horizon, freq=FREQ)
    fcast = m.predict(future).tail(horizon)

    return pd.DataFrame({
        "date": fcast["ds"].values,
        "state": state,
        "predicted_sales": [round(float(v), 2) for v in fcast["yhat"].values],
    })


#  Main entry point 

def forecast(registry, state: str, horizon: int = HORIZON, model_name: str | None = None) -> dict:
    """
    Generate a multi-week forecast for one state.

    Parameters
    ----------
    registry   : ModelRegistry instance (holds models + history)
    state      : US state name
    horizon    : weeks ahead (default 8)
    model_name : model to use (auto-selects best if None)

    Returns
    -------
    dict with keys: state, model_name, horizon, forecasts (list of dicts)
    """
    # resolve model
    if model_name is None:
        model_name = registry.get_best_model_name(state)
    if model_name is None:
        # fallback: pick first available tree model that has a pickle
        for tm in TREE_MODELS:
            if registry.get_model(state, tm) is not None:
                model_name = tm
                break
    if model_name is None:
        model_name = "SARIMA"  # ultimate fallback

    history = registry.get_state_history(state)

    log.info(f"Forecasting {horizon}w for {state} with {model_name}")

    if model_name in TREE_MODELS:
        model_obj = registry.get_model(state, model_name)
        if model_obj is None:
            raise ValueError(
                f"No saved model file for {model_name}/{state}. "
                f"Run the notebook training cells first."
            )
        df = _forecast_tree(model_obj, history, state, horizon)

    elif model_name == "SARIMA":
        df = _forecast_sarima(history, state, horizon)

    elif model_name == "Prophet":
        df = _forecast_prophet(history, state, horizon)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return {
        "state": state,
        "model_name": model_name,
        "horizon": horizon,
        "forecasts": df.to_dict(orient="records"),
    }
