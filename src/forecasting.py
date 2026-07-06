"""
Revenue forecasting with Holt-Winters exponential smoothing.

Why Holt-Winters over ARIMA/Prophet here:
 - Captures trend + yearly seasonality, the two dominant patterns in bookings.
 - Fast, dependency-light, and explainable to a non-technical audience.
 - Empirical residual-based intervals avoid over-promising precision.

The module also backtests the model (rolling-origin) so the dashboard can
show honest accuracy (MAPE) instead of an unvalidated line.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")  # statsmodels convergence chatter


def _fit(series: pd.Series) -> ExponentialSmoothing:
    seasonal = "add" if len(series) >= 24 else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def forecast_bookings(monthly: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Forecast `horizon` months ahead with 80% empirical intervals.

    Parameters
    ----------
    monthly : DataFrame with columns [month, bookings] (complete months only)
    horizon : months to forecast
    """
    series = monthly.set_index("month")["bookings"].asfreq("MS").fillna(0.0)
    fit = _fit(series)

    fc = fit.forecast(horizon)
    resid = series - fit.fittedvalues
    # Empirical 80% interval from in-sample residuals, widened with horizon
    q = np.quantile(resid.dropna(), [0.10, 0.90])
    widen = np.sqrt(np.arange(1, horizon + 1))

    out = pd.DataFrame({
        "month": fc.index,
        "forecast": fc.values,
        "lo80": np.maximum(fc.values + q[0] * widen, 0),
        "hi80": fc.values + q[1] * widen,
    })
    return out


def backtest_mape(monthly: pd.DataFrame, folds: int = 6, horizon: int = 3) -> float:
    """Rolling-origin backtest: average MAPE across `folds` cut points."""
    series = monthly.set_index("month")["bookings"].asfreq("MS").fillna(0.0)
    if len(series) < 18 + folds:
        folds = max(1, len(series) - 18)

    errs = []
    for k in range(folds, 0, -1):
        cut = len(series) - k - horizon + 1
        if cut < 15:
            continue
        train, test = series.iloc[:cut], series.iloc[cut:cut + horizon]
        try:
            pred = _fit(train).forecast(len(test))
            mask = test.values > 0
            if mask.any():
                errs.append(np.mean(np.abs((test.values[mask] - pred.values[mask]) / test.values[mask])))
        except Exception:
            continue
    return float(np.mean(errs)) if errs else float("nan")


def quarter_projection(monthly: pd.DataFrame, fc: pd.DataFrame,
                       as_of: pd.Timestamp) -> dict:
    """Project current-quarter total = actuals to date + forecast for the rest."""
    q = pd.Timestamp(as_of).to_period("Q")
    q_months = pd.period_range(q.start_time, q.end_time, freq="M").to_timestamp()

    actual = monthly[monthly["month"].isin(q_months)]["bookings"].sum()
    remaining = fc[fc["month"].isin(q_months)]
    return {
        "quarter": str(q),
        "actual_to_date": float(actual),
        "projected_total": float(actual + remaining["forecast"].sum()),
        "projected_lo": float(actual + remaining["lo80"].sum()),
        "projected_hi": float(actual + remaining["hi80"].sum()),
    }
