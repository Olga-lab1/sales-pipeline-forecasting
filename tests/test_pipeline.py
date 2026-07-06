"""Tests for data generation, metrics, and forecasting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_data import generate_opportunities  # noqa: E402
from metrics import (STAGE_WEIGHTS, funnel_conversion, headline_kpis,  # noqa: E402
                     monthly_bookings, pipeline_coverage, rep_leaderboard)
from forecasting import backtest_mape, forecast_bookings, quarter_projection  # noqa: E402


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_opportunities(n=3000, seed=7)


def test_generation_shape_and_columns(df):
    assert len(df) > 2000
    for col in ["opportunity_id", "segment", "stage", "amount",
                "created_date", "close_date", "is_won", "lost_at_stage"]:
        assert col in df.columns
    assert df["opportunity_id"].is_unique


def test_generation_is_deterministic():
    a = generate_opportunities(n=500, seed=1)
    b = generate_opportunities(n=500, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_amounts_positive_and_realistic(df):
    assert (df["amount"] > 0).all()
    # Enterprise deals should be larger than SMB on average
    seg_avg = df.groupby("segment")["amount"].mean()
    assert seg_avg["Enterprise"] > seg_avg["Mid-Market"] > seg_avg["SMB"]


def test_win_rate_in_plausible_band(df):
    closed = df[df["is_closed"]]
    assert 0.15 < closed["is_won"].mean() < 0.55


def test_headline_kpis_keys_and_types(df):
    k = headline_kpis(df)
    assert set(k) >= {"open_pipeline", "weighted_pipeline", "win_rate",
                      "bookings_this_qtr", "slipped_deals_pct"}
    assert k["open_pipeline"] >= k["weighted_pipeline"] >= 0


def test_funnel_is_monotonically_decreasing(df):
    fn = funnel_conversion(df)
    assert fn["deals"].is_monotonic_decreasing
    assert fn["pct_of_created"].iloc[0] == 1.0


def test_stage_weights_cover_all_stages(df):
    assert set(df["stage"].unique()) <= set(STAGE_WEIGHTS)


def test_pipeline_coverage_math(df):
    cov = pipeline_coverage(df, quota=1_000_000)
    assert cov["raw_coverage"] == pytest.approx(cov["raw_pipeline"] / 1_000_000)
    assert cov["raw_pipeline"] >= cov["weighted_pipeline"]


def test_leaderboard_sorted(df):
    lb = rep_leaderboard(df)
    assert lb["revenue"].is_monotonic_decreasing
    assert len(lb) <= 10


def test_forecast_shape_and_intervals(df):
    monthly = monthly_bookings(df)
    fc = forecast_bookings(monthly, horizon=6)
    assert len(fc) == 6
    assert (fc["lo80"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["hi80"]).all()
    assert (fc["lo80"] >= 0).all()


def test_backtest_returns_finite_mape(df):
    monthly = monthly_bookings(df)
    mape = backtest_mape(monthly)
    assert mape == mape  # not NaN
    assert 0 < mape < 1.5


def test_quarter_projection_bounds(df):
    monthly = monthly_bookings(df)
    fc = forecast_bookings(monthly, horizon=6)
    qp = quarter_projection(monthly, fc, df["created_date"].max())
    assert qp["projected_lo"] <= qp["projected_total"] <= qp["projected_hi"]
