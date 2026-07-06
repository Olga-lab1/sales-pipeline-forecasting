"""
Pipeline health metrics: the KPIs a CRO reviews in a forecast call.
Every function takes the opportunity DataFrame and returns tidy,
chart-ready outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_ORDER = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won"]

# Probability-weighting used for weighted pipeline (standard CRM convention)
STAGE_WEIGHTS = {
    "Prospecting": 0.10,
    "Qualification": 0.25,
    "Proposal": 0.50,
    "Negotiation": 0.75,
    "Closed Won": 1.00,
    "Closed Lost": 0.00,
}


def load_opportunities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["created_date", "close_date"])
    df["is_won"] = df["stage"] == "Closed Won"
    df["is_closed"] = df["stage"].isin(["Closed Won", "Closed Lost"])
    return df


def headline_kpis(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Executive summary metrics."""
    as_of = as_of or df["created_date"].max()
    open_df = df[~df["is_closed"]]
    closed = df[df["is_closed"]]
    won = df[df["is_won"]]

    qtr_start = pd.Timestamp(as_of).to_period("Q").start_time
    won_this_qtr = won[won["close_date"] >= qtr_start]

    weighted = (open_df["amount"] * open_df["stage"].map(STAGE_WEIGHTS)).sum()

    return {
        "open_pipeline": float(open_df["amount"].sum()),
        "weighted_pipeline": float(weighted),
        "open_deals": int(len(open_df)),
        "win_rate": float(closed["is_won"].mean()) if len(closed) else 0.0,
        "avg_deal_size": float(won["amount"].mean()) if len(won) else 0.0,
        "avg_cycle_days": float(won["sales_cycle_days"].mean()) if len(won) else 0.0,
        "bookings_this_qtr": float(won_this_qtr["amount"].sum()),
        "slipped_deals_pct": float(open_df["close_date_pushed"].mean()) if len(open_df) else 0.0,
    }


def monthly_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Closed-won revenue by month — the series we forecast."""
    won = df[df["is_won"]].copy()
    won["month"] = won["close_date"].dt.to_period("M").dt.to_timestamp()
    out = won.groupby("month", as_index=False)["amount"].sum()
    out = out.rename(columns={"amount": "bookings"})
    # Drop the last partial month so the forecast trains on complete months only
    return out.iloc[:-1] if len(out) > 1 else out


def funnel_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Share of all opportunities that reached each stage, using the
    furthest stage a deal got to (open stage, lost_at_stage, or Closed Won)."""
    idx = {s: i for i, s in enumerate(STAGE_ORDER)}

    def furthest(row) -> int:
        if row["stage"] == "Closed Won":
            return idx["Closed Won"]
        if row["stage"] == "Closed Lost":
            return idx.get(row.get("lost_at_stage"), 0)
        return idx.get(row["stage"], 0)

    reached_idx = df.apply(furthest, axis=1)
    total = len(df)
    rows = []
    for i, stage in enumerate(STAGE_ORDER):
        cnt = int((reached_idx >= i).sum())
        rows.append({"stage": stage, "deals": cnt, "pct_of_created": cnt / total})
    return pd.DataFrame(rows)


def win_rate_by(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    closed = df[df["is_closed"]]
    g = closed.groupby(dim).agg(
        deals=("opportunity_id", "count"),
        win_rate=("is_won", "mean"),
        avg_amount=("amount", "mean"),
        revenue=("amount", lambda s: s[closed.loc[s.index, "is_won"]].sum()),
    ).reset_index()
    return g.sort_values("revenue", ascending=False)


def rep_leaderboard(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    closed = df[df["is_closed"]]
    g = closed.groupby("owner").agg(
        closed_deals=("opportunity_id", "count"),
        win_rate=("is_won", "mean"),
        revenue=("amount", lambda s: s[closed.loc[s.index, "is_won"]].sum()),
        avg_cycle=("sales_cycle_days", "mean"),
    ).reset_index()
    return g.sort_values("revenue", ascending=False).head(top_n)


def pipeline_coverage(df: pd.DataFrame, quota: float) -> dict:
    """Coverage ratio = open weighted pipeline / remaining quota.
    Rule of thumb: healthy is 3x+ unweighted, ~1x weighted."""
    open_df = df[~df["is_closed"]]
    raw = open_df["amount"].sum()
    weighted = (open_df["amount"] * open_df["stage"].map(STAGE_WEIGHTS)).sum()
    return {
        "raw_coverage": raw / quota if quota else np.nan,
        "weighted_coverage": weighted / quota if quota else np.nan,
        "raw_pipeline": float(raw),
        "weighted_pipeline": float(weighted),
    }


def loss_reasons(df: pd.DataFrame) -> pd.DataFrame:
    lost = df[df["stage"] == "Closed Lost"]
    g = lost.groupby("loss_reason").agg(
        deals=("opportunity_id", "count"),
        lost_revenue=("amount", "sum"),
    ).reset_index().sort_values("lost_revenue", ascending=False)
    return g
