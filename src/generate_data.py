"""
Synthetic CRM Opportunity Data Generator
=========================================
Generates a realistic Salesforce-style opportunity dataset for a B2B SaaS
company, including seasonality, rep-level performance variation, stage
progression, and win/loss dynamics.

All data is synthetic. Run:  python src/generate_data.py
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42

SEGMENTS = {
    "Enterprise": {"share": 0.15, "acv_mean": 145_000, "acv_sd": 55_000, "win_rate": 0.22, "cycle_days": 120},
    "Mid-Market": {"share": 0.35, "acv_mean": 48_000, "acv_sd": 18_000, "win_rate": 0.28, "cycle_days": 75},
    "SMB": {"share": 0.50, "acv_mean": 12_000, "acv_sd": 5_000, "win_rate": 0.34, "cycle_days": 35},
}

REGIONS = {"AMER": 0.45, "EMEA": 0.33, "APAC": 0.22}

PRODUCTS = {
    "Platform Core": 0.42,
    "Analytics Add-on": 0.24,
    "Security Suite": 0.19,
    "API Enterprise": 0.15,
}

LEAD_SOURCES = {
    "Inbound - Website": 0.28,
    "Outbound - SDR": 0.24,
    "Partner Referral": 0.16,
    "Event / Conference": 0.14,
    "Customer Expansion": 0.18,
}

STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]

# Stage-to-stage conversion probabilities used for funnel realism
STAGE_CONVERSION = {
    "Prospecting": 0.62,
    "Qualification": 0.55,
    "Proposal": 0.58,
    "Negotiation": 0.60,
}

LOSS_REASONS = {
    "Price": 0.31,
    "Chose Competitor": 0.27,
    "No Decision / Stalled": 0.22,
    "Missing Features": 0.12,
    "Timing / Budget Cycle": 0.08,
}

REP_FIRST = ["Ava", "Liam", "Maya", "Noah", "Zara", "Ethan", "Ines", "Kai", "Lena", "Marco",
             "Priya", "Tomas", "Sofia", "Derek", "Hana", "Oscar", "Ruth", "Felix", "Nadia", "Jonas"]
REP_LAST = ["Chen", "Okafor", "Berzins", "Silva", "Novak", "Haas", "Kimura", "Duarte", "Lindqvist", "Moreau",
            "Petrov", "Alvarez", "Osei", "Vanags", "Keller", "Bianchi", "Ito", "Larsen", "Sharma", "Wolf"]

COMPANY_PREFIX = ["Apex", "Northwind", "Vertex", "Blue Harbor", "Ironclad", "Solstice", "Quantum", "Cobalt",
                  "Redwood", "Atlas", "Meridian", "Falcon", "Granite", "Lumen", "Pacific", "Summit",
                  "Orion", "Cascade", "Pinnacle", "Horizon", "Sterling", "Vantage", "Nimbus", "Keystone"]
COMPANY_SUFFIX = ["Systems", "Logistics", "Health", "Financial", "Retail Group", "Manufacturing", "Media",
                  "Technologies", "Energy", "Consulting", "Insurance", "Analytics", "Foods", "Robotics"]


def _weighted_choice(rng: np.random.Generator, mapping: dict, size: int) -> np.ndarray:
    keys = list(mapping.keys())
    probs = np.array([mapping[k] if isinstance(mapping[k], float) else mapping[k]["share"] for k in keys])
    probs = probs / probs.sum()
    return rng.choice(keys, size=size, p=probs)


def generate_opportunities(n: int = 6000, start: date = date(2023, 1, 1),
                           end: date = date(2026, 6, 30), seed: int = RNG_SEED) -> pd.DataFrame:
    """Generate a synthetic opportunity table with realistic dynamics."""
    rng = np.random.default_rng(seed)

    # --- Reps: each has a skill multiplier so performance varies by person
    reps = [f"{fn} {ln}" for fn, ln in zip(rng.choice(REP_FIRST, 20, replace=False),
                                       rng.choice(REP_LAST, 20, replace=False))]
    rep_skill = {r: float(np.clip(rng.normal(1.0, 0.18), 0.6, 1.5)) for r in reps}
    rep_region = {r: _weighted_choice(rng, REGIONS, 1)[0] for r in reps}

    # --- Created dates: upward trend + Q4 push + summer dip
    total_days = (end - start).days
    day_offsets = rng.integers(0, total_days, size=n * 3)  # oversample, then thin by seasonality
    created = np.array([start + timedelta(days=int(d)) for d in day_offsets])
    month = np.array([c.month for c in created])
    growth = 1.0 + (day_offsets / total_days) * 0.9              # pipeline grows ~90% over the window
    seasonal = np.where(np.isin(month, [11, 12]), 1.35,           # Q4 push
               np.where(np.isin(month, [7, 8]), 0.72, 1.0))       # summer dip
    keep_prob = (growth * seasonal)
    keep_prob = keep_prob / keep_prob.max()
    mask = rng.random(len(created)) < keep_prob
    created = created[mask][:n]
    n = len(created)

    segment = _weighted_choice(rng, SEGMENTS, n)
    seg_conf = np.array([[SEGMENTS[s]["acv_mean"], SEGMENTS[s]["acv_sd"],
                          SEGMENTS[s]["win_rate"], SEGMENTS[s]["cycle_days"]] for s in segment])

    amount = rng.lognormal(mean=np.log(seg_conf[:, 0]), sigma=0.42)
    amount = np.round(amount / 500) * 500  # round to $500 like real CRMs

    owner = rng.choice(reps, size=n)
    skill = np.array([rep_skill[o] for o in owner])

    # --- Outcome model: base win rate * rep skill * lead-source lift
    source = _weighted_choice(rng, LEAD_SOURCES, n)
    source_lift = np.where(source == "Customer Expansion", 1.45,
                  np.where(source == "Partner Referral", 1.2,
                  np.where(source == "Outbound - SDR", 0.85, 1.0)))
    p_win = np.clip(seg_conf[:, 2] * skill * source_lift, 0.02, 0.85)

    # --- Sales cycle length (lognormal around segment norm, winners close a bit faster)
    cycle = rng.lognormal(mean=np.log(seg_conf[:, 3]), sigma=0.45).astype(int) + 5

    close = np.array([c + timedelta(days=int(d)) for c, d in zip(created, cycle)])
    today = end

    is_closed = close <= np.array([today] * n)
    won = (rng.random(n) < p_win) & is_closed

    # --- Assign current stage
    stage = np.empty(n, dtype=object)
    stage[won] = "Closed Won"
    stage[is_closed & ~won] = "Closed Lost"
    open_idx = np.where(~is_closed)[0]
    # Open deals sit in a funnel-shaped distribution of stages
    stage[open_idx] = rng.choice(STAGES[:4], size=len(open_idx), p=[0.38, 0.27, 0.21, 0.14])

    loss_reason = np.where(stage == "Closed Lost",
                           _weighted_choice(rng, LOSS_REASONS, n), None)

    # Stage at which lost deals died (funnel realism: most die early)
    lost_at = rng.choice(STAGES[:4], size=n, p=[0.42, 0.28, 0.18, 0.12])
    lost_at_stage = np.where(stage == "Closed Lost", lost_at, None)

    # --- Slippage: some open deals have close dates pushed (a classic pipeline-health signal)
    pushed = (~is_closed) & (rng.random(n) < 0.22)
    push_days = rng.integers(15, 90, size=n)
    close = np.array([c + timedelta(days=int(p)) if pu else c
                      for c, p, pu in zip(close, push_days, pushed)])

    df = pd.DataFrame({
        "opportunity_id": [f"OPP-{100000 + i}" for i in range(n)],
        "account_name": [f"{rng.choice(COMPANY_PREFIX)} {rng.choice(COMPANY_SUFFIX)}" for _ in range(n)],
        "segment": segment,
        "region": [rep_region[o] for o in owner],
        "product": _weighted_choice(rng, PRODUCTS, n),
        "lead_source": source,
        "owner": owner,
        "created_date": pd.to_datetime(created),
        "close_date": pd.to_datetime(close),
        "stage": stage,
        "amount": amount,
        "sales_cycle_days": cycle,
        "close_date_pushed": pushed,
        "loss_reason": loss_reason,
        "lost_at_stage": lost_at_stage,
    })

    df["is_won"] = (df["stage"] == "Closed Won")
    df["is_closed"] = df["stage"].isin(["Closed Won", "Closed Lost"])
    df = df.sort_values("created_date").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CRM opportunity data.")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--out", type=str, default="data/opportunities.csv")
    args = parser.parse_args()

    df = generate_opportunities(n=args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    won = df[df.is_won]
    print(f"Wrote {len(df):,} opportunities -> {out}")
    print(f"  Date range : {df.created_date.min().date()} to {df.created_date.max().date()}")
    print(f"  Closed-won : {len(won):,} deals / ${won.amount.sum()/1e6:.1f}M")
    print(f"  Win rate   : {df[df.is_closed].is_won.mean():.1%} (closed deals)")


if __name__ == "__main__":
    main()
