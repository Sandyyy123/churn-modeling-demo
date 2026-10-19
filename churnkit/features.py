"""Point-in-time feature construction with a leakage guard.

``build_features`` reads only an explicit allow-list of customer attributes and
event rows whose ``event_time`` is strictly before the cutoff. ``assert_point_in_time``
re-checks every table it consumed, and ``check_no_forbidden`` rejects extract-time
columns (current status, cancel date, cancellation reason, current plan).

``build_features_naive`` is deliberately wrong: it aggregates the whole extract,
as a quick notebook often does. It exists only to show how much leakage inflates
test metrics; it is never used for scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import FORBIDDEN_COLUMNS, PLANS
from .windows import eligible_customers

CUSTOMER_ALLOWLIST = ["customer_id", "signup_date", "signup_plan", "channel", "promo"]
NUMERIC_FEATURES = [
    "tenure_days", "monthly_price", "sessions_last_cycle", "sessions_prev_avg", "usage_trend",
    "zero_usage_cycles_90d", "tickets_90d", "failed_payments_90d", "downgrades_90d", "promo_active",
]
CATEGORICAL_FEATURES = ["plan", "channel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class LeakageError(AssertionError):
    pass


def check_no_forbidden(columns) -> None:
    hit = sorted(set(columns) & FORBIDDEN_COLUMNS)
    if hit:
        raise LeakageError(f"Extract-time columns must not be used as features: {hit}")


def asof(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Rows known strictly before ``cutoff``."""
    return df.loc[df["event_time"] < pd.Timestamp(cutoff)]


def assert_point_in_time(tables: dict, cutoff) -> None:
    cutoff = pd.Timestamp(cutoff)
    for name, df in tables.items():
        if len(df) and df["event_time"].max() >= cutoff:
            raise LeakageError(f"{name} contains rows at or after the cutoff {cutoff.date()}")


def build_features(data: dict, cutoff, lookback_days: int = 120) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    ids = eligible_customers(data, cutoff)
    cust = data["customers"][CUSTOMER_ALLOWLIST]
    check_no_forbidden(cust.columns)
    used = {k: asof(data[k], cutoff) for k in ("activity", "tickets", "payments", "plan_changes")}
    assert_point_in_time(used, cutoff)

    f = ids.merge(cust, on="customer_id", how="left")
    f["tenure_days"] = (cutoff - f["signup_date"]).dt.days

    plan_now = used["plan_changes"].sort_values("event_time").groupby("customer_id")["new_plan"].last()
    f["plan"] = f["customer_id"].map(plan_now).fillna(f["signup_plan"])
    f["monthly_price"] = f["plan"].map(PLANS)

    act = used["activity"].sort_values("event_time")
    last = act.groupby("customer_id")["sessions"].last()
    f["sessions_last_cycle"] = f["customer_id"].map(last)
    win = act.loc[act["event_time"] >= cutoff - pd.Timedelta(days=lookback_days)]
    prev = win.groupby("customer_id")["sessions"].apply(lambda s: s.iloc[:-1].mean() if len(s) > 1 else np.nan)
    f["sessions_prev_avg"] = f["customer_id"].map(prev)
    f["usage_trend"] = (f["sessions_last_cycle"] + 1) / (f["sessions_prev_avg"] + 1)

    d90 = cutoff - pd.Timedelta(days=90)
    recent = act.loc[act["event_time"] >= d90]
    f["zero_usage_cycles_90d"] = f["customer_id"].map(recent.loc[recent["sessions"] == 0].groupby("customer_id").size()).fillna(0)
    t = used["tickets"]
    f["tickets_90d"] = f["customer_id"].map(t.loc[t["event_time"] >= d90].groupby("customer_id").size()).fillna(0)
    p = used["payments"]
    pf = p.loc[(p["event_time"] >= d90) & (p["status"] == "failed")]
    f["failed_payments_90d"] = f["customer_id"].map(pf.groupby("customer_id").size()).fillna(0)
    pc = used["plan_changes"]
    f["downgrades_90d"] = f["customer_id"].map(pc.loc[pc["event_time"] >= d90].groupby("customer_id").size()).fillna(0)
    f["promo_active"] = (f["promo"] & (f["tenure_days"] < 120)).astype(int)
    f["cutoff"] = cutoff
    out = f[["customer_id", "cutoff"] + FEATURES].copy()
    check_no_forbidden(out.columns)
    return out


def build_features_naive(data: dict, cutoff) -> pd.DataFrame:
    """WRONG ON PURPOSE: whole-extract aggregates and the extract-time plan. Demo of leakage only."""
    cutoff = pd.Timestamp(cutoff)
    ids = eligible_customers(data, cutoff)
    c = data["customers"]
    f = ids.merge(c, on="customer_id", how="left")
    f["tenure_days"] = (cutoff - f["signup_date"]).dt.days
    f["plan"] = f["current_plan_at_extract"]
    f["monthly_price"] = f["plan"].map(PLANS)
    a = data["activity"].sort_values("event_time")
    f["sessions_last_cycle"] = f["customer_id"].map(a.groupby("customer_id")["sessions"].last())
    f["sessions_prev_avg"] = f["customer_id"].map(a.groupby("customer_id")["sessions"].mean())
    f["usage_trend"] = (f["sessions_last_cycle"] + 1) / (f["sessions_prev_avg"] + 1)
    f["zero_usage_cycles_90d"] = f["customer_id"].map(a.loc[a["sessions"] == 0].groupby("customer_id").size()).fillna(0)
    f["tickets_90d"] = f["customer_id"].map(data["tickets"].groupby("customer_id").size()).fillna(0)
    p = data["payments"]
    f["failed_payments_90d"] = f["customer_id"].map(p.loc[p["status"] == "failed"].groupby("customer_id").size()).fillna(0)
    f["downgrades_90d"] = f["customer_id"].map(data["plan_changes"].groupby("customer_id").size()).fillna(0)
    f["promo_active"] = (f["promo"] & (f["tenure_days"] < 120)).astype(int)
    f["cutoff"] = cutoff
    return f[["customer_id", "cutoff"] + FEATURES].copy()
