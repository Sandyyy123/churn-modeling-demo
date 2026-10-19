"""Kaplan-Meier retention curves with right-censoring (implemented directly, no extra dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier(durations, events) -> pd.DataFrame:
    """Product-limit estimator.

    ``durations``: time to cancellation or to censoring; ``events``: 1 = cancelled, 0 = censored.
    Returns one row per distinct time with n_at_risk, n_events, n_censored and survival S(t).
    At tied times, events are counted before censorings (the usual convention).
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    if d.shape != e.shape or d.size == 0:
        raise ValueError("durations and events must be non-empty and the same length")
    rows, s = [], 1.0
    at_risk = d.size
    for t in np.unique(d):
        at_t = d == t
        n_ev = int(e[at_t].sum())
        n_c = int(at_t.sum()) - n_ev
        if n_ev:
            s *= 1.0 - n_ev / at_risk
        rows.append((t, at_risk, n_ev, n_c, s))
        at_risk -= n_ev + n_c
    return pd.DataFrame(rows, columns=["time", "n_at_risk", "n_events", "n_censored", "survival"])


def survival_at(km: pd.DataFrame, t: float) -> float:
    """S(t) as a step function (1.0 before the first event time)."""
    before = km.loc[km["time"] <= t, "survival"]
    return float(before.iloc[-1]) if len(before) else 1.0


def tenure_table(data: dict, data_end) -> pd.DataFrame:
    """Tenure from signup to cancellation, censored at the extract date for active customers."""
    c = data["customers"][["customer_id", "signup_date", "signup_plan"]]
    canc = data["cancellations"][["customer_id", "event_time"]]
    df = c.merge(canc, on="customer_id", how="left")
    stop = df["event_time"].fillna(pd.Timestamp(data_end) + pd.Timedelta(days=1))
    df["duration_days"] = (stop - df["signup_date"]).dt.days
    df["event"] = df["event_time"].notna().astype(int)
    return df[["customer_id", "signup_plan", "duration_days", "event"]]
