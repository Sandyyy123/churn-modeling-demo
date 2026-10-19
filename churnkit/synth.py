"""Synthetic subscription data generated in code with a fixed seed.

No real customer data is used. The generator produces the kind of tables a
subscription business usually has (customers, usage per billing cycle, support
tickets, payments, plan changes, cancellations) plus a few "extract-time" columns
that look useful but leak the future, so the leakage guard has something to catch.

Every event row carries ``event_time``: the moment that piece of information
became known. Point-in-time features may only use rows with
``event_time < cutoff``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA_START = pd.Timestamp("2023-01-01")
SIGNUP_END = pd.Timestamp("2025-10-31")
DATA_END = pd.Timestamp("2025-12-31")  # last observed day; anything later is unknown
CYCLE_DAYS = 30
PLANS = {"basic": 9.99, "standard": 19.99, "premium": 39.99}
PLAN_ORDER = list(PLANS)
CHANNELS = ["organic", "paid_social", "referral", "partner"]

# Columns that exist in the extract but must never feed a model built "as of" a past date.
FORBIDDEN_COLUMNS = frozenset({"status_at_extract", "current_plan_at_extract", "cancel_date", "reason"})


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_customers: int = 3000, seed: int = 20261020) -> dict[str, pd.DataFrame]:
    """Return a dict of DataFrames: customers, activity, tickets, payments, plan_changes, cancellations."""
    rng = np.random.default_rng(seed)
    span = (SIGNUP_END - DATA_START).days
    cust, act, tix, pay, plan_rows, canc = [], [], [], [], [], []

    for i in range(n_customers):
        cid = f"C{i + 1:05d}"
        signup = DATA_START + pd.Timedelta(days=int(rng.integers(0, span + 1)))
        level = int(rng.choice(3, p=[0.5, 0.35, 0.15]))
        signup_plan = PLAN_ORDER[level]
        channel = str(rng.choice(CHANNELS, p=[0.4, 0.3, 0.2, 0.1]))
        promo = bool(rng.random() < 0.25)
        engagement = float(rng.normal())
        decline_from = int(rng.integers(2, 18)) if rng.random() < 0.35 else None
        base_sessions = float(np.exp(2.3 + 0.45 * engagement))
        cancel_date, reason = None, None
        prev_sessions, prev_failed, prev_tickets = None, False, 0
        k = 0
        while True:
            start = signup + pd.Timedelta(days=CYCLE_DAYS * k)
            if start > DATA_END:
                break
            end = start + pd.Timedelta(days=CYCLE_DAYS)
            failed = bool(rng.random() < 0.02 + 0.04 * (channel == "paid_social") + 0.03 * (engagement < -1))
            pay.append((cid, start, "failed" if failed else "paid"))
            mult = 1.0
            if decline_from is not None and k >= decline_from:
                mult = float(np.exp(-0.35 * (k - decline_from + 1)))
            sessions = int(rng.poisson(base_sessions * mult))
            n_tickets = int(rng.poisson(0.12 + 0.35 * (sessions < 3) + 0.2 * failed))
            for _ in range(n_tickets):
                t = start + pd.Timedelta(days=int(rng.integers(0, CYCLE_DAYS)))
                if t <= DATA_END:
                    tix.append((cid, t))
            downgraded = False
            if level > 0 and rng.random() < 0.015 + 0.05 * (sessions < 3):
                level -= 1
                downgraded = True
                t = start + pd.Timedelta(days=int(rng.integers(0, CYCLE_DAYS)))
                if t <= DATA_END:
                    plan_rows.append((cid, t, PLAN_ORDER[level]))
            low_before = prev_sessions is not None and prev_sessions <= 3
            z = (-4.3 - 0.6 * engagement + 1.0 * (sessions <= 2) + 0.9 * low_before + 0.35 * n_tickets
                 + 0.4 * prev_tickets + 1.0 * failed + 0.9 * prev_failed + 1.0 * (promo and k == 3)
                 + 0.35 * (level == 0) - 0.03 * min(k, 24) + 0.7 * downgraded)
            if rng.random() < _sigmoid(z):
                c = start + pd.Timedelta(days=int(rng.integers(0, CYCLE_DAYS)))
                if c <= DATA_END:
                    cancel_date = c
                    reason = str(rng.choice(["price", "low_use", "support", "payment", "other"]))
                    frac = max((c - start).days, 1) / CYCLE_DAYS
                    act.append((cid, start, c, int(round(sessions * frac))))
                break
            if end > DATA_END:  # the current cycle is not complete at the extract date
                break
            act.append((cid, start, end, sessions))
            prev_sessions, prev_failed, prev_tickets = sessions, failed, n_tickets
            k += 1

        status = "cancelled" if cancel_date is not None else "active"
        cust.append((cid, signup, signup_plan, channel, promo, PLAN_ORDER[level], status, cancel_date))
        if cancel_date is not None:
            canc.append((cid, cancel_date, reason))

    customers = pd.DataFrame(cust, columns=["customer_id", "signup_date", "signup_plan", "channel", "promo",
                                            "current_plan_at_extract", "status_at_extract", "cancel_date"])
    return {
        "customers": customers,
        "activity": pd.DataFrame(act, columns=["customer_id", "period_start", "event_time", "sessions"]),
        "tickets": pd.DataFrame(tix, columns=["customer_id", "event_time"]),
        "payments": pd.DataFrame(pay, columns=["customer_id", "event_time", "status"]),
        "plan_changes": pd.DataFrame(plan_rows, columns=["customer_id", "event_time", "new_plan"]),
        "cancellations": pd.DataFrame(canc, columns=["customer_id", "event_time", "reason"]),
    }
