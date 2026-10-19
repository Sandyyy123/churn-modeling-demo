"""Churn label, observation/prediction windows and right-censoring.

Definitions used throughout (all dates are midnight timestamps):

* cutoff ``c``: the "as of" moment. Features may only use information with
  ``event_time < c``.
* eligible: signed up before ``c`` and no cancellation known before ``c``.
* prediction window ``[c, c + H)``: label = 1 if the customer cancels inside it.
* fully observed: the window ends on or before the extract, ``c + H <= data_end + 1 day``.
  If it is not fully observed and no cancellation was seen up to ``data_end``,
  the outcome is unknown (right-censored) and the row is excluded from the
  binary label, but kept for survival-style summaries with its partial duration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ONE_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class WindowSpec:
    horizon_days: int = 90
    lookback_days: int = 120


def eligible_customers(data: dict, cutoff) -> pd.DataFrame:
    """Customers active at ``cutoff`` using only information known before it."""
    cutoff = pd.Timestamp(cutoff)
    cust = data["customers"][["customer_id", "signup_date"]]
    known_cancel = data["cancellations"].loc[data["cancellations"]["event_time"] < cutoff, "customer_id"]
    mask = (cust["signup_date"] < cutoff) & ~cust["customer_id"].isin(set(known_cancel))
    return cust.loc[mask, ["customer_id"]].reset_index(drop=True)


def label_snapshot(data: dict, cutoff, horizon_days: int, data_end) -> pd.DataFrame:
    """One row per eligible customer: label (1/0/NaN), censored flag, duration and event for survival use."""
    cutoff = pd.Timestamp(cutoff)
    data_end = pd.Timestamp(data_end)
    window_end = cutoff + pd.Timedelta(days=horizon_days)
    ids = eligible_customers(data, cutoff)
    canc = data["cancellations"][["customer_id", "event_time"]].rename(columns={"event_time": "cancel_time"})
    df = ids.merge(canc, on="customer_id", how="left")
    observed_end = data_end + ONE_DAY
    fully_observed = window_end <= observed_end
    churn_in_window = df["cancel_time"].notna() & (df["cancel_time"] >= cutoff) & (df["cancel_time"] < window_end)
    label = np.where(churn_in_window, 1.0, np.where(fully_observed, 0.0, np.nan))
    df["label"] = label
    df["censored"] = np.isnan(label)
    stop = df["cancel_time"].where(df["cancel_time"].notna(), observed_end)
    df["duration_days"] = (stop - cutoff).dt.days.astype(int)
    df["event"] = df["cancel_time"].notna().astype(int)
    df["cutoff"] = cutoff
    return df.drop(columns=["cancel_time"])


def check_temporal_split(train_cutoffs, eval_cutoff, horizon_days: int) -> None:
    """Raise if any training label window reaches into the evaluation period.

    A training row's label is only known at ``cutoff + H``. If that is after the
    evaluation cutoff, the model would be trained on outcomes that happen during
    the evaluation period, which is temporal leakage.
    """
    eval_cutoff = pd.Timestamp(eval_cutoff)
    horizon = pd.Timedelta(days=horizon_days)
    bad = [str(pd.Timestamp(c).date()) for c in train_cutoffs if pd.Timestamp(c) + horizon > eval_cutoff]
    if bad:
        raise ValueError(f"Training label windows overlap the evaluation cutoff {eval_cutoff.date()}: {bad}")
