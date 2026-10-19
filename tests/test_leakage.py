"""Leakage guard: features at a cutoff must not change when anything after the cutoff changes."""
import numpy as np
import pandas as pd
import pytest

from churnkit.features import (FEATURES, LeakageError, assert_point_in_time, build_features,
                               build_features_naive, check_no_forbidden)

CUTOFF = pd.Timestamp("2025-03-01")


def scramble_future(data, cutoff, seed=0):
    rng = np.random.default_rng(seed)
    d = {k: v.copy() for k, v in data.items()}
    fut = d["activity"]["event_time"] >= cutoff
    d["activity"].loc[fut, "sessions"] = rng.integers(0, 500, int(fut.sum()))
    fut = d["payments"]["event_time"] >= cutoff
    d["payments"].loc[fut, "status"] = "failed"
    extra = d["customers"][["customer_id"]].assign(event_time=cutoff + pd.Timedelta(days=1))
    d["tickets"] = pd.concat([d["tickets"], extra], ignore_index=True)
    d["plan_changes"] = pd.concat([d["plan_changes"], extra.assign(new_plan="basic")], ignore_index=True)
    d["cancellations"] = d["cancellations"].loc[d["cancellations"]["event_time"] < cutoff]
    c = d["customers"]
    c["status_at_extract"] = "cancelled"
    c["current_plan_at_extract"] = "premium"
    c["cancel_date"] = cutoff
    return d


def test_features_ignore_everything_after_cutoff(data):
    before = build_features(data, CUTOFF).sort_values("customer_id").reset_index(drop=True)
    after = build_features(scramble_future(data, CUTOFF), CUTOFF).sort_values("customer_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_naive_features_do_change(data):
    """The deliberately wrong builder reads the future, so the same perturbation moves it."""
    a = build_features_naive(data, CUTOFF).sort_values("customer_id").reset_index(drop=True)
    b = build_features_naive(scramble_future(data, CUTOFF), CUTOFF).sort_values("customer_id").reset_index(drop=True)
    assert not a[FEATURES].equals(b[FEATURES])


def test_forbidden_columns_rejected():
    with pytest.raises(LeakageError):
        check_no_forbidden(["tenure_days", "status_at_extract"])
    check_no_forbidden(FEATURES)


def test_point_in_time_assertion_catches_future_rows(data):
    with pytest.raises(LeakageError):
        assert_point_in_time({"activity": data["activity"]}, CUTOFF)


def test_feature_rows_are_eligible_customers_only(data):
    f = build_features(data, CUTOFF)
    c = data["customers"].set_index("customer_id")
    assert (c.loc[f["customer_id"], "signup_date"] < CUTOFF).all()
    known = data["cancellations"].loc[data["cancellations"]["event_time"] < CUTOFF, "customer_id"]
    assert not f["customer_id"].isin(set(known)).any()
