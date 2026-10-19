import numpy as np
import pandas as pd
import pytest

from churnkit.windows import check_temporal_split, eligible_customers, label_snapshot

T = pd.Timestamp


def tiny():
    customers = pd.DataFrame({"customer_id": ["a", "b", "c", "d", "e", "f"],
                              "signup_date": [T("2025-01-01")] * 5 + [T("2025-06-01")]})
    cancellations = pd.DataFrame({"customer_id": ["a", "b", "c", "d"],
                                  "event_time": [T("2025-04-30"), T("2025-05-01"), T("2025-07-30"), T("2025-07-29")],
                                  "reason": ["x"] * 4})
    return {"customers": customers, "cancellations": cancellations}


def test_eligibility_uses_only_past_cancellations():
    ids = set(eligible_customers(tiny(), "2025-05-01")["customer_id"])
    # a cancelled before the cutoff; f signed up after it
    assert ids == {"b", "c", "d", "e"}


def test_label_boundaries():
    lab = label_snapshot(tiny(), "2025-05-01", 89, data_end="2025-12-31").set_index("customer_id")
    # window is [2025-05-01, 2025-07-29)
    assert lab.at["b", "label"] == 1  # cancels on the cutoff day: inside the window
    assert lab.at["d", "label"] == 0  # cancels on the window end day: outside (half-open window)
    assert lab.at["c", "label"] == 0
    assert lab.at["e", "label"] == 0
    assert not lab["censored"].any()


def test_right_censoring_when_window_not_observed():
    lab = label_snapshot(tiny(), "2025-11-01", 90, data_end="2025-12-31").set_index("customer_id")
    # nobody cancels after the cutoff and the window runs past the extract date
    assert lab["censored"].all()
    assert np.isnan(lab["label"]).all()
    assert (lab["duration_days"] == (T("2026-01-01") - T("2025-11-01")).days).all()
    assert (lab["event"] == 0).all()


def test_churn_inside_partial_window_is_still_a_positive():
    d = tiny()
    d["cancellations"].loc[len(d["cancellations"])] = ["e", T("2025-12-10"), "x"]
    lab = label_snapshot(d, "2025-11-01", 90, data_end="2025-12-31").set_index("customer_id")
    assert lab.at["e", "label"] == 1 and not lab.at["e", "censored"]


def test_temporal_split_guard():
    check_temporal_split([T("2025-01-01"), T("2025-04-01")], T("2025-07-01"), 90)
    with pytest.raises(ValueError):
        check_temporal_split([T("2025-01-01"), T("2025-05-01")], T("2025-07-01"), 90)
