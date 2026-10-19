import pandas as pd

from churnkit import synth


def test_same_seed_same_data():
    a = synth.generate(150, seed=3)
    b = synth.generate(150, seed=3)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_different_seed_differs():
    a = synth.generate(150, seed=3)["activity"]
    b = synth.generate(150, seed=4)["activity"]
    assert not a.equals(b)


def test_nothing_after_extract_date(data):
    for k in ("activity", "tickets", "payments", "plan_changes", "cancellations"):
        if len(data[k]):
            assert data[k]["event_time"].max() <= synth.DATA_END


def test_cancellations_match_customer_status(data):
    c = data["customers"]
    cancelled = set(c.loc[c["status_at_extract"] == "cancelled", "customer_id"])
    assert cancelled == set(data["cancellations"]["customer_id"])
    assert data["cancellations"]["customer_id"].is_unique
