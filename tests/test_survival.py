import pytest

from churnkit.survival import kaplan_meier, survival_at, tenure_table
from churnkit import synth


def test_km_matches_hand_calculation():
    # times 1,2,2,3(c),4,5(c): S(1)=5/6, S(2)=5/6*3/5=0.5, S(4)=0.5*1/2=0.25
    km = kaplan_meier([1, 2, 2, 3, 4, 5], [1, 1, 1, 0, 1, 0])
    assert survival_at(km, 0) == 1.0
    assert survival_at(km, 1) == pytest.approx(5 / 6)
    assert survival_at(km, 2) == pytest.approx(0.5)
    assert survival_at(km, 3.5) == pytest.approx(0.5)
    assert survival_at(km, 4) == pytest.approx(0.25)
    assert km["n_at_risk"].tolist() == [6, 5, 3, 2, 1]


def test_km_without_censoring_is_empirical_survival():
    t = [3, 1, 4, 1, 5, 9, 2, 6]
    km = kaplan_meier(t, [1] * len(t))
    for x in range(0, 10):
        assert survival_at(km, x) == pytest.approx(sum(v > x for v in t) / len(t))


def test_km_rejects_bad_input():
    with pytest.raises(ValueError):
        kaplan_meier([], [])


def test_tenure_censors_active_customers(data):
    tt = tenure_table(data, synth.DATA_END)
    active = data["customers"]["status_at_extract"] == "active"
    assert (tt.loc[active.to_numpy(), "event"] == 0).all()
    assert (tt["duration_days"] >= 0).all()
