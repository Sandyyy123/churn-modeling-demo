import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from churnkit import pipeline as P
from churnkit.evaluate import decile_table, expected_calibration_error, metrics, reliability_table
from churnkit.features import FEATURES
from churnkit.models import Calibrated, PriorModel, make_gbm, make_logistic


@pytest.fixture(scope="module")
def frames(data):
    train = P.labelled(pd.concat([P.snapshot(data, c) for c in P.TRAIN_CUTOFFS], ignore_index=True))
    calib = P.labelled(pd.concat([P.snapshot(data, c) for c in P.CALIB_CUTOFFS], ignore_index=True))
    test = P.labelled(P.snapshot(data, P.TEST_CUTOFF))
    return train, calib, test


def test_models_beat_the_prior(frames):
    train, _, test = frames
    prior = PriorModel().fit(train[FEATURES], train["label"])
    p0 = metrics(test["label"], prior.predict_proba(test[FEATURES])[:, 1])
    for m in (make_logistic(), make_gbm()):
        m.fit(train[FEATURES], train["label"])
        r = metrics(test["label"], m.predict_proba(test[FEATURES])[:, 1])
        assert r["roc_auc"] > 0.6 > p0["roc_auc"]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_does_not_worsen_in_sample_brier(frames, method):
    train, calib, _ = frames
    g = make_gbm().fit(train[FEATURES], train["label"])
    raw = g.predict_proba(calib[FEATURES])[:, 1]
    cal = Calibrated(g, method).fit(calib[FEATURES], calib["label"]).predict_proba(calib[FEATURES])[:, 1]
    assert ((cal >= 0) & (cal <= 1)).all()
    assert brier_score_loss(calib["label"], cal) <= brier_score_loss(calib["label"], raw) + 1e-9


def test_sigmoid_calibration_keeps_ranking(frames):
    train, calib, test = frames
    g = make_gbm().fit(train[FEATURES], train["label"])
    c = Calibrated(g, "sigmoid").fit(calib[FEATURES], calib["label"])
    raw = c.raw_score(test[FEATURES])
    cal = c.predict_proba(test[FEATURES])[:, 1]
    order = np.argsort(raw, kind="mergesort")
    assert np.all(np.diff(cal[order]) >= -1e-12)


def test_reliability_and_ece():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95, 0.05, 0.95, 0.05, 0.05, 0.05, 0.95])
    t = reliability_table(y, p)
    assert t["n"].sum() == len(y)
    assert expected_calibration_error(y, p) == pytest.approx(0.05)
    assert expected_calibration_error(y, y.astype(float)) == 0.0


def test_deciles_and_lift():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    d = decile_table(y, p)
    assert d["n"].sum() == 1000 and len(d) == 10
    assert d.loc[0, "lift"] > 1.5 and d["cum_capture"].iloc[-1] == pytest.approx(1.0)
