import json

import pandas as pd

from churnkit.pipeline import run


def test_end_to_end_outputs(tmp_path):
    res = run(n_customers=900, seed=5, out_dir=tmp_path)
    for f in ("metrics.json", "run_summary.json", "reliability.csv", "decile_lift.csv", "feature_importance.csv",
              "km_by_plan.csv", "at_risk_ranked.csv"):
        assert (tmp_path / f).is_file(), f
    ranked = pd.read_csv(tmp_path / "at_risk_ranked.csv")
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["churn_probability_90d"].is_monotonic_decreasing
    assert ranked["customer_id"].is_unique
    assert set(ranked["risk_decile"]) == set(range(1, 11))
    assert ranked["driver_1"].fillna("").str.len().gt(0).mean() > 0.9
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["final_model"] in ("logistic_calibrated", "gbm_calibrated")
    s = res["summary"]
    assert s["rows"]["scored_active_customers"] == len(ranked)
    assert s["censoring_demo"]["censored_no_label"] > 0


def test_run_is_reproducible(tmp_path):
    a = run(n_customers=400, seed=9, out_dir=tmp_path / "a")
    b = run(n_customers=400, seed=9, out_dir=tmp_path / "b")
    assert a["metrics"]["models"] == b["metrics"]["models"]
    assert (tmp_path / "a" / "at_risk_ranked.csv").read_text() == (tmp_path / "b" / "at_risk_ranked.csv").read_text()
