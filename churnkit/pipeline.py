"""End-to-end churn run: data -> windows -> point-in-time features -> models -> calibration -> ranked list."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss, roc_auc_score

from . import synth
from .evaluate import decile_table, metrics, reliability_table
from .explain import local_drivers, reference_values
from .features import FEATURES, build_features, build_features_naive
from .models import Calibrated, PriorModel, make_gbm, make_logistic
from .survival import kaplan_meier, survival_at, tenure_table
from .windows import WindowSpec, check_temporal_split, label_snapshot

SPEC = WindowSpec(horizon_days=90, lookback_days=120)
TRAIN_CUTOFFS = list(pd.date_range("2024-01-01", "2025-01-01", freq="MS"))
CALIB_CUTOFFS = list(pd.date_range("2025-05-01", "2025-07-01", freq="MS"))
TEST_CUTOFF = pd.Timestamp("2025-10-01")
CENSOR_DEMO_CUTOFF = pd.Timestamp("2025-11-01")
SCORE_CUTOFF = synth.DATA_END + pd.Timedelta(days=1)


def snapshot(data, cutoff, naive: bool = False) -> pd.DataFrame:
    build = build_features_naive if naive else build_features
    X = build(data, cutoff, SPEC.lookback_days) if not naive else build(data, cutoff)
    lab = label_snapshot(data, cutoff, SPEC.horizon_days, synth.DATA_END)
    return X.merge(lab.drop(columns=["cutoff"]), on="customer_id", how="inner")


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df["censored"]].reset_index(drop=True)


def permutation_auc_drop(model, X, y, repeats: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X[FEATURES])[:, 1])
    rows = []
    for c in FEATURES:
        drops = []
        for _ in range(repeats):
            Xp = X[FEATURES].copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append((c, float(np.mean(drops)), float(np.std(drops))))
    return pd.DataFrame(rows, columns=["feature", "auc_drop", "auc_drop_sd"]).sort_values("auc_drop", ascending=False)


def run(n_customers: int = 3000, seed: int = 20261020, out_dir: str | Path = "outputs") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = synth.generate(n_customers, seed)
    check_temporal_split(TRAIN_CUTOFFS, min(CALIB_CUTOFFS), SPEC.horizon_days)
    check_temporal_split(TRAIN_CUTOFFS + CALIB_CUTOFFS, TEST_CUTOFF, SPEC.horizon_days)

    train = labelled(pd.concat([snapshot(data, c) for c in TRAIN_CUTOFFS], ignore_index=True))
    calib = labelled(pd.concat([snapshot(data, c) for c in CALIB_CUTOFFS], ignore_index=True))
    test = labelled(snapshot(data, TEST_CUTOFF))
    censor_demo = snapshot(data, CENSOR_DEMO_CUTOFF)
    score = build_features(data, SCORE_CUTOFF, SPEC.lookback_days)

    Xtr, ytr = train[FEATURES], train["label"].to_numpy()
    prior = PriorModel().fit(Xtr, ytr)
    logit = make_logistic().fit(Xtr, ytr)
    gbm = make_gbm().fit(Xtr, ytr)

    # Selection rule fixed in advance: the model with the lower log loss on the calibration windows
    # is kept, then its scores are mapped to probabilities by Platt scaling fitted on those same
    # windows. The test window is used once, at the end, to report every model side by side.
    calib_logloss = {name: float(log_loss(calib["label"], m.predict_proba(calib[FEATURES])[:, 1]))
                     for name, m in (("logistic", logit), ("gbm", gbm))}
    chosen_name = min(calib_logloss, key=calib_logloss.get)
    chosen = Calibrated({"logistic": logit, "gbm": gbm}[chosen_name], "sigmoid").fit(calib[FEATURES], calib["label"])

    yte = test["label"].to_numpy()
    preds = {
        "prior_rate": prior.predict_proba(test[FEATURES])[:, 1],
        "logistic": logit.predict_proba(test[FEATURES])[:, 1],
        "gbm": gbm.predict_proba(test[FEATURES])[:, 1],
        f"{chosen_name}_calibrated": chosen.predict_proba(test[FEATURES])[:, 1],
    }
    results = {name: metrics(yte, p) for name, p in preds.items()}

    # Leakage demonstration: same model class, same windows, whole-extract features.
    ntrain = labelled(pd.concat([snapshot(data, c, naive=True) for c in TRAIN_CUTOFFS], ignore_index=True))
    ntest = labelled(snapshot(data, TEST_CUTOFF, naive=True))
    naive_model = make_gbm().fit(ntrain[FEATURES], ntrain["label"])
    leaky_auc = float(roc_auc_score(ntest["label"], naive_model.predict_proba(ntest[FEATURES])[:, 1]))

    rel = pd.concat([reliability_table(yte, preds[k]).assign(model=k)
                     for k in ("logistic", "gbm", f"{chosen_name}_calibrated")], ignore_index=True)
    rel.to_csv(out / "reliability.csv", index=False)
    decile_table(yte, preds[f"{chosen_name}_calibrated"]).to_csv(out / "decile_lift.csv", index=False)
    imp = permutation_auc_drop(chosen, test, yte)
    imp.to_csv(out / "feature_importance.csv", index=False)

    tt = tenure_table(data, synth.DATA_END)
    km_rows, km_summary = [], []
    for plan in synth.PLAN_ORDER:
        sub = tt.loc[tt["signup_plan"] == plan]
        km = kaplan_meier(sub["duration_days"], sub["event"]).assign(plan=plan)
        km_rows.append(km)
        km_summary.append({"plan": plan, "customers": int(len(sub)), "cancellations": int(sub["event"].sum()),
                           "censored": int((sub["event"] == 0).sum()),
                           **{f"retained_{d}d": round(survival_at(km, d), 4) for d in (90, 180, 365, 540)}})
    pd.concat(km_rows, ignore_index=True).to_csv(out / "km_by_plan.csv", index=False)

    ref = reference_values(Xtr)
    sp = chosen.predict_proba(score[FEATURES])[:, 1]
    ranked = score[["customer_id", "plan", "monthly_price", "tenure_days"]].copy()
    ranked["churn_probability_90d"] = np.round(sp, 4)
    ranked["expected_revenue_at_risk_90d"] = np.round(sp * ranked["monthly_price"] * 3, 2)
    ranked = pd.concat([ranked, local_drivers(chosen, score, ref)], axis=1)
    ranked["_score"] = chosen.raw_score(score[FEATURES])
    ranked = ranked.sort_values(["_score", "customer_id"], ascending=[False, True], kind="mergesort").drop(columns="_score")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["risk_decile"] = np.floor((ranked["rank"] - 1) * 10 / len(ranked)).astype(int) + 1
    ranked.to_csv(out / "at_risk_ranked.csv", index=False)

    metrics_out = {"test_cutoff": str(TEST_CUTOFF.date()), "horizon_days": SPEC.horizon_days, "models": results,
                   "calibration_window_log_loss": calib_logloss,
                   "final_model": f"{chosen_name}_calibrated", "leakage_demo": {"naive_whole_extract_gbm_auc": leaky_auc,
                                    "point_in_time_gbm_auc": results["gbm"]["roc_auc"]}}
    (out / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    summary = {
        "seed": seed, "n_customers": n_customers, "data_start": str(synth.DATA_START.date()),
        "data_end": str(synth.DATA_END.date()), "horizon_days": SPEC.horizon_days,
        "lookback_days": SPEC.lookback_days,
        "train_cutoffs": [str(c.date()) for c in TRAIN_CUTOFFS], "calibration_cutoffs": [str(c.date()) for c in CALIB_CUTOFFS],
        "test_cutoff": str(TEST_CUTOFF.date()), "score_cutoff": str(SCORE_CUTOFF.date()),
        "rows": {"train": int(len(train)), "calibration": int(len(calib)), "test": int(len(test)),
                 "scored_active_customers": int(len(score))},
        "churn_rate": {"train": float(ytr.mean()), "calibration": float(calib["label"].mean()), "test": float(yte.mean())},
        "censoring_demo": {"cutoff": str(CENSOR_DEMO_CUTOFF.date()), "eligible": int(len(censor_demo)),
                           "churned_before_extract": int((censor_demo["label"] == 1).sum()),
                           "censored_no_label": int(censor_demo["censored"].sum())},
        "tables": {k: int(len(v)) for k, v in data.items()},
        "km_summary": km_summary, "final_model": f"{chosen_name}_calibrated",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "metrics": metrics_out}
