"""Per-customer risk drivers by reference substitution (model-agnostic).

For each feature, the customer's value is replaced with a typical value from the
training data (median, or most common category) and the drop in predicted churn
probability is recorded. The features whose substitution lowers the risk most are
that customer's drivers. These are associations learned by the model, not causal
effects: a driver says why the model ranks someone high, not what an intervention
would change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


LABELS = {
    "tenure_days": "tenure (days)", "monthly_price": "monthly price", "sessions_last_cycle": "sessions last cycle",
    "sessions_prev_avg": "avg sessions, prior cycles", "usage_trend": "usage trend (last vs prior)",
    "zero_usage_cycles_90d": "zero-use cycles, 90d", "tickets_90d": "support tickets, 90d",
    "failed_payments_90d": "failed payments, 90d", "downgrades_90d": "downgrades, 90d", "promo_active": "on promo",
    "plan": "plan", "channel": "channel",
}


def reference_values(X_train: pd.DataFrame) -> dict:
    ref = {c: float(X_train[c].median()) for c in NUMERIC_FEATURES}
    ref.update({c: X_train[c].mode().iloc[0] for c in CATEGORICAL_FEATURES})
    return ref


def local_drivers(model, X: pd.DataFrame, ref: dict, top_k: int = 3) -> pd.DataFrame:
    base = model.predict_proba(X[FEATURES])[:, 1]
    deltas = {}
    for c in FEATURES:
        Xs = X[FEATURES].copy()
        Xs[c] = ref[c]
        deltas[c] = base - model.predict_proba(Xs)[:, 1]
    D = pd.DataFrame(deltas, index=X.index)
    out = []
    for i in X.index:
        row = D.loc[i].sort_values(ascending=False)
        row = row[row > 0].head(top_k)
        out.append([f"{LABELS[c]}: {_fmt(X.at[i, c])} (+{v * 100:.1f} pts)" for c, v in row.items()] + [""] * (top_k - len(row)))
    return pd.DataFrame(out, index=X.index, columns=[f"driver_{k + 1}" for k in range(top_k)])


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
    return str(v)
