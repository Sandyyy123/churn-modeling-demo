"""Discrimination, calibration and targeting metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: mean predicted vs observed churn rate, with counts."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((b, edges[b], edges[b + 1], int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "mean_pred", "obs_rate"])


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * (t["mean_pred"] - t["obs_rate"]).abs()).sum() / t["n"].sum())


def decile_table(y, p) -> pd.DataFrame:
    """Customers sorted by predicted risk, split into 10 equal groups (1 = highest risk)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df = df.sort_values("p", ascending=False, kind="mergesort").reset_index(drop=True)
    df["decile"] = np.floor(np.arange(len(df)) * 10 / len(df)).astype(int) + 1
    base = df["y"].mean()
    g = df.groupby("decile").agg(n=("y", "size"), churners=("y", "sum"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
    g["lift"] = g["obs_rate"] / base if base > 0 else np.nan
    g["cum_capture"] = g["churners"].cumsum() / df["y"].sum() if df["y"].sum() > 0 else np.nan
    return g.reset_index()


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, base))
    dec = decile_table(y, p)
    return {
        "n": int(y.size), "churn_rate": float(base),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier), "brier_base_rate": float(brier_ref),
        "brier_skill": float(1 - brier / brier_ref) if brier_ref > 0 else float("nan"),
        "log_loss": float(log_loss(y, p)), "ece": expected_calibration_error(y, p),
        "mean_pred": float(p.mean()),
        "top_decile_lift": float(dec.loc[0, "lift"]), "top_decile_capture": float(dec.loc[0, "cum_capture"]),
    }
