"""Baseline and gradient-boosting churn classifiers plus isotonic calibration."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEED = 20261020


def _preprocess(scale: bool) -> ColumnTransformer:
    num_steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        num_steps.append(StandardScaler())
    return ColumnTransformer([
        ("num", make_pipeline(*num_steps), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def make_logistic() -> Pipeline:
    """Baseline: regularised logistic regression on scaled, imputed features."""
    return Pipeline([("prep", _preprocess(scale=True)),
                     ("clf", LogisticRegression(C=0.5, max_iter=2000))])


def make_gbm() -> Pipeline:
    """Shallow, heavily regularised trees; settings chosen on the calibration windows, never on test."""
    return Pipeline([("prep", _preprocess(scale=False)),
                     ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, max_leaf_nodes=4,
                                                            min_samples_leaf=200, l2_regularization=1.0,
                                                            early_stopping=False, random_state=SEED))])


class PriorModel:
    """Predicts the training churn rate for everyone: the floor any model has to beat."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class Calibrated:
    """Wrap a fitted model and map its scores to probabilities, fitted on a later time window.

    ``method="sigmoid"`` (Platt scaling on the logit of the score) keeps the ranking intact and is
    stable on a few thousand rows; ``method="isotonic"`` is more flexible but produces tied steps.
    """

    def __init__(self, model, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.model = model
        self.method = method

    def _score(self, X):
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        s = self._score(X)
        if self.method == "isotonic":
            self.map_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
        else:
            self.map_ = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
        return self

    def predict_proba(self, X):
        s = self._score(X)
        p = self.map_.predict(s) if self.method == "isotonic" else self.map_.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])

    def raw_score(self, X):
        """Uncalibrated score, used for ranking (calibration is monotone, so the order is the same)."""
        return self._score(X)
