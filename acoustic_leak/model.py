"""Logistic regression and evaluation with explicit recording groups."""
from __future__ import annotations

from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import AcousticFeatures


def make_pipeline(C: float = 1.0) -> Pipeline:
    """Build an unfitted waveform-to-classifier pipeline; 1 denotes a leak."""
    if isinstance(C, (bool, np.bool_)) or not isinstance(C, Real) or not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite")
    return Pipeline([
        ("features", AcousticFeatures()),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C, class_weight="balanced", solver="newton-cg", tol=1e-8,
            max_iter=1000,
        )),
    ])


def _labels(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y)
    if np.iscomplexobj(y) or y.shape != (n,) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Labels must be an aligned vector containing both 0 and 1")
    return y.astype(np.int64)


def evaluate(labels, scores, threshold: float = 0.5) -> dict:
    """Evaluate scores against labels using score >= threshold for class 1.

    Both classes must be present, so class-specific rates and AUC are defined.
    The confusion matrix has true labels in rows, predictions in columns.
    """
    raw = np.asarray(scores)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("Scores must be a nonempty real vector")
    scores = raw.astype(np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and lie in [0, 1]")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and lie in [0, 1]")
    labels = _labels(labels, len(scores))
    matrix = confusion_matrix(labels, scores >= threshold, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return {
        "accuracy": float((tn + tp) / len(labels)),
        "balanced_accuracy": float((recall + 1 - fpr) / 2),
        "leak_recall": float(recall),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fn / (fn + tp)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": matrix.tolist(),
    }


def cross_validate(signals, labels, groups, *, C: float = 1.0,
                   n_splits: int = 3, random_state: int = 0) -> dict:
    """Return out-of-fold scores for one fixed C, never a fitted final model.

    groups must identify original recordings or acquisition sources. All clips
    from a group stay in one fold. The scaler is fitted inside each fold.
    If these scores are used to choose C, they are model-selection evidence,
    not an independent test result.
    """
    signals = np.asarray(signals)
    if signals.ndim != 2 or len(signals) == 0:
        raise ValueError("Signals must be a nonempty two-dimensional array")
    labels = _labels(labels, len(signals))
    groups = np.asarray(groups)
    if groups.shape != (len(signals),) or any(
        not isinstance(group, (str, np.str_)) or not group.strip() for group in groups
    ):
        raise ValueError("groups must contain one nonempty string per recording")
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("There must be at least n_splits distinct groups")
    # Check the full input before starting any fit.
    AcousticFeatures().fit(signals)
    make_pipeline(C)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(signals, labels, groups))
    for train, valid in splits:
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[valid])) != 2:
            raise ValueError("Every training and validation fold must contain both classes; revise the grouping or fold count")
        if np.intersect1d(groups[train], groups[valid]).size:
            raise ValueError("Training and validation groups overlap")
    scores = np.full(len(labels), np.nan)
    fold_ids = np.full(len(labels), -1, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("error", message="Line Search failed")
        for fold, (train, valid) in enumerate(splits):
            model = make_pipeline(C).fit(signals[train], labels[train])
            scores[valid] = model.predict_proba(signals[valid])[:, 1]
            fold_ids[valid] = fold
    return {"scores": scores, "folds": fold_ids, "metrics": evaluate(labels, scores)}
