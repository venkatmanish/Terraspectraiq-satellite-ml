"""The one evaluation engine used by training, tests, README tables and the Streamlit app.

Conventions (multiclass):
* Precision / recall / F1 are macro-averaged (every land-cover class weighs the same).
* AUC is one-vs-rest ROC-AUC computed from predicted class probabilities, macro-averaged.
  It is never computed from hard labels.
* MCC is the multiclass Matthews correlation coefficient.
* Undefined values are reported as NaN, never silently replaced by zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import schema

METRIC_ORDER = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


@dataclass
class Evaluation:
    metrics: dict[str, float]
    confusion: pd.DataFrame
    per_class: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    auc_classes: tuple[int, ...] = ()


def ovr_macro_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Macro one-vs-rest AUC over the classes that are actually present in ``y_true``.

    ``proba[:, j]`` must be the probability of ``classes[j]`` (i.e. aligned to the
    fitted estimator's ``classes_``). A class absent from ``y_true`` has no positives,
    so its one-vs-rest AUC is undefined; it is excluded and the caller is told which
    classes were used. If fewer than two classes are present, AUC is undefined (NaN).
    When all classes are present this equals
    ``roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)``.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    if proba.ndim != 2 or proba.shape != (len(y_true), len(classes)):
        raise ValueError(f"proba shape {proba.shape} does not match ({len(y_true)}, {len(classes)})")
    present = [c for c in classes if np.any(y_true == c)]
    if len(present) < 2:
        return float("nan"), tuple(int(c) for c in present)
    scores = []
    for c in present:
        j = int(np.flatnonzero(classes == c)[0])
        scores.append(roc_auc_score((y_true == c).astype(int), proba[:, j]))
    return float(np.mean(scores)), tuple(int(c) for c in present)


def evaluate(y_true, y_pred, proba, classes) -> Evaluation:
    """Compute the six assignment metrics plus confusion matrix and per-class report."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = np.asarray(classes).astype(int)
    notes: list[str] = []

    auc, auc_classes = ovr_macro_auc(y_true, proba, classes)
    if np.isnan(auc):
        notes.append("AUC is undefined: the evaluation labels contain fewer than two classes.")
    elif len(auc_classes) < len(classes):
        missing = sorted(set(classes.tolist()) - set(auc_classes))
        notes.append(f"AUC averaged over {len(auc_classes)} classes present in the file; absent: {missing}.")

    kw = dict(average="macro", zero_division=np.nan)
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(y_true, y_pred, **kw)),
        "Recall": float(recall_score(y_true, y_pred, **kw)),
        "F1": float(f1_score(y_true, y_pred, **kw)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(np.r_[y_true, y_pred])) > 1 else float("nan"),
    }
    if np.isnan(metrics["MCC"]):
        notes.append("MCC is undefined: labels and predictions contain a single class.")

    labels = list(schema.CLASS_LABELS)
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    )
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=np.nan
    )
    per_class = pd.DataFrame(
        {"Class": [schema.CLASS_NAMES[c] for c in labels], "Precision": prec, "Recall": rec, "F1": f1, "Support": support},
        index=pd.Index(labels, name="Code"),
    )
    undefined = per_class.index[(per_class["Support"] > 0) & per_class["Precision"].isna()].tolist()
    if undefined:
        notes.append(f"Precision undefined for classes never predicted: {undefined}; excluded from the macro average.")
    return Evaluation(metrics=metrics, confusion=cm, per_class=per_class, notes=notes, auc_classes=auc_classes)


def roc_curves(y_true, proba, classes) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """One-vs-rest ROC curve per class present in y_true: {class: (fpr, tpr, auc)}."""
    y_true = np.asarray(y_true).astype(int)
    out = {}
    for j, c in enumerate(np.asarray(classes).astype(int)):
        positives = y_true == c
        if positives.any() and (~positives).any():
            fpr, tpr, _ = roc_curve(positives.astype(int), proba[:, j])
            out[int(c)] = (fpr, tpr, float(roc_auc_score(positives.astype(int), proba[:, j])))
    return out


def top_confusions(cm: pd.DataFrame, k: int = 3) -> list[tuple[int, int, int]]:
    """Largest off-diagonal cells as (actual, predicted, count)."""
    cells = [
        (int(a), int(p), int(cm.loc[a, p]))
        for a in cm.index
        for p in cm.columns
        if a != p and cm.loc[a, p] > 0
    ]
    return sorted(cells, key=lambda t: -t[2])[:k]
