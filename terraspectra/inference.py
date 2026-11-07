"""Inference-only scoring used by the Streamlit app (and its tests).

Only ``predict`` and ``predict_proba`` of already-fitted pipelines are called here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema
from .metrics import Evaluation, evaluate


@dataclass
class Scored:
    model_key: str
    y_pred: np.ndarray
    proba: np.ndarray            # columns aligned to classes
    classes: np.ndarray
    evaluation: Evaluation | None  # None when the upload has no target column


def score(model_key: str, pipeline, X: pd.DataFrame, y: np.ndarray | None) -> Scored:
    X = X[list(schema.FEATURES)]
    y_pred = np.asarray(pipeline.predict(X)).astype(int)
    proba = np.asarray(pipeline.predict_proba(X), dtype=float)
    classes = np.asarray(pipeline.classes_).astype(int)
    evaluation = evaluate(y, y_pred, proba, classes) if y is not None else None
    return Scored(model_key, y_pred, proba, classes, evaluation)


def predictions_frame(scored: Scored, y: np.ndarray | None) -> pd.DataFrame:
    out = pd.DataFrame({"predicted_class": scored.y_pred})
    out["predicted_name"] = out["predicted_class"].map(schema.CLASS_NAMES)
    out["confidence"] = scored.proba.max(axis=1)
    for j, c in enumerate(scored.classes):
        out[f"p_class_{c}"] = scored.proba[:, j]
    if y is not None:
        out.insert(0, "true_class", y)
        out["correct"] = out["true_class"] == out["predicted_class"]
    return out
