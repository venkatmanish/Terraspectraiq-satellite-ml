import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from terraspectra.metrics import evaluate, ovr_macro_auc, top_confusions

CLASSES = np.array([1, 2, 3, 4, 5, 7])


def one_hot(labels, classes=CLASSES, p=0.9):
    proba = np.full((len(labels), len(classes)), (1 - p) / (len(classes) - 1))
    for i, c in enumerate(labels):
        proba[i, list(classes).index(c)] = p
    return proba


def test_perfect_predictions():
    y = np.array([1, 2, 3, 4, 5, 7] * 3)
    ev = evaluate(y, y, one_hot(y), CLASSES)
    for name in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"):
        assert ev.metrics[name] == pytest.approx(1.0)
    assert ev.notes == []


def test_hand_computed_example():
    # 3 classes present; confusions computed by hand.
    y_true = np.array([1, 1, 1, 2, 2, 3])
    y_pred = np.array([1, 1, 2, 2, 2, 3])
    ev = evaluate(y_true, y_pred, one_hot(y_pred), CLASSES)
    assert ev.metrics["Accuracy"] == pytest.approx(5 / 6)
    # precision: c1 = 2/2, c2 = 2/3, c3 = 1/1 -> mean 8/9
    assert ev.metrics["Precision"] == pytest.approx((1 + 2 / 3 + 1) / 3)
    # recall: c1 = 2/3, c2 = 1, c3 = 1
    assert ev.metrics["Recall"] == pytest.approx((2 / 3 + 1 + 1) / 3)
    assert ev.confusion.loc[1, 2] == 1
    assert ev.confusion.to_numpy().sum() == 6


def test_mcc_matches_definition_from_confusion_matrix():
    rng = np.random.default_rng(1)
    y_true = rng.choice(CLASSES, 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.choice(CLASSES, 300))
    C = evaluate(y_true, y_pred, one_hot(y_pred), CLASSES).confusion.to_numpy().astype(float)
    t, p, c, s = C.sum(axis=1), C.sum(axis=0), np.trace(C), C.sum()
    expected = (c * s - t @ p) / math.sqrt((s**2 - p @ p) * (s**2 - t @ t))
    assert evaluate(y_true, y_pred, one_hot(y_pred), CLASSES).metrics["MCC"] == pytest.approx(expected)


def test_auc_equals_sklearn_ovr_macro_when_all_classes_present():
    rng = np.random.default_rng(2)
    y = rng.choice(CLASSES, 500)
    raw = rng.random((500, 6)) + np.eye(6)[np.searchsorted(CLASSES, y)] * 0.8
    proba = raw / raw.sum(axis=1, keepdims=True)
    ours, used = ovr_macro_auc(y, proba, CLASSES)
    ref = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=CLASSES)
    assert ours == pytest.approx(ref)
    assert used == tuple(CLASSES)


def test_auc_uses_probability_columns_by_class_not_position():
    y = np.array([7, 7, 1, 1])
    proba = np.zeros((4, 6))
    proba[:2, 5] = 0.9   # column 5 is class 7
    proba[2:, 0] = 0.9   # column 0 is class 1
    proba += 0.02
    proba /= proba.sum(axis=1, keepdims=True)
    auc, _ = ovr_macro_auc(y, proba, CLASSES)
    assert auc == pytest.approx(1.0)
    # The same scores with shuffled class order must give a different (wrong-aligned) answer
    auc_misaligned, _ = ovr_macro_auc(y, proba, CLASSES[::-1])
    assert auc_misaligned < 1.0


def test_auc_with_absent_classes_is_averaged_over_present_ones():
    y = np.array([1, 1, 2, 2])
    proba = one_hot(np.array([1, 2, 2, 2]))
    ev = evaluate(y, np.array([1, 2, 2, 2]), proba, CLASSES)
    assert ev.auc_classes == (1, 2)
    assert not math.isnan(ev.metrics["AUC"])
    assert any("absent" in n for n in ev.notes)


def test_single_class_file_reports_undefined_auc_not_zero():
    y = np.array([3, 3, 3])
    ev = evaluate(y, y, one_hot(y), CLASSES)
    assert math.isnan(ev.metrics["AUC"])
    assert math.isnan(ev.metrics["MCC"])
    assert ev.metrics["Accuracy"] == 1.0
    assert any("AUC is undefined" in n for n in ev.notes)


def test_never_predicted_class_gives_undefined_precision_note():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 1, 1, 1])
    ev = evaluate(y_true, y_pred, one_hot(y_pred), CLASSES)
    assert math.isnan(ev.per_class.loc[2, "Precision"])
    assert ev.per_class.loc[2, "Recall"] == 0.0
    assert any("never predicted" in n for n in ev.notes)


def test_proba_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        ovr_macro_auc(np.array([1, 2]), np.ones((2, 5)) / 5, CLASSES)


def test_top_confusions_orders_by_count():
    y_true = np.array([3, 3, 3, 4, 4, 1])
    y_pred = np.array([4, 4, 3, 3, 4, 1])
    cm = evaluate(y_true, y_pred, one_hot(y_pred), CLASSES).confusion
    assert top_confusions(cm, 2) == [(3, 4, 2), (4, 3, 1)]
