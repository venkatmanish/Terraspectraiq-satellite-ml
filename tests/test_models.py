import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from terraspectra import schema
from terraspectra.data import load_partition, split_xy
from terraspectra.inference import score
from terraspectra.metrics import METRIC_ORDER
from terraspectra.registry import ArtifactError, artifact_path, load_model


def test_all_five_artifacts_exist_and_load(models):
    assert set(models) == set(schema.MODEL_KEYS)
    for key, pipe in models.items():
        assert artifact_path(key).exists()
        check_is_fitted(pipe)
        assert list(pipe.classes_) == list(schema.CLASS_LABELS)


@pytest.mark.parametrize("key", schema.MODEL_KEYS)
def test_saved_model_reproduces_manifest_test_metrics(key, models, manifest):
    X, y = split_xy(load_partition("test"))
    result = score(key, models[key], X, y).evaluation
    for name in METRIC_ORDER:
        assert result.metrics[name] == pytest.approx(manifest["models"][key]["test_metrics"][name], abs=1e-12)


@pytest.mark.parametrize("key", schema.MODEL_KEYS)
def test_predictions_are_valid_and_probabilities_sum_to_one(key, models, test_csv):
    scored = score(key, models[key], test_csv[list(schema.FEATURES)], None)
    assert set(np.unique(scored.y_pred)) <= set(schema.CLASS_LABELS)
    assert scored.proba.shape == (len(test_csv), 6)
    np.testing.assert_allclose(scored.proba.sum(axis=1), 1.0, atol=1e-9)


def test_reported_comparison_matches_manifest(manifest):
    import pandas as pd
    table = pd.read_csv(schema.REPORT_DIR / "model_comparison.csv", index_col="key")
    for key in schema.MODEL_KEYS:
        for name in METRIC_ORDER:
            assert table.loc[key, name] == pytest.approx(manifest["models"][key]["test_metrics"][name], abs=1e-6)


def test_fresh_process_gives_identical_predictions(models, test_csv):
    """A brand-new interpreter (as on Streamlit Cloud) must reproduce in-process predictions."""
    code = (
        "import json, sys, pandas as pd; sys.path.insert(0, '.');"
        "from terraspectra import schema; from terraspectra.registry import load_model;"
        "X = pd.read_csv(schema.TEST_CSV)[list(schema.FEATURES)];"
        "print(json.dumps({k: load_model(k).predict(X).tolist() for k in schema.MODEL_KEYS}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=schema.ROOT, capture_output=True, text=True, check=True)
    fresh = json.loads(out.stdout)
    X = test_csv[list(schema.FEATURES)]
    for key, pipe in models.items():
        assert fresh[key] == pipe.predict(X).tolist()


def test_tampered_artifact_is_refused(manifest, tmp_path):
    bad = json.loads(json.dumps(manifest))
    bad["models"]["knn"]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(ArtifactError, match="checksum"):
        load_model("knn", manifest=bad)


def test_selection_used_cv_not_test(manifest):
    cv = {k: v["cv"]["f1_macro_mean"] for k, v in manifest["models"].items()}
    assert manifest["selection"]["cv_winner"] == max(cv, key=cv.get)


def test_uploaded_data_never_fits_a_model(models, test_csv, monkeypatch):
    """Scoring an upload must not call fit / partial_fit / fit_transform anywhere."""
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("a fitting method was called during inference")

    targets = {Pipeline, BaseEstimator}
    for pipe in models.values():
        for step in pipe.named_steps.values():
            targets.add(type(step))
            inner = getattr(step, "estimator", None)
            if inner is not None:
                targets.add(type(inner))
    for cls in targets:
        for name in ("fit", "partial_fit", "fit_transform", "fit_predict"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, forbidden, raising=False)

    X, y = test_csv[list(schema.FEATURES)], test_csv[schema.TARGET].to_numpy()
    for key, pipe in models.items():
        assert score(key, pipe, X, y).evaluation.metrics["Accuracy"] > 0.7
