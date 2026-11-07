"""Headless tests of the Streamlit app (streamlit.testing.v1.AppTest)."""
import re

import pytest
from streamlit.testing.v1 import AppTest

from terraspectra import schema

APP = str(schema.ROOT / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def shown_metrics(at):
    tiles = next(m.value for m in at.markdown if "tsiq-tile" in m.value and "Accuracy" in m.value)
    return re.findall(r'<div class="v">([0-9.]+|undefined)</div>', tiles)


def test_app_starts_without_errors(app):
    assert not app.exception
    assert not app.error
    assert [t.label for t in app.tabs][:2] == ["Evaluate test data", "Model benchmark"]


def test_sidebar_has_model_dropdown_with_all_five(app):
    box = app.sidebar.selectbox[0]
    assert box.label == "Classifier"
    assert len(box.options) == 5


@pytest.mark.parametrize("key", schema.MODEL_KEYS)
def test_switching_model_shows_that_models_metrics(app, manifest, key):
    app.sidebar.selectbox[0].set_value(key).run()
    assert not app.exception
    expected = [f"{manifest['models'][key]['test_metrics'][m]:.4f}" for m in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")]
    assert shown_metrics(app) == expected


def test_explorer_filters_run(app):
    radio = next(r for r in app.radio if r.label == "Model agreement")
    radio.set_value("All five wrong").run()
    assert not app.exception
