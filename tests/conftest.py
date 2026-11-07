import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from terraspectra import schema  # noqa: E402
from terraspectra.registry import load_model, read_manifest  # noqa: E402


@pytest.fixture(scope="session")
def manifest():
    return read_manifest()


@pytest.fixture(scope="session")
def models(manifest):
    return {k: load_model(k, manifest=manifest) for k in schema.MODEL_KEYS}


@pytest.fixture(scope="session")
def test_csv():
    return pd.read_csv(schema.TEST_CSV)
