"""Persisting and loading the fitted pipelines.

joblib files are pickles, so they must only ever be loaded from a trusted source.
Every artifact's SHA-256 is recorded in model/manifest.json at training time and
checked before unpickling; the app refuses to load a file that does not match.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from . import schema
from .data import sha256

MANIFEST = "manifest.json"


class ArtifactError(RuntimeError):
    pass


def artifact_path(key: str, model_dir: Path | None = None) -> Path:
    return (model_dir or schema.MODEL_DIR) / f"{key}.joblib"


def save_model(key: str, pipeline, model_dir: Path | None = None) -> dict:
    path = artifact_path(key, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def write_manifest(manifest: dict, model_dir: Path | None = None) -> Path:
    path = (model_dir or schema.MODEL_DIR) / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return path


def read_manifest(model_dir: Path | None = None) -> dict:
    path = (model_dir or schema.MODEL_DIR) / MANIFEST
    if not path.exists():
        raise ArtifactError(f"{path} not found — run `python train.py` first.")
    return json.loads(path.read_text())


def load_model(key: str, model_dir: Path | None = None, manifest: dict | None = None):
    """Load one fitted pipeline after verifying its checksum against the manifest."""
    manifest = manifest or read_manifest(model_dir)
    entry = manifest["models"].get(key)
    if entry is None:
        raise ArtifactError(f"Model {key!r} is not listed in the manifest.")
    path = artifact_path(key, model_dir)
    if not path.exists():
        raise ArtifactError(f"{path} is missing.")
    if sha256(path) != entry["artifact"]["sha256"]:
        raise ArtifactError(f"{path.name} does not match the checksum in the manifest; refusing to load it.")
    pipeline = joblib.load(path)
    if list(pipeline.classes_) != list(schema.CLASS_LABELS):
        raise ArtifactError(f"{key}: class order {list(pipeline.classes_)} differs from the schema.")
    if list(getattr(pipeline, "feature_names_in_", schema.FEATURES)) != list(schema.FEATURES):
        raise ArtifactError(f"{key}: feature order differs from the schema.")
    return pipeline


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")
