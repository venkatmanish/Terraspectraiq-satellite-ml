"""Loading and integrity-checking the official UCI partitions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema


class DataIntegrityError(RuntimeError):
    """Raised when a raw file does not match the published dataset."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_partition(split: str, raw_dir: Path | None = None, verify_checksum: bool = True) -> pd.DataFrame:
    """Return one official partition ("train" or "test") as a DataFrame.

    Columns are the 36 named features in UCI order followed by the target.
    The original class codes (1, 2, 3, 4, 5, 7) are preserved as-is.
    """
    if split not in schema.RAW_FILES:
        raise ValueError(f"split must be one of {list(schema.RAW_FILES)}, got {split!r}")
    fname, expected_hash, expected_rows = schema.RAW_FILES[split]
    path = (raw_dir or schema.RAW_DIR) / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. See data/README.md for download instructions.")
    if verify_checksum and sha256(path) != expected_hash:
        raise DataIntegrityError(f"{fname} checksum differs from the official UCI file.")

    values = np.loadtxt(path, dtype=np.int64)
    n_cols = len(schema.FEATURES) + 1
    if values.ndim != 2 or values.shape != (expected_rows, n_cols):
        raise DataIntegrityError(f"{fname}: expected shape ({expected_rows}, {n_cols}), got {values.shape}")

    frame = pd.DataFrame(values[:, :-1], columns=list(schema.FEATURES))
    frame[schema.TARGET] = values[:, -1]

    bad_labels = set(np.unique(frame[schema.TARGET])) - set(schema.CLASS_LABELS)
    if bad_labels:
        raise DataIntegrityError(f"{fname}: unexpected class codes {sorted(bad_labels)}")
    x = frame[list(schema.FEATURES)].to_numpy()
    if x.min() < schema.PIXEL_MIN or x.max() > schema.PIXEL_MAX:
        raise DataIntegrityError(f"{fname}: pixel values outside 0-255")
    return frame


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return frame[list(schema.FEATURES)], frame[schema.TARGET].to_numpy()


def neighbourhood(row: pd.Series | np.ndarray) -> np.ndarray:
    """Reshape 36 values into a (3 rows, 3 cols, 4 bands) array, pixel-major as in sat.doc."""
    values = np.asarray(row[list(schema.FEATURES)] if isinstance(row, pd.Series) else row, dtype=float)
    return values.reshape(schema.GRID_ROWS, schema.GRID_COLS, len(schema.BANDS))


def overlap_rate(reference: pd.DataFrame, query: pd.DataFrame, exclude_self: bool = False) -> float:
    """Share of `query` rows whose 3x3 window shares 6 of 9 pixels with some `reference` row.

    Two windows one pixel apart (horizontally or vertically) share a 3x2 / 2x3 block.
    This quantifies the spatial dependence that sat.doc warns about. With
    ``exclude_self`` (reference is query), a row is not allowed to match itself.
    """
    def grids(df: pd.DataFrame) -> np.ndarray:
        return df[list(schema.FEATURES)].to_numpy(dtype=np.int16).reshape(-1, 3, 3, 4)

    def index(blocks) -> dict[bytes, set[int]]:
        out: dict[bytes, set[int]] = {}
        for i, b in enumerate(blocks):
            out.setdefault(b.tobytes(), set()).add(i)
        return out

    ref, qry = grids(reference), grids(query)
    lookups = [  # (block of the reference row, matching block of the query row)
        (index(g[:, :2] for g in ref), lambda g: g[:, 1:]),  # reference is right of query
        (index(g[:, 1:] for g in ref), lambda g: g[:, :2]),  # reference is left of query
        (index(g[:2] for g in ref), lambda g: g[1:]),        # reference is below query
        (index(g[1:] for g in ref), lambda g: g[:2]),        # reference is above query
    ]
    hits = []
    for i, g in enumerate(qry):
        hit = False
        for table, block in lookups:
            matches = table.get(block(g).tobytes(), set())
            if matches - ({i} if exclude_self else set()):
                hit = True
                break
        hits.append(hit)
    return float(np.mean(hits))
