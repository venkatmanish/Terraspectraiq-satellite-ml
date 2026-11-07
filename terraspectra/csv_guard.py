"""Validation of user-uploaded evaluation CSVs.

The upload is only ever *scored* by already-fitted pipelines. Nothing here (or in
the app) calls fit / partial_fit / fit_transform on uploaded data.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000


class CSVValidationError(ValueError):
    """The upload cannot be evaluated; the message is shown to the user verbatim."""


@dataclass
class ValidatedUpload:
    X: pd.DataFrame                 # exactly schema.FEATURES, in training order, float
    y: np.ndarray | None            # None when the target column is absent
    warnings: list[str] = field(default_factory=list)


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise CSVValidationError("The file is empty.")
    if len(raw) > MAX_BYTES:
        raise CSVValidationError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise CSVValidationError(f"Could not parse the file as CSV: {exc}") from exc
    if frame.empty:
        raise CSVValidationError("The CSV has a header but no data rows.")
    return frame


def validate(frame: pd.DataFrame, require_target: bool = False) -> ValidatedUpload:
    warnings: list[str] = []
    frame = frame.rename(columns=lambda c: str(c).strip())
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise CSVValidationError(f"Duplicate column names: {dupes}")
    if len(frame) > MAX_ROWS:
        raise CSVValidationError(f"Too many rows ({len(frame):,}); the limit is {MAX_ROWS:,}.")

    missing = [f for f in schema.FEATURES if f not in frame.columns]
    if missing:
        preview = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
        raise CSVValidationError(
            f"{len(missing)} of the 36 required feature columns are missing ({preview}). "
            "Download the sample test_data.csv to see the expected header."
        )
    extra = [c for c in frame.columns if c not in schema.FEATURES and c != schema.TARGET]
    if extra:
        warnings.append(f"Ignored {len(extra)} extra column(s): {', '.join(map(str, extra[:5]))}{' …' if len(extra) > 5 else ''}")
    if list(frame.columns[: len(schema.FEATURES)]) != list(schema.FEATURES):
        warnings.append("Feature columns were reordered to match the training schema.")

    X = frame[list(schema.FEATURES)].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(X.to_numpy(dtype=float))
    if bad.any():
        rows = np.flatnonzero(bad.any(axis=1))
        raise CSVValidationError(
            f"{len(rows)} row(s) contain missing or non-numeric feature values "
            f"(first at data row {rows[0] + 1}). Missing values are not imputed."
        )
    X = X.astype(float)
    out_of_range = (X < schema.PIXEL_MIN) | (X > schema.PIXEL_MAX)
    if out_of_range.to_numpy().any():
        raise CSVValidationError(
            f"{int(out_of_range.to_numpy().any(axis=1).sum())} row(s) have values outside the 8-bit range 0–255."
        )

    y = None
    if schema.TARGET in frame.columns:
        target = pd.to_numeric(frame[schema.TARGET], errors="coerce")
        if target.isna().any():
            raise CSVValidationError(f"Target column '{schema.TARGET}' has {int(target.isna().sum())} missing or non-numeric value(s).")
        if not np.all(np.equal(np.mod(target.to_numpy(dtype=float), 1), 0)):
            raise CSVValidationError(f"Target column '{schema.TARGET}' must contain integer class codes.")
        y = target.to_numpy().astype(int)
        illegal = sorted(set(np.unique(y)) - set(schema.CLASS_LABELS))
        if illegal:
            raise CSVValidationError(
                f"Unknown class code(s) {illegal}. Valid codes are {list(schema.CLASS_LABELS)} "
                "(class 6 does not exist in this dataset)."
            )
        if len(np.unique(y)) < len(schema.CLASS_LABELS):
            absent = sorted(set(schema.CLASS_LABELS) - set(np.unique(y)))
            warnings.append(f"Classes absent from this file: {absent}. Some per-class metrics will be undefined.")
    elif require_target:
        raise CSVValidationError(f"Target column '{schema.TARGET}' is required to compute metrics.")
    else:
        warnings.append(f"No '{schema.TARGET}' column: predictions are shown, but metrics need true labels.")
    return ValidatedUpload(X=X, y=y, warnings=warnings)
