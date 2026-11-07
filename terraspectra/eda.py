"""Quantitative dataset profile used by the training report and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .data import overlap_rate


def class_counts(frame: pd.DataFrame) -> pd.Series:
    return frame[schema.TARGET].value_counts().reindex(schema.CLASS_LABELS, fill_value=0)


def centre_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean centre-pixel value per band and class (rows: class code, cols: B1..B4)."""
    sig = frame.groupby(schema.TARGET)[list(schema.CENTRE_FEATURES)].mean()
    sig.columns = list(schema.BANDS)
    return sig.reindex(schema.CLASS_LABELS)


def band_correlation_summary(frame: pd.DataFrame) -> dict[str, float]:
    corr = np.corrcoef(frame[list(schema.FEATURES)].to_numpy(dtype=float).T)
    n = len(schema.FEATURES)
    same_band = [corr[i, j] for i in range(n) for j in range(n) if i != j and i % 4 == j % 4]
    cross_band = [corr[i, j] for i in range(n) for j in range(n) if i % 4 != j % 4]
    return {
        "mean_corr_same_band_across_pixels": float(np.mean(same_band)),
        "mean_corr_across_bands": float(np.mean(cross_band)),
    }


def summarise(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    counts = class_counts(train)
    x = train[list(schema.FEATURES)]
    return {
        "rows": {"train": len(train), "test": len(test), "total": len(train) + len(test)},
        "n_features": len(schema.FEATURES),
        "class_counts_train": {int(k): int(v) for k, v in counts.items()},
        "class_counts_test": {int(k): int(v) for k, v in class_counts(test).items()},
        "imbalance_ratio_train": float(counts.max() / counts.min()),
        "missing_values": int(train.isna().sum().sum() + test.isna().sum().sum()),
        "duplicate_rows_train": int(train.duplicated().sum()),
        "test_rows_identical_to_train": int(
            test[list(schema.FEATURES)].merge(x.drop_duplicates(), how="inner").shape[0]
        ),
        "pixel_range": [int(x.to_numpy().min()), int(x.to_numpy().max())],
        "overlap_train_within_train": overlap_rate(train, train, exclude_self=True),
        "overlap_test_with_train": overlap_rate(train, test),
        "centre_signature_means": {
            int(code): {band: float(v) for band, v in row.items()}
            for code, row in centre_signatures(train).round(2).iterrows()
        },
        **band_correlation_summary(train),
    }
