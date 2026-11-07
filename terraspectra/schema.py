"""Single source of truth for the dataset schema.

Every module (training, evaluation, CSV validation, Streamlit UI) imports the
feature order, target name and class semantics from here, so a column can never
silently drift between training and inference.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
TEST_CSV = ROOT / "test_data.csv"

# Official UCI files (dataset id 146), SHA-256 of the files as downloaded from
# https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip
RAW_FILES = {
    "train": ("sat.trn", "e896dc88a960fa2404160fc4c3cb3dc53fcf4afd80ba920bf2d261bd42d12613", 4435),
    "test": ("sat.tst", "4b9167b8a92baafafed7c8809aef86d0683a5e685c19d98d119fa6974f0f2479", 2000),
}
DATASET_DOI = "https://doi.org/10.24432/C55887"
DATASET_URL = "https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite"

# The four Landsat MSS bands. sat.doc says two are visible (green, red) and two
# near-infrared; the green/red/NIR/NIR order follows the MSS convention and is
# consistent with the data (cotton crop: lowest in B2, highest in B3/B4).
BANDS = ("B1", "B2", "B3", "B4")
BAND_DESCRIPTIONS = {
    "B1": "Visible green",
    "B2": "Visible red",
    "B3": "Near-infrared 1",
    "B4": "Near-infrared 2",
}

# Pixels are read left-to-right, top-to-bottom; each contributes four bands.
GRID_ROWS = GRID_COLS = 3
FEATURES: tuple[str, ...] = tuple(
    f"r{r}c{c}_b{b}"
    for r in range(1, GRID_ROWS + 1)
    for c in range(1, GRID_COLS + 1)
    for b in range(1, len(BANDS) + 1)
)
CENTRE_FEATURES: tuple[str, ...] = tuple(f"r2c2_b{b}" for b in range(1, 5))  # UCI attributes 17-20
TARGET = "land_cover_class"

# Class 6 ("mixture") was removed by the dataset authors. Labels are NOT 1..6.
CLASS_NAMES: dict[int, str] = {
    1: "Red soil",
    2: "Cotton crop",
    3: "Grey soil",
    4: "Damp grey soil",
    5: "Soil with vegetation stubble",
    7: "Very damp grey soil",
}
CLASS_LABELS: tuple[int, ...] = tuple(sorted(CLASS_NAMES))
CLASS_SHORT: dict[int, str] = {
    1: "Red soil", 2: "Cotton", 3: "Grey soil", 4: "Damp grey", 5: "Veg. stubble", 7: "V. damp grey",
}

PIXEL_MIN, PIXEL_MAX = 0, 255  # 8-bit MSS digital numbers

RANDOM_STATE = 146  # the UCI dataset id; used for every stochastic component
CV_FOLDS = 5

MODEL_KEYS = ("logistic_regression", "decision_tree", "knn", "naive_bayes", "svm")
MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes (Gaussian)",
    "svm": "Support Vector Machine",
}


def class_label(code: int) -> str:
    return f"{code} · {CLASS_NAMES.get(int(code), 'unknown')}"
