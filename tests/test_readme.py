"""Every number in the README tables must equal the saved experiment results."""
import re

import pytest

from terraspectra import schema

README = (schema.ROOT / "README.md").read_text()
NAMES = {
    "Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree", "KNN": "knn",
    "Naive Bayes": "naive_bayes", "Support Vector Machine": "svm",
}


def table_rows(header_start: str):
    block = README.split(header_start, 1)[1].split("\n\n", 1)[0]
    rows = {}
    for line in block.splitlines()[2:]:
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def test_comparison_table_matches_manifest(manifest):
    rows = table_rows("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
    assert set(rows) == set(NAMES)
    for name, key in NAMES.items():
        expected = [f"{manifest['models'][key]['test_metrics'][m]:.4f}" for m in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")]
        assert rows[name] == expected, name


def test_cv_table_matches_manifest(manifest):
    rows = table_rows("| Model | Train-fold F1 | CV F1 | Test F1 |")
    for name, key in NAMES.items():
        cv = manifest["models"][key]["cv"]
        assert rows[name][0] == f"{cv['train_f1_macro_mean']:.4f}"
        assert rows[name][1] == f"{cv['f1_macro_mean']:.4f} ± {cv['f1_macro_std']:.4f}"
        assert rows[name][2] == f"{manifest['models'][key]['test_metrics']['F1']:.4f}"


def test_readme_names_the_cv_winner(manifest):
    winner = manifest["selection"]["cv_winner"]
    assert winner == "svm"
    assert re.search(r"Overall, Winner.*Support Vector Machine", README)


@pytest.mark.parametrize("section", ["## a. Problem statement", "## b. Dataset description", "## c. GitHub repository link", "## d. Models used", "### Observations"])
def test_required_sections_present(section):
    assert section in README
