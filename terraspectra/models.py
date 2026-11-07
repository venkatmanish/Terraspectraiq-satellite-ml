"""The five classifier pipelines and their bounded hyperparameter grids.

Each candidate is a full scikit-learn Pipeline, so scaling is fitted inside every
CV fold (no leakage) and the exact same preprocessing is persisted with the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .schema import RANDOM_STATE


@dataclass
class Candidate:
    key: str
    pipeline: Pipeline
    grid: dict | list[dict]
    preprocessing: str
    rationale: str
    estimator_label: str = ""

    def finalize(self, best_params: dict) -> Pipeline:
        """Unfitted pipeline for the selected configuration (refitted on all of sat.trn)."""
        return clone(self.pipeline).set_params(**best_params)


@dataclass
class CalibratedSVMCandidate(Candidate):
    """SVC tuned on its native decision function, then wrapped for probabilities.

    SVC(probability=True) is deprecated in scikit-learn 1.9. CalibratedClassifierCV with
    ensemble=False fits one SVC on all training rows and learns per-class sigmoid (Platt)
    calibrators from out-of-fold decision values, giving genuine probabilities for AUC.
    """

    def finalize(self, best_params: dict) -> Pipeline:
        svc_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
        svc = SVC(random_state=RANDOM_STATE, **svc_params)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=inner_cv)),
        ])


def candidates() -> dict[str, Candidate]:
    return {
        "logistic_regression": Candidate(
            key="logistic_regression",
            pipeline=Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)),
            ]),
            grid={"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            preprocessing="StandardScaler (fitted on training folds only)",
            rationale="Multinomial softmax with L2 penalty; scaling makes the penalty treat all bands equally.",
        ),
        "decision_tree": Candidate(
            key="decision_tree",
            pipeline=Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            grid={
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 6, 10, 14],
                "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__ccp_alpha": [0.0, 0.001, 0.003, 0.01],
            },
            preprocessing="None — splits are invariant to monotone rescaling",
            rationale="Axis-aligned threshold rules on raw digital numbers; regularised by depth, leaf size and pruning.",
        ),
        "knn": Candidate(
            key="knn",
            pipeline=Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            grid={
                "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
            preprocessing="StandardScaler (distances otherwise dominated by high-variance bands)",
            rationale="Instance-based vote among nearest spectral neighbourhoods.",
        ),
        "naive_bayes": Candidate(
            key="naive_bayes",
            pipeline=Pipeline([("clf", GaussianNB())]),
            grid={"clf__var_smoothing": np.logspace(-9, 0, 10).tolist()},
            preprocessing="None — Gaussian likelihoods are fitted per feature",
            rationale="Gaussian class-conditional densities with a (strongly violated) feature-independence assumption.",
        ),
        "svm": CalibratedSVMCandidate(
            key="svm",
            pipeline=Pipeline([("scale", StandardScaler()), ("clf", SVC(random_state=RANDOM_STATE))]),
            grid=[
                {"clf__kernel": ["rbf"], "clf__C": [1.0, 10.0, 100.0], "clf__gamma": [0.01, 0.03, 0.1, 0.3]},
                {"clf__kernel": ["linear"], "clf__C": [0.1, 1.0]},
            ],
            preprocessing="StandardScaler (kernel distances are scale-sensitive)",
            rationale="Maximum-margin kernel classifier; sigmoid (Platt) calibration supplies the probabilities used for AUC.",
            estimator_label="SVC + sigmoid calibration (CalibratedClassifierCV, ensemble=False)",
        ),
    }
