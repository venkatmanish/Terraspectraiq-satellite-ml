"""TerraSpectraIQ experiment runner.

    python train.py              # full run: tune, refit, evaluate, persist, report
    python train.py --quick      # smaller grids (smoke test / slow machines)

Protocol
1. Load the official UCI partitions (sat.trn = 4,435 rows, sat.tst = 2,000 rows) and verify checksums.
2. For each of the five classifiers, run a bounded grid search with stratified 5-fold CV
   *inside sat.trn only*, selecting on macro-F1. The test file is not touched.
3. Refit the selected configuration once on all of sat.trn.
4. Evaluate each refitted pipeline exactly once on sat.tst with the shared metric engine.
5. Persist every pipeline + a manifest (hyperparameters, CV scores, test metrics, checksums),
   write test_data.csv and the comparison reports.

Note: sat.doc asks users not to *report* cross-validated results because overlapping
3x3 windows make CV optimistic. CV is used here only to rank hyperparameters; all
reported metrics come from the single official train/test split.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from terraspectra import __version__, eda, schema
from terraspectra.data import load_partition, split_xy
from terraspectra.metrics import METRIC_ORDER, evaluate, top_confusions
from terraspectra.models import candidates
from terraspectra.registry import load_model, save_model, write_manifest

QUICK_GRIDS = {
    "logistic_regression": {"clf__C": [1.0]},
    "decision_tree": {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 5]},
    "knn": {"clf__n_neighbors": [5, 9], "clf__weights": ["distance"]},
    "naive_bayes": {"clf__var_smoothing": [1e-9]},
    "svm": [{"clf__kernel": ["rbf"], "clf__C": [10.0], "clf__gamma": ["scale"]}],
}
SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "mcc": make_scorer(matthews_corrcoef),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def tune(key, cand, X, y, cv, quick):
    grid = QUICK_GRIDS[key] if quick else cand.grid
    search = GridSearchCV(
        cand.pipeline, grid, scoring=SCORING, refit="f1_macro", cv=cv,
        n_jobs=-1, return_train_score=True, error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X, y)  # convergence of the *selected* model is enforced strictly at refit time
    elapsed = time.perf_counter() - t0
    i = search.best_index_
    res = search.cv_results_
    cv_summary = {
        "f1_macro_mean": float(res["mean_test_f1_macro"][i]),
        "f1_macro_std": float(res["std_test_f1_macro"][i]),
        "accuracy_mean": float(res["mean_test_accuracy"][i]),
        "mcc_mean": float(res["mean_test_mcc"][i]),
        "train_f1_macro_mean": float(res["mean_train_f1_macro"][i]),
        "n_candidates": int(len(res["params"])),
        "search_seconds": round(elapsed, 2),
    }
    table = pd.DataFrame(res["params"]).astype(str)
    for name in SCORING:
        table[f"cv_{name}_mean"] = res[f"mean_test_{name}"]
        table[f"cv_{name}_std"] = res[f"std_test_{name}"]
    table["train_f1_macro_mean"] = res["mean_train_f1_macro"]
    table["rank"] = res["rank_test_f1_macro"]
    return search.best_params_, cv_summary, table.sort_values("rank")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny grids for a fast smoke run")
    args = parser.parse_args(argv)

    log("=" * 72)
    log(f"TerraSpectraIQ v{__version__} | Python {platform.python_version()} | scikit-learn {sklearn.__version__}")
    log(f"Host: {platform.node()} | {platform.platform()} | {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    log("=" * 72)

    train, test = load_partition("train"), load_partition("test")
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    log(f"Loaded official UCI partitions: train {X_train.shape}, test {X_test.shape} (checksums verified)")

    schema.REPORT_DIR.mkdir(exist_ok=True)
    (schema.REPORT_DIR / "cv").mkdir(exist_ok=True)
    profile = eda.summarise(train, test)
    (schema.REPORT_DIR / "eda_summary.json").write_text(json.dumps(profile, indent=2) + "\n")
    log(f"Class counts (train): {profile['class_counts_train']}")
    log(f"Spatial overlap: {profile['overlap_train_within_train']:.1%} of train rows and "
        f"{profile['overlap_test_with_train']:.1%} of test rows share 6/9 pixels with a train row")

    # The assignment's test_data.csv is the complete official held-out partition.
    test.to_csv(schema.TEST_CSV, index=False)
    log(f"Wrote {schema.TEST_CSV.name} ({len(test)} rows, 36 features + '{schema.TARGET}')")

    cv = StratifiedKFold(n_splits=schema.CV_FOLDS, shuffle=True, random_state=schema.RANDOM_STATE)
    manifest_models, rows, confusion_tables = {}, [], {}
    for key, cand in candidates().items():
        name = schema.MODEL_DISPLAY[key]
        log(f"\n--- {name} ---")
        best_params, cv_summary, cv_table = tune(key, cand, X_train, y_train, cv, args.quick)
        cv_table.to_csv(schema.REPORT_DIR / "cv" / f"{key}.csv", index=False)
        log(f"CV: {cv_summary['n_candidates']} configs in {cv_summary['search_seconds']}s | best {best_params}")
        log(f"CV macro-F1 {cv_summary['f1_macro_mean']:.4f} ± {cv_summary['f1_macro_std']:.4f} "
            f"(train-fold {cv_summary['train_f1_macro_mean']:.4f})")

        final = cand.finalize(best_params)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            final.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = final.predict(X_test)
        proba = final.predict_proba(X_test)
        predict_seconds = time.perf_counter() - t0
        result = evaluate(y_test, y_pred, proba, final.classes_)
        train_acc = float((final.predict(X_train) == y_train).mean())

        artifact = save_model(key, final)
        reloaded = load_model(key, manifest={"models": {key: {"artifact": artifact}}})
        if not (np.array_equal(reloaded.predict(X_test), y_pred)
                and np.allclose(reloaded.predict_proba(X_test), proba, atol=1e-12)):
            raise RuntimeError(f"{key}: predictions changed after save/load")

        m = result.metrics
        log("Test: " + " | ".join(f"{k} {m[k]:.4f}" for k in METRIC_ORDER) + f" | fit {fit_seconds:.2f}s")
        confusion_tables[key] = result.confusion
        rows.append({"key": key, "Model": name, **m})
        manifest_models[key] = {
            "display_name": name,
            "estimator": cand.estimator_label or type(final.named_steps["clf"]).__name__,
            "preprocessing": cand.preprocessing,
            "rationale": cand.rationale,
            "selected_params": {k.replace("clf__", ""): v for k, v in best_params.items()},
            "cv": cv_summary,
            "train_accuracy": train_acc,
            "test_metrics": m,
            "per_class_test": result.per_class.reset_index().to_dict(orient="records"),
            "top_confusions": top_confusions(result.confusion, 3),
            "fit_seconds": round(fit_seconds, 3),
            "predict_seconds_2000_rows": round(predict_seconds, 3),
            "artifact": artifact,
        }

    comparison = pd.DataFrame(rows).set_index("key")
    cv_frame = pd.DataFrame({k: v["cv"] for k, v in manifest_models.items()}).T
    winner = cv_frame.sort_values(["f1_macro_mean", "mcc_mean"], ascending=False).index[0]
    test_leader = comparison.sort_values(["F1", "MCC"], ascending=False).index[0]

    comparison.round(6).to_csv(schema.REPORT_DIR / "model_comparison.csv")
    for key, cm in confusion_tables.items():
        cm.to_csv(schema.REPORT_DIR / "cv" / f"{key}_test_confusion.csv")

    # Centre-pixel-only ablation (4 of 36 features), CV inside sat.trn only, same tuned settings.
    ablation = {}
    log("\n--- Ablation: centre pixel only (CV on sat.trn) ---")
    for key, cand in candidates().items():
        params = {f"clf__{k}": v for k, v in manifest_models[key]["selected_params"].items()}
        pipe = clone(cand.pipeline).set_params(**params)  # tuned estimator on 4 centre-pixel features
        scores = cross_val_score(pipe, X_train[list(schema.CENTRE_FEATURES)], y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        ablation[key] = {
            "centre_only_cv_f1_macro": float(scores.mean()),
            "all_36_cv_f1_macro": manifest_models[key]["cv"]["f1_macro_mean"],
        }
        log(f"{schema.MODEL_DISPLAY[key]:<26} centre-only {scores.mean():.4f} vs all-36 {ablation[key]['all_36_cv_f1_macro']:.4f}")

    manifest = {
        "project": "TerraSpectraIQ",
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": bool(args.quick),
        "dataset": {
            "name": "Statlog (Landsat Satellite)", "uci_id": 146, "doi": schema.DATASET_DOI,
            "url": schema.DATASET_URL,
            "files": {k: {"name": v[0], "sha256": v[1], "rows": v[2]} for k, v in schema.RAW_FILES.items()},
        },
        "features": list(schema.FEATURES),
        "target": schema.TARGET,
        "classes": {str(k): v for k, v in schema.CLASS_NAMES.items()},
        "protocol": {
            "tuning": f"GridSearchCV, StratifiedKFold(n_splits={schema.CV_FOLDS}, shuffle=True) on sat.trn only",
            "selection_metric": "macro F1 (cross-validated); tie-break MCC",
            "final_fit": "selected configuration refitted once on all 4,435 sat.trn rows",
            "final_evaluation": "single evaluation on the 2,000 sat.tst rows",
            "auc": "one-vs-rest ROC-AUC from predict_proba, macro-averaged",
            "averaging": "macro for precision, recall and F1",
            "random_state": schema.RANDOM_STATE,
        },
        "selection": {"cv_winner": winner, "test_macro_f1_leader": test_leader},
        "ablation_centre_pixel": ablation,
        "environment": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "models": manifest_models,
    }
    write_manifest(manifest)

    log("\n" + "=" * 72)
    log("FINAL COMPARISON — official sat.tst hold-out (2,000 rows)")
    log("=" * 72)
    log(comparison.set_index("Model")[list(METRIC_ORDER)].to_string(float_format=lambda v: f"{v:.4f}"))
    log(f"\nCV-selected winner (macro-F1 on sat.trn folds): {schema.MODEL_DISPLAY[winner]}")
    log(f"Highest test macro-F1: {schema.MODEL_DISPLAY[test_leader]}")
    log(f"Artifacts: {schema.MODEL_DIR}/*.joblib + manifest.json | Reports: {schema.REPORT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
