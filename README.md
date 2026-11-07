# TerraSpectraIQ — Multispectral Land-Cover Classification Studio

Five classical classifiers trained on Landsat satellite spectral neighbourhoods, compared with six
evaluation metrics, and served through an interactive Streamlit evaluation app.

- **GitHub repository:** https://github.com/venkatmanish/Terraspectraiq-satellite-ml
- **Live Streamlit app:** https://terraspectraiq.streamlit.app/

---

## a. Problem statement

Given the multispectral readings of a 3×3 block of Landsat Multi-Spectral Scanner (MSS) pixels,
predict the **land-cover class of the centre pixel** (red soil, cotton crop, grey soil, damp grey
soil, soil with vegetation stubble or very damp grey soil).

Different surfaces reflect visible and near-infrared light differently. Vegetation absorbs red light
and reflects strongly in near-infrared, and wetter soils are darker across all bands. This makes
land cover predictable from band intensities. The task is a **six-class classification problem**.
The goal is to implement Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive
Bayes and a Support Vector Machine on the same data, compare them with Accuracy, AUC, Precision,
Recall, F1 and MCC, and deploy the trained models in a Streamlit app that scores an uploaded test CSV
without retraining.

## b. Dataset description

| Property | Value |
|---|---|
| Name | **Statlog (Landsat Satellite)**, UCI Machine Learning Repository, dataset id 146 |
| Source | https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite · DOI [10.24432/C55887](https://doi.org/10.24432/C55887) · CC BY 4.0 |
| Instances | **6,435** = 4,435 training (`sat.trn`) + 2,000 test (`sat.tst`), the official split |
| Features | **36** integers (0–255): 4 spectral bands × 9 pixels of a 3×3 neighbourhood |
| Target | `land_cover_class`, **6 classes** with the original codes 1, 2, 3, 4, 5, 7 |
| Missing values / duplicate rows | 0 / 0 |

**Feature layout.** Values are stored pixel by pixel, left-to-right and top-to-bottom, with four bands
per pixel. Columns are named `r{row}c{col}_b{band}`, so `r1c1_b1 … r1c1_b4` is the top-left pixel and
`r2c2_b1 … r2c2_b4` (UCI attributes 17–20) is the centre pixel being classified. Two bands are visible
(B1 green, B2 red) and two are near-infrared (B3, B4). The data agree with this order: cotton crop has
its lowest mean in B2 (39.9) and its highest in B3/B4 (113.9 / 118.3).

**Classes** (training / test rows):

| Code | Land cover | Train | Test |
|---|---|---|---|
| 1 | Red soil | 1,072 | 461 |
| 2 | Cotton crop | 479 | 224 |
| 3 | Grey soil | 961 | 397 |
| 4 | Damp grey soil | 415 | 211 |
| 5 | Soil with vegetation stubble | 470 | 237 |
| 7 | Very damp grey soil | 1,038 | 470 |

There is **no class 6**: the dataset authors removed the "mixture" class. Class codes are kept exactly
as published and are never renumbered. The largest class is 2.58× the smallest, which is why
precision, recall and F1 are **macro-averaged**.

**Spatial dependence.** The rows are overlapping windows cut from one 82×100-pixel scene: 97.7% of
training rows and 97.0% of test rows share 6 of their 9 pixels with a training window shifted by one
pixel. This is why `sat.doc` asks users not to report cross-validated results. It also means the
hold-out test measures performance **within this scene**, not on new regions or years.

`test_data.csv` in this repository is the complete official `sat.tst` partition (2,000 rows, 36
features and `land_cover_class`).

## c. GitHub repository link

https://github.com/venkatmanish/Terraspectraiq-satellite-ml

```
app.py                  Streamlit app (inference only)
train.py                Experiment: tune → refit → evaluate → save models and reports
requirements.txt        Pinned dependencies (scikit-learn must match the saved models)
test_data.csv           Official held-out test set (2,000 rows) for uploading to the app
model/                  Five fitted pipelines (*.joblib) + manifest.json (parameters, scores, checksums)
terraspectra/           Package: schema, data loading, models, metrics, CSV validation, inference
reports/                model_comparison.csv, eda_summary.json, per-model CV grids and confusion matrices
data/raw/               Original UCI files sat.trn, sat.tst, sat.doc (checksum-verified)
tests/                  pytest suite (data, metrics, CSV validation, models, app)
```

## d. Models used

All five models are scikit-learn `Pipeline`s trained on the same 36 features with the same protocol:

1. A small hyperparameter grid is searched with **stratified 5-fold cross-validation inside `sat.trn`
   only**, selecting on macro-F1. Any scaler is fitted inside each fold.
2. The selected configuration is **refitted once** on all 4,435 training rows.
3. It is **evaluated once** on the 2,000 official test rows. The test set played no part in any choice.

| Model | Pipeline | Selected hyperparameters (by CV) |
|---|---|---|
| Logistic Regression | StandardScaler → LogisticRegression (lbfgs, multinomial) | C = 10 |
| Decision Tree | DecisionTreeClassifier | criterion = entropy, max_depth = None, min_samples_leaf = 2, ccp_alpha = 0.003 |
| KNN | StandardScaler → KNeighborsClassifier | k = 5, Manhattan distance (p = 1), distance weighting |
| Naive Bayes | GaussianNB | var_smoothing = 1e-3 |
| Support Vector Machine | StandardScaler → SVC (RBF) with sigmoid calibration | C = 10, gamma = 0.1 |

AUC needs class probabilities. The SVM therefore uses `CalibratedClassifierCV(SVC, method="sigmoid", ensemble=False)`,
which fits one SVC on all training rows plus Platt calibrators learned from out-of-fold decision values.
`SVC(probability=True)` is deprecated in scikit-learn 1.9.

### Comparison table — official test set (2,000 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8380 | 0.9750 | 0.8100 | 0.7967 | 0.7996 | 0.8011 |
| Decision Tree | 0.8545 | 0.9593 | 0.8358 | 0.8260 | 0.8289 | 0.8214 |
| KNN | 0.9020 | 0.9787 | 0.8927 | 0.8849 | 0.8885 | 0.8795 |
| Naive Bayes | 0.7975 | 0.9550 | 0.7872 | 0.7884 | 0.7841 | 0.7543 |
| Support Vector Machine | **0.9160** | **0.9848** | **0.9088** | **0.8989** | **0.9026** | **0.8969** |

AUC = macro one-vs-rest ROC-AUC from predicted probabilities. Precision, Recall and F1 are macro
averages. MCC = multiclass Matthews correlation coefficient. Every number is produced by
`terraspectra/metrics.py`, the same code the Streamlit app uses, and is stored in `model/manifest.json`.

Cross-validated macro-F1 on `sat.trn` (mean ± std over 5 folds; optimistic because of the overlapping
windows, so used only for ranking):

| Model | Train-fold F1 | CV F1 | Test F1 |
|---|---|---|---|
| Logistic Regression | 0.8441 | 0.8198 ± 0.0147 | 0.7996 |
| Decision Tree | 0.8965 | 0.8347 ± 0.0038 | 0.8289 |
| KNN | 1.0000 | 0.8904 ± 0.0065 | 0.8885 |
| Naive Bayes | 0.7775 | 0.7762 ± 0.0158 | 0.7841 |
| Support Vector Machine | 0.9730 | 0.9017 ± 0.0040 | 0.9026 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good AUC (0.9750, third best) but low macro recall (0.7967): it ranks classes well, yet its linear boundaries cannot follow the soil-moisture gradient. Damp grey soil recall is only 0.318; 99 of its 211 test rows are predicted as very damp grey soil. Train-fold and CV F1 are close (0.844 vs 0.820), so the problem is **underfitting (bias)**, not overfitting. CV preferred weak regularisation (C = 10 from 0.01–100), which also points to a model that needs more flexibility. |
| Decision Tree | Better than Logistic Regression on F1 (0.8289) and damp grey recall (0.517). The train-fold vs CV gap (0.897 vs 0.835) remains even after cost-complexity pruning (ccp_alpha = 0.003), so it **overfits somewhat**. Its AUC (0.9593) is second lowest, because pruned leaves give coarse, piecewise-constant probabilities. Its axis-aligned rules disagree with Logistic Regression on 305 of 2,000 test rows. |
| KNN | Second best (accuracy 0.9020, F1 0.8885) and the **best damp grey soil recall of all models (0.678)**. Distance weighting makes training accuracy 1.000 by construction, but CV ≈ test (0.890 vs 0.889) shows it still generalises within the scene. Together with the SVM it gains the most from the 3×3 context: CV F1 rises from 0.804 with the centre pixel only to 0.890 with all 36 features. As a lazy learner it keeps the whole training set (the largest artifact, about 400 KB) and computes distances at prediction time. |
| Naive Bayes | Lowest on all six metrics (accuracy 0.7975, AUC 0.9550, MCC 0.7543). The independence assumption is badly violated: the same band in neighbouring pixels correlates at 0.88, so the 36 features are largely redundant copies of four band signals. As a result the model is **over-confident**: 90% of its wrong predictions have probability > 0.9. Its errors also differ from the other models': 66 red-soil rows are called vegetation stubble. Train ≈ CV ≈ test (0.78), so it is **bias-limited**, and extra neighbourhood features barely help (+0.007 CV F1). |
| Support Vector Machine | **Best on all six metrics**: accuracy 0.9160, AUC 0.9848, F1 0.9026, MCC 0.8969. Its CV spread is low (std 0.0040), and it gains 0.082 CV F1 from the neighbourhood features. The RBF kernel captures the nonlinear spectral boundaries. It has the highest damp grey soil precision (0.805, vs KNN 0.711) at similar recall (0.664). On the same test rows it is right where KNN is wrong 78 times, and wrong where KNN is right 50 times. Cost: the slowest to fit (0.81 s including calibration) and to predict (0.28 s for 2,000 rows). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF).** It was chosen by cross-validated macro-F1 on the training set (0.9017) *before* the test set was evaluated, and the hold-out confirms it: it ranks first on all six metrics. Its lead over KNN is modest (F1 0.9026 vs 0.8885), so KNN is a reasonable cheaper alternative. Every model struggles with **damp grey soil**, which lies spectrally between grey and very damp grey soil: all five models are wrong on the same 64 test rows, 33 of them damp grey soil. |

---

## Streamlit app

The app (`app.py`) loads the five saved pipelines from `model/`, checks each file's SHA-256 against
`model/manifest.json`, and **only calls `predict` / `predict_proba`**. Uploaded data is never used to
fit or tune anything, and a test (`tests/test_models.py::test_uploaded_data_never_fits_a_model`) enforces this.

- **Sidebar:** model selection dropdown, CSV upload, download of the sample `test_data.csv`.
- **Evaluate test data:** the six metrics, confusion matrix, classification report and ROC curves for
  the selected model, plus all five models compared on the same uploaded file. Until a file is
  uploaded, the bundled official test set is shown.
- **Model benchmark:** the hold-out table, CV vs test scores, the centre-pixel ablation and per-model evidence.
- **Spectral explorer:** rebuilds any row as four 3×3 band grids and shows what each model predicts and how confident it is.
- **Dataset / Methodology:** class balance, spectral signatures, overlap statistics, metric definitions and limitations.

**Upload format:** a CSV with the 36 feature columns (`r1c1_b1` … `r3c3_b4`, any order) and
`land_cover_class` (codes 1, 2, 3, 4, 5, 7). Extra columns are ignored. Missing, non-numeric or
out-of-range (0–255) values and unknown class codes are rejected with a clear message. Values are
never imputed. If a class is absent from the file, AUC is averaged over the classes present. With a
single class, AUC is shown as *undefined* rather than 0.

## Reproduce

```bash
git clone https://github.com/venkatmanish/Terraspectraiq-satellite-ml.git
cd Terraspectraiq-satellite-ml
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py            # ~30 s: tunes, refits, evaluates and rewrites model/ and reports/
pip install pytest && python -m pytest -q
streamlit run app.py
```

`train.py` is deterministic (`random_state = 146`) and prints the same comparison table as above.
`python train.py --quick` runs tiny grids for a fast smoke test.

## Limitations

- The windows overlap and come from one small scene, so these scores describe interpolation within
  that scene. They do not show that the models would transfer to other regions, sensors or years.
- Cross-validated scores are optimistic for the same reason and were used only to rank hyperparameters.
- Inputs are raw 8-bit Landsat MSS digital numbers from the 1980s, not calibrated reflectance. Other
  imagery would need the same sensor bands and scaling.
- The saved models are scikit-learn pickles: load them only from this trusted repository and with
  the pinned scikit-learn version.

## Citation

Srinivasan, A. (1993). *Statlog (Landsat Satellite)* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55887
