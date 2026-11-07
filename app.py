"""TerraSpectraIQ — Streamlit evaluation studio.

Run locally:  streamlit run app.py

The app only loads the five pre-trained pipelines from model/ and scores data with
predict / predict_proba. It never fits, refits or tunes anything on uploaded data.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from terraspectra import eda, schema
from terraspectra.csv_guard import CSVValidationError, read_csv_bytes, validate
from terraspectra.data import load_partition, neighbourhood
from terraspectra.inference import predictions_frame, score
from terraspectra.metrics import METRIC_ORDER, roc_curves, top_confusions
from terraspectra.registry import ArtifactError, load_model, read_manifest

st.set_page_config(page_title="TerraSpectraIQ", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- visual system
# Class colours: validated categorical palette (dark steps) in fixed class-code order.
CLASS_COLORS = {1: "#3987e5", 2: "#d95926", 3: "#199e70", 4: "#c98500", 5: "#d55181", 7: "#008300"}
ACCENT, MUTED, INK, INK_2, GRID = "#3FB8AF", "#4A5A78", "#E6EDF5", "#9FB0C8", "rgba(230,237,245,0.08)"
SEQ = [[0.0, "#13213A"], [0.25, "#184f95"], [0.6, "#3987e5"], [1.0, "#cde2fb"]]  # one-hue blue ramp

METRIC_HELP = {
    "Accuracy": "Share of rows classified correctly.",
    "AUC": "Macro one-vs-rest ROC-AUC from predicted probabilities.",
    "Precision": "Macro-averaged: of rows predicted as a class, share that truly are.",
    "Recall": "Macro-averaged: of rows of a class, share that were found.",
    "F1": "Macro-averaged harmonic mean of precision and recall.",
    "MCC": "Matthews correlation: balanced agreement over the full confusion matrix.",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.2rem; max-width: 1380px; }
.tsiq-side { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; color: #E6EDF5; }
.tsiq-side .a { color: #3FB8AF; }
.tsiq-hero { border: 1px solid rgba(63,184,175,0.25); border-radius: 14px; padding: 18px 22px;
  background: linear-gradient(120deg, rgba(19,33,58,0.95), rgba(11,20,38,0.6) 60%, rgba(63,184,175,0.10)); margin-bottom: 14px; }
.tsiq-title { margin: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2rem; color: #E6EDF5; letter-spacing: -0.02em; }
.tsiq-title .a { color: #3FB8AF; } .tsiq-title .iq { color: #E6EDF5; background: rgba(63,184,175,0.18); border-radius: 6px; padding: 0 6px; margin-left: 2px; }
.tsiq-hero p { margin: 4px 0 0; color: #9FB0C8; font-size: 0.95rem; }
.tsiq-chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.tsiq-chip { font-size: 0.75rem; padding: 3px 9px; border-radius: 999px; border: 1px solid rgba(159,176,200,0.3); color: #C9D5E6; }
.tsiq-tiles { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 6px 0 14px; }
@media (max-width: 900px) { .tsiq-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.tsiq-tile { background: #13213A; border: 1px solid rgba(159,176,200,0.15); border-radius: 12px; padding: 12px 14px; }
.tsiq-tile .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9FB0C8; }
.tsiq-tile .v { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; color: #E6EDF5; margin-top: 2px; }
.tsiq-tile .d { font-size: 0.72rem; color: #7F90AA; margin-top: 2px; line-height: 1.3; }
.tsiq-note { font-size: 0.85rem; color: #9FB0C8; }
.tsiq-pill-ok { color: #0ca30c; font-weight: 600; } .tsiq-pill-bad { color: #e66767; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#13213A", bordercolor=ACCENT, font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_2))
    return fig


def fmt(v: float, digits: int = 4) -> str:
    return "undefined" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def cell_text_colors(z: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Dark ink on the light end of the ramp, light ink on the dark end (keeps >=4.5:1)."""
    share = (np.asarray(z, dtype=float) - zmin) / max(zmax - zmin, 1e-9)
    return np.where(share > 0.55, "#0B1426", INK)


def annotate_heatmap(fig: go.Figure, z: np.ndarray, labels: np.ndarray, x, y, zmin: float, zmax: float, row=None, col=None) -> None:
    colors = cell_text_colors(z, zmin, zmax)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            kw = dict(row=row, col=col) if row else {}
            fig.add_annotation(x=x[j], y=y[i], text=str(labels[i, j]), showarrow=False,
                               font=dict(color=colors[i, j], size=13, family="JetBrains Mono, monospace"), **kw)


def best_highlight(frame: pd.DataFrame):
    """Bold teal for the best value in each metric column (falls back to the plain frame)."""
    try:
        return frame.style.format("{:.4f}", na_rep="undefined").highlight_max(
            axis=0, props=f"color: {ACCENT}; font-weight: 700;")
    except Exception:  # Styler needs jinja2; never let styling break the page
        return frame


def metric_tiles(metrics: dict[str, float]) -> None:
    tiles = "".join(
        f'<div class="tsiq-tile"><div class="k">{name}</div><div class="v">{fmt(metrics[name])}</div>'
        f'<div class="d">{METRIC_HELP[name]}</div></div>'
        for name in METRIC_ORDER
    )
    st.markdown(f'<div class="tsiq-tiles">{tiles}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the five trained pipelines…")
def load_registry():
    manifest = read_manifest()
    return manifest, {k: load_model(k, manifest=manifest) for k in schema.MODEL_KEYS}


@st.cache_data(show_spinner=False)
def training_frame() -> pd.DataFrame:
    return load_partition("train")


@st.cache_data(show_spinner=False)
def dataset_profile() -> dict:
    return eda.summarise(load_partition("train"), load_partition("test"))


@st.cache_data(show_spinner=False)
def bundled_test_bytes() -> bytes:
    return schema.TEST_CSV.read_bytes()


@st.cache_data(show_spinner=False, max_entries=8)
def parse_upload(raw: bytes):
    """Validate once per distinct file content; returns (X, y, warnings) or raises CSVValidationError."""
    up = validate(read_csv_bytes(raw))
    return up.X, up.y, up.warnings


@st.cache_data(show_spinner=False, max_entries=48)
def score_cached(digest: str, model_key: str, _X: pd.DataFrame, _y):
    # Cache key = (file digest, model key). The leading underscore keeps Streamlit from re-hashing
    # the frame; the digest identifies it, so results can never leak across files or models.
    _, models = load_registry()
    return score(model_key, models[model_key], _X, _y)


try:
    MANIFEST, MODELS = load_registry()
except ArtifactError as exc:
    st.error(f"Model artifacts could not be loaded: {exc}")
    st.stop()

# ---------------------------------------------------------------- sidebar: the required controls
with st.sidebar:
    st.markdown('<div class="tsiq-side">Terra<span class="a">Spectra</span> IQ</div>', unsafe_allow_html=True)
    st.caption("Evaluation controls")
    model_key = st.selectbox(
        "Classifier", schema.MODEL_KEYS, index=schema.MODEL_KEYS.index(MANIFEST["selection"]["cv_winner"]),
        format_func=lambda k: schema.MODEL_DISPLAY[k], help="Every view below updates for the chosen saved model.",
    )
    uploaded = st.file_uploader(
        "Upload a labelled test CSV", type=["csv"],
        help=f"36 feature columns (r1c1_b1 … r3c3_b4) and the target '{schema.TARGET}'. Max 10 MB.",
    )
    st.download_button(
        "Download sample test_data.csv", bundled_test_bytes(), file_name="test_data.csv", mime="text/csv",
        help="The official 2,000-row UCI hold-out set used for the benchmark.", width="stretch",
    )
    st.divider()
    st.caption(
        f"Models trained {MANIFEST['created_utc'][:10]} · scikit-learn {MANIFEST['environment']['scikit_learn']} · "
        "inference only — uploads are never used for training."
    )

# ---------------------------------------------------------------- resolve the evaluation data
if uploaded is not None:
    raw, source_label = uploaded.getvalue(), f"uploaded file **{uploaded.name}**"
else:
    raw, source_label = bundled_test_bytes(), "bundled **test_data.csv** (official UCI hold-out, 2,000 rows)"
digest = hashlib.sha256(raw).hexdigest()
try:
    X_eval, y_eval, upload_warnings = parse_upload(raw)
    upload_error = None
except CSVValidationError as exc:
    X_eval = y_eval = None
    upload_warnings, upload_error = [], str(exc)

winner = MANIFEST["selection"]["cv_winner"]
st.markdown(
    f"""
<div class="tsiq-hero">
  <div class="tsiq-title">Terra<span class="a">Spectra</span><span class="iq">IQ</span></div>
  <p>Land-cover classification from Landsat MSS spectral neighbourhoods · five saved classifiers, one shared evaluation engine.</p>
  <div class="tsiq-chips">
    <span class="tsiq-chip">UCI Statlog (Landsat Satellite) · id 146</span>
    <span class="tsiq-chip">6,435 rows · 36 features · 6 classes</span>
    <span class="tsiq-chip">Selection metric: CV macro-F1</span>
    <span class="tsiq-chip">CV-selected model: {schema.MODEL_DISPLAY[winner]}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_eval, tab_bench, tab_explore, tab_data, tab_method = st.tabs(
    ["Evaluate test data", "Model benchmark", "Spectral explorer", "Dataset", "Methodology"]
)

# ================================================================ 1. EVALUATE
with tab_eval:
    st.markdown(f"Evaluating **{schema.MODEL_DISPLAY[model_key]}** on the {source_label}.")
    if uploaded is None:
        st.info("No file uploaded — showing the bundled official test set. Use **Upload a labelled test CSV** in the sidebar to evaluate your own file.", icon="ℹ️")
    if upload_error:
        st.error(f"This file cannot be evaluated: {upload_error}")
    else:
        for w in upload_warnings:
            st.warning(w)
        scored = score_cached(digest, model_key, X_eval, y_eval)
        if scored.evaluation is None:
            st.subheader("Predictions")
            st.dataframe(predictions_frame(scored, None).head(500), width="stretch", height=360)
        else:
            ev = scored.evaluation
            st.caption(f"{len(y_eval):,} labelled rows · model loaded from model/{model_key}.joblib · no retraining")
            metric_tiles(ev.metrics)
            for note in ev.notes:
                st.caption(f"⚠️ {note}")

            left, right = st.columns([1.05, 1])
            with left:
                cm = ev.confusion
                row_tot = cm.sum(axis=1).replace(0, np.nan)
                share = cm.div(row_tot, axis=0).fillna(0)
                labels = [schema.CLASS_SHORT[c] for c in cm.index]
                hover = [[f"{cm.iloc[i, j]} rows · {share.iloc[i, j]:.1%} of actual {labels[i]}" for j in range(6)] for i in range(6)]
                fig = go.Figure(go.Heatmap(
                    z=share.to_numpy(), x=labels, y=labels, customdata=hover, colorscale=SEQ, zmin=0, zmax=1,
                    xgap=2, ygap=2, colorbar=dict(title="Row share", tickformat=".0%", outlinewidth=0),
                    hovertemplate="Actual %{y} → predicted %{x}<br>%{customdata}<extra></extra>",
                ))
                annotate_heatmap(fig, share.to_numpy(), cm.to_numpy(), labels, labels, 0, 1)
                fig.update_yaxes(autorange="reversed", title="Actual class")
                fig.update_xaxes(title="Predicted class", side="bottom")
                fig.update_layout(title=dict(text="Confusion matrix (counts, shaded by row share)", font=dict(size=14)))
                st.plotly_chart(style_fig(fig, 430, legend=False), width="stretch")
            with right:
                st.markdown("**Classification report**")
                report = ev.per_class.reset_index()
                report["Class"] = report["Code"].map(schema.CLASS_SHORT)
                report["Code"] = report["Code"].astype(str)
                macro = pd.DataFrame(
                    {"Code": ["—"], "Class": ["Macro avg"], "Precision": [ev.metrics["Precision"]], "Recall": [ev.metrics["Recall"]],
                     "F1": [ev.metrics["F1"]], "Support": [int(report["Support"].sum())]},
                )
                st.dataframe(
                    pd.concat([report, macro], ignore_index=True), width="stretch", height=288, hide_index=True,
                    column_config={
                        "Code": st.column_config.TextColumn(width="small"),
                        "Precision": st.column_config.NumberColumn(format="%.4f"),
                        "Recall": st.column_config.NumberColumn(format="%.4f"),
                        "F1": st.column_config.NumberColumn(format="%.4f"),
                        "Support": st.column_config.NumberColumn(format="%d"),
                    },
                )
                st.caption("Undefined cells (e.g. precision of a class never predicted) are left blank, never set to 0.")
                worst = top_confusions(ev.confusion, 3)
                if worst:
                    st.markdown("**Most frequent confusions**")
                    for a, p, n in worst:
                        st.markdown(f"- {schema.CLASS_NAMES[a]} → predicted **{schema.CLASS_NAMES[p]}**: {n} rows")

            with st.expander("One-vs-rest ROC curves (from predicted probabilities)"):
                curves = roc_curves(y_eval, scored.proba, scored.classes)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1), name="Chance", hoverinfo="skip"))
                for c, (fpr, tpr, auc) in curves.items():
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=CLASS_COLORS[c], width=2),
                                             name=f"{schema.CLASS_SHORT[c]} (AUC {auc:.3f})",
                                             hovertemplate="FPR %{x:.3f} · TPR %{y:.3f}<extra>" + schema.CLASS_SHORT[c] + "</extra>"))
                fig.update_xaxes(title="False positive rate", range=[0, 1])
                fig.update_yaxes(title="True positive rate", range=[0, 1.01])
                st.plotly_chart(style_fig(fig, 420), width="stretch")

            st.divider()
            st.subheader("All five saved models on this file")
            rows = []
            for k in schema.MODEL_KEYS:
                ev_k = score_cached(digest, k, X_eval, y_eval).evaluation
                rows.append({"Model": schema.MODEL_DISPLAY[k], **ev_k.metrics})
            table = pd.DataFrame(rows).set_index("Model")
            st.dataframe(best_highlight(table), width="stretch")
            st.caption("Best value per metric in teal. All five use the same saved pipelines and the same uploaded rows.")
            rank_metric = st.radio("Rank models by", METRIC_ORDER, index=METRIC_ORDER.index("F1"), horizontal=True)
            ordered = table[rank_metric].sort_values()
            fig = go.Figure(go.Bar(
                x=ordered.values, y=ordered.index, orientation="h",
                marker=dict(color=[ACCENT if n == schema.MODEL_DISPLAY[model_key] else MUTED for n in ordered.index], cornerradius=4),
                text=[fmt(v) for v in ordered.values], textposition="outside", textfont=dict(color=INK),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            lo = np.nanmin(ordered.values) if np.isfinite(ordered.values).any() else 0
            fig.update_xaxes(range=[max(0, lo - 0.08), 1.0], title=rank_metric)
            st.plotly_chart(style_fig(fig, 280, legend=False), width="stretch")
            st.caption("Teal bar = classifier selected in the sidebar.")

            preds = predictions_frame(scored, y_eval)
            st.download_button(
                f"Download {schema.MODEL_DISPLAY[model_key]} predictions (CSV)", preds.to_csv(index=False).encode(),
                file_name=f"predictions_{model_key}.csv", mime="text/csv",
            )

# ================================================================ 2. BENCHMARK
with tab_bench:
    st.subheader("Official hold-out benchmark")
    st.markdown(
        "Each model was tuned with stratified 5-fold CV **inside `sat.trn` only**, refitted once on all 4,435 training rows, "
        "and evaluated **once** on the 2,000 official `sat.tst` rows. Numbers below are read from `model/manifest.json`, "
        "written by `train.py` with the same metric engine this app uses."
    )
    mm = MANIFEST["models"]
    bench = pd.DataFrame({schema.MODEL_DISPLAY[k]: mm[k]["test_metrics"] for k in schema.MODEL_KEYS}).T[list(METRIC_ORDER)].astype(float)
    st.dataframe(best_highlight(bench), width="stretch")
    st.caption("Best value per metric in teal · precision, recall and F1 are macro-averaged · AUC is macro one-vs-rest.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cross-validation (sat.trn) vs hold-out (sat.tst) macro-F1**")
        cvt = pd.DataFrame({
            "Model": [schema.MODEL_DISPLAY[k] for k in schema.MODEL_KEYS],
            "Train folds": [mm[k]["cv"]["train_f1_macro_mean"] for k in schema.MODEL_KEYS],
            "CV mean": [mm[k]["cv"]["f1_macro_mean"] for k in schema.MODEL_KEYS],
            "CV std": [mm[k]["cv"]["f1_macro_std"] for k in schema.MODEL_KEYS],
            "Test": [mm[k]["test_metrics"]["F1"] for k in schema.MODEL_KEYS],
        }).set_index("Model")
        st.dataframe(cvt, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in cvt.columns})
        st.caption("A large train-fold vs CV gap indicates memorisation (KNN with distance weighting reaches 1.0 by construction).")
    with c2:
        st.markdown("**Does the 3×3 neighbourhood help? (CV macro-F1, sat.trn)**")
        abl = MANIFEST["ablation_centre_pixel"]
        names = [schema.MODEL_DISPLAY[k] for k in schema.MODEL_KEYS]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Centre pixel only (4 features)", y=names, x=[abl[k]["centre_only_cv_f1_macro"] for k in schema.MODEL_KEYS],
                             orientation="h", marker=dict(color=MUTED, cornerradius=4), hovertemplate="%{y}: %{x:.4f}<extra>centre only</extra>"))
        fig.add_trace(go.Bar(name="Full neighbourhood (36 features)", y=names, x=[abl[k]["all_36_cv_f1_macro"] for k in schema.MODEL_KEYS],
                             orientation="h", marker=dict(color=ACCENT, cornerradius=4), hovertemplate="%{y}: %{x:.4f}<extra>all 36</extra>"))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, legend=dict(orientation="h", y=-0.12, x=0))
        fig.update_xaxes(range=[0.7, 0.92])
        st.plotly_chart(style_fig(fig, 330), width="stretch")

    st.subheader("Per-model evidence")
    for k in schema.MODEL_KEYS:
        m = mm[k]
        per = pd.DataFrame(m["per_class_test"]).set_index("Code")
        best_c, worst_c = per["Recall"].idxmax(), per["Recall"].idxmin()
        conf = m["top_confusions"][0] if m["top_confusions"] else None
        with st.expander(f"{m['display_name']} — test macro-F1 {m['test_metrics']['F1']:.4f}", expanded=(k == winner)):
            a, b = st.columns([1.2, 1])
            with a:
                params = ", ".join(f"`{p}={v}`" for p, v in m["selected_params"].items())
                st.markdown(
                    f"- **Estimator:** {m['estimator']}  \n- **Preprocessing:** {m['preprocessing']}  \n"
                    f"- **Selected by CV:** {params}  \n- **Why this model:** {m['rationale']}"
                )
            with b:
                st.markdown(
                    f"- Best recall: **{schema.CLASS_NAMES[int(best_c)]}** ({per.loc[best_c, 'Recall']:.3f})  \n"
                    f"- Weakest recall: **{schema.CLASS_NAMES[int(worst_c)]}** ({per.loc[worst_c, 'Recall']:.3f})  \n"
                    + (f"- Top confusion: {schema.CLASS_NAMES[conf[0]]} → {schema.CLASS_NAMES[conf[1]]} ({conf[2]} rows)  \n" if conf else "")
                    + f"- Train accuracy {m['train_accuracy']:.4f} vs test {m['test_metrics']['Accuracy']:.4f}  \n"
                    f"- Fit {m['fit_seconds']:.2f}s · predict 2,000 rows {m['predict_seconds_2000_rows']:.3f}s"
                )
    st.success(
        f"**Selected model: {schema.MODEL_DISPLAY[winner]}.** It has the highest cross-validated macro-F1 on `sat.trn` "
        f"({mm[winner]['cv']['f1_macro_mean']:.4f} ± {mm[winner]['cv']['f1_macro_std']:.4f}); the choice was made before looking at the test set. "
        f"Highest observed test macro-F1: {schema.MODEL_DISPLAY[MANIFEST['selection']['test_macro_f1_leader']]}.",
        icon="🏆",
    )

# ================================================================ 3. SPECTRAL EXPLORER
with tab_explore:
    st.subheader("Spectral neighbourhood explorer")
    st.markdown(
        "Each row is a 3×3 window of Landsat MSS pixels (≈80 m each) in four bands. The grids below rebuild the 36 values "
        "in the exact UCI order; the outlined cell is the centre pixel whose land cover is being classified. "
        "These are raw 8-bit digital numbers, not calibrated reflectance, and the rows carry no geographic coordinates."
    )
    if upload_error or X_eval is None:
        st.warning("Load a valid CSV to explore it.")
    else:
        all_scored = {k: score_cached(digest, k, X_eval, y_eval) for k in schema.MODEL_KEYS}
        P = pd.DataFrame({k: s.y_pred for k, s in all_scored.items()})
        n_distinct = P.nunique(axis=1)
        f1, f2, f3 = st.columns([1.3, 1, 1])
        if y_eval is not None:
            with f1:
                true_pick = st.multiselect("True class", schema.CLASS_LABELS, default=list(schema.CLASS_LABELS), format_func=schema.class_label)
            with f2:
                outcome = st.radio(f"{schema.MODEL_DISPLAY[model_key]} outcome", ["All", "Correct", "Misclassified"], horizontal=True)
            with f3:
                agreement = st.radio("Model agreement", ["Any", "Models disagree", "All five wrong"], horizontal=True)
            mask = np.isin(y_eval, true_pick)
            if outcome == "Correct":
                mask &= P[model_key].to_numpy() == y_eval
            elif outcome == "Misclassified":
                mask &= P[model_key].to_numpy() != y_eval
            if agreement == "Models disagree":
                mask &= n_distinct.to_numpy() > 1
            elif agreement == "All five wrong":
                mask &= (P.to_numpy() != y_eval[:, None]).all(axis=1)
            n_wrong_all = int((P.to_numpy() != y_eval[:, None]).all(axis=1).sum())
            st.caption(
                f"In this file: all five models agree on {int((n_distinct == 1).sum()):,} rows, disagree on {int((n_distinct > 1).sum()):,}, "
                f"and are all wrong on {n_wrong_all:,}. Disagreement shows where models differ, not that the label is wrong."
            )
        else:
            mask = np.ones(len(X_eval), dtype=bool)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            st.info("No rows match these filters.")
        else:
            def row_label(i: int) -> str:
                truth = f"true {schema.CLASS_SHORT[int(y_eval[i])]} · " if y_eval is not None else ""
                return f"Row {i + 1} · {truth}{schema.MODEL_DISPLAY[model_key]} → {schema.CLASS_SHORT[int(P.loc[i, model_key])]}"
            pick = st.selectbox(f"Sample ({len(idx):,} matching rows)", idx.tolist(), format_func=row_label)
            grid = neighbourhood(X_eval.iloc[pick].to_numpy())
            train_x = training_frame()[list(schema.FEATURES)].to_numpy()
            lo, hi = float(train_x.min()), float(train_x.max())
            fig = make_subplots(rows=1, cols=4, subplot_titles=[f"{b} · {schema.BAND_DESCRIPTIONS[b]}" for b in schema.BANDS], horizontal_spacing=0.03)
            fig.update_annotations(font=dict(color=INK_2, size=12))  # subplot titles only — cell labels are added after
            for j in range(4):
                band = grid[:, :, j]
                fig.add_trace(go.Heatmap(
                    z=band, zmin=lo, zmax=hi, colorscale=SEQ, showscale=(j == 3), xgap=2, ygap=2,
                    colorbar=dict(title="DN", outlinewidth=0, len=0.9),
                    hovertemplate=f"{schema.BANDS[j]} · row %{{y}}, col %{{x}}: %{{z:.0f}}<extra></extra>",
                    x=[1, 2, 3], y=[1, 2, 3],
                ), row=1, col=j + 1)
                annotate_heatmap(fig, band, band.astype(int), [1, 2, 3], [1, 2, 3], lo, hi, row=1, col=j + 1)
                fig.add_shape(type="rect", x0=1.5, x1=2.5, y0=1.5, y1=2.5, line=dict(color=ACCENT, width=3), row=1, col=j + 1)
            fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False)
            fig.update_xaxes(showticklabels=False, showgrid=False)
            st.plotly_chart(style_fig(fig, 270, legend=False), width="stretch")
            st.caption(f"Shared colour scale across bands: training-set range {lo:.0f}–{hi:.0f}.")

            c1, c2 = st.columns([1, 1.1])
            with c1:
                st.markdown("**What each saved model predicts for this row**")
                rows = []
                for k, s in all_scored.items():
                    j_pred = int(np.flatnonzero(s.classes == s.y_pred[pick])[0])
                    rec = {"Model": schema.MODEL_DISPLAY[k], "Prediction": schema.class_label(s.y_pred[pick]),
                           "Confidence": float(s.proba[pick, j_pred])}
                    if y_eval is not None:
                        j_true = int(np.flatnonzero(s.classes == y_eval[pick])[0])
                        rec["P(true class)"] = float(s.proba[pick, j_true])
                        rec["Correct"] = "✓" if s.y_pred[pick] == y_eval[pick] else "✗"
                    rows.append(rec)
                st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch",
                             column_config={"Confidence": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0),
                                            "P(true class)": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0)})
                if y_eval is not None:
                    st.markdown(f"True land cover: **{schema.class_label(y_eval[pick])}**")
                st.caption("Confidence is the model's predicted probability (SVM: sigmoid-calibrated; Naive Bayes is typically over-confident).")
            with c2:
                sig = eda.centre_signatures(training_frame())
                fig = go.Figure()
                show = {int(P.loc[pick, model_key])} | ({int(y_eval[pick])} if y_eval is not None else set())
                for c in schema.CLASS_LABELS:
                    if c in show:
                        fig.add_trace(go.Scatter(x=list(schema.BANDS), y=sig.loc[c].to_numpy(), mode="lines+markers",
                                                 line=dict(color=CLASS_COLORS[c], width=2, dash="dash"), marker=dict(size=8),
                                                 name=f"Mean {schema.CLASS_SHORT[c]} (train)"))
                fig.add_trace(go.Scatter(x=list(schema.BANDS), y=grid[1, 1], mode="lines+markers", line=dict(color=INK, width=3),
                                         marker=dict(size=10), name="This centre pixel"))
                fig.update_layout(title=dict(text="Centre-pixel spectral signature vs class means", font=dict(size=14)),
                                  legend=dict(orientation="h", y=-0.2))
                fig.update_yaxes(title="Digital number")
                st.plotly_chart(style_fig(fig, 330), width="stretch")

# ================================================================ 4. DATASET
with tab_data:
    st.subheader("Dataset intelligence (training partition unless stated)")
    prof = dataset_profile()
    tiles = {
        "Train / test rows": f"{prof['rows']['train']:,} / {prof['rows']['test']:,}",
        "Features (9 px × 4 bands)": "36",
        "Imbalance (max/min)": f"{prof['imbalance_ratio_train']:.2f}×",
        "Missing values": f"{prof['missing_values']}",
        "Train rows overlapping": f"{prof['overlap_train_within_train']:.1%}",
        "Test rows overlapping train": f"{prof['overlap_test_with_train']:.1%}",
    }
    st.markdown('<div class="tsiq-tiles">' + "".join(
        f'<div class="tsiq-tile"><div class="k">{k}</div><div class="v" style="font-size:1.15rem">{v}</div></div>' for k, v in tiles.items()
    ) + "</div>", unsafe_allow_html=True)
    st.caption("“Overlapping” = shares 6 of its 9 pixels with a window shifted by one pixel. This spatial dependence is why `sat.doc` warns "
               "against reporting cross-validated scores, and why the hold-out measures within-scene performance, not generalisation to new regions.")

    d1, d2 = st.columns(2)
    train = training_frame()
    with d1:
        counts = eda.class_counts(train)
        fig = go.Figure(go.Bar(
            x=counts.values, y=[schema.CLASS_NAMES[c] for c in counts.index], orientation="h",
            marker=dict(color=[CLASS_COLORS[c] for c in counts.index], cornerradius=4),
            text=[f"{v:,} ({v / counts.sum():.0%})" for v in counts.values], textposition="outside", textfont=dict(color=INK),
            hovertemplate="%{y}: %{x} rows<extra></extra>",
        ))
        fig.update_layout(title=dict(text="Class distribution — sat.trn", font=dict(size=14)))
        fig.update_traces(cliponaxis=False)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, counts.max() * 1.4], title="Rows")
        st.plotly_chart(style_fig(fig, 330, legend=False), width="stretch")
    with d2:
        sig = eda.centre_signatures(train)
        fig = go.Figure()
        for c in schema.CLASS_LABELS:
            fig.add_trace(go.Scatter(x=list(schema.BANDS), y=sig.loc[c].to_numpy(), mode="lines+markers", name=schema.CLASS_SHORT[c],
                                     line=dict(color=CLASS_COLORS[c], width=2), marker=dict(size=8),
                                     hovertemplate=schema.CLASS_SHORT[c] + " · %{x}: %{y:.1f}<extra></extra>"))
        fig.update_layout(title=dict(text="Mean centre-pixel signature by class", font=dict(size=14)))
        fig.update_yaxes(title="Mean digital number")
        st.plotly_chart(style_fig(fig, 330), width="stretch")

    band = st.radio("Centre-pixel band distribution", schema.BANDS, index=2, horizontal=True,
                    format_func=lambda b: f"{b} · {schema.BAND_DESCRIPTIONS[b]}")
    col = f"r2c2_b{schema.BANDS.index(band) + 1}"
    fig = go.Figure()
    for c in schema.CLASS_LABELS:
        fig.add_trace(go.Box(y=train.loc[train[schema.TARGET] == c, col], name=schema.CLASS_SHORT[c], marker_color=CLASS_COLORS[c],
                             line=dict(width=1.5), boxpoints=False))
    fig.update_yaxes(title=f"{band} digital number")
    st.plotly_chart(style_fig(fig, 320, legend=False), width="stretch")
    st.caption(
        f"Same band, different pixels correlate at {prof['mean_corr_same_band_across_pixels']:.2f} on average (strong spatial redundancy); "
        f"different bands correlate at {prof['mean_corr_across_bands']:.2f}. The grey-soil family (grey, damp grey, very damp grey) overlaps heavily in every band."
    )

# ================================================================ 5. METHODOLOGY
with tab_method:
    st.subheader("How the numbers are produced")
    st.markdown(f"""
**Data.** Official UCI files `sat.trn` (4,435 rows) and `sat.tst` (2,000 rows), checksum-verified on load. Original class codes
1, 2, 3, 4, 5, 7 are kept (class 6 was removed by the dataset authors). `test_data.csv` is the complete `sat.tst` partition.

**Protocol.** For every classifier a bounded hyperparameter grid is searched with stratified 5-fold CV inside `sat.trn`, selecting on
macro-F1. The chosen configuration is refitted on all of `sat.trn` and evaluated once on `sat.tst`. The test set was not used for any
choice. Scaling lives inside each pipeline, so it is learned from training folds only.

**Metrics** (one shared implementation in `terraspectra/metrics.py`):
- *Accuracy* — correct / total.
- *AUC* — one-vs-rest ROC-AUC per class from `predict_proba` (columns aligned to `classes_`), macro-averaged. Classes absent from an
  uploaded file are excluded and listed; with fewer than two classes AUC is reported as undefined, not 0.
- *Precision, Recall, F1* — macro averages, so the small classes (damp grey soil, 415 training rows) count as much as the large ones.
- *MCC* — multiclass Matthews correlation from the full confusion matrix.

**SVM probabilities.** `SVC(probability=True)` is deprecated in scikit-learn 1.9, so the SVM is wrapped in
`CalibratedClassifierCV(method="sigmoid", ensemble=False)`: one SVC fitted on all training rows plus per-class Platt calibrators
learned from out-of-fold decision values.

**Limitations.**
- Neighbourhoods overlap heavily ({prof['overlap_test_with_train']:.0%} of test windows share six pixels with a training window), so the
  benchmark measures interpolation within one 82×100-pixel scene, not transfer to other regions, sensors or years.
- CV scores are optimistic for the same reason; they were used only to rank hyperparameters.
- Inputs are 1980s Landsat MSS digital numbers. Other imagery would need the same sensor, bands and radiometric scaling.

**Citation.** Srinivasan, A. (1993). *Statlog (Landsat Satellite)* [Dataset]. UCI Machine Learning Repository.
{schema.DATASET_DOI} (CC BY 4.0).

**Source code:** [github.com/venkatmanish/Terraspectraiq-satellite-ml](https://github.com/venkatmanish/Terraspectraiq-satellite-ml)
""")
