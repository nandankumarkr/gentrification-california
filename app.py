"""
Streamlit dashboard for the gentrification-california project.

Put this file in the project root (next to data/, models/, results/) and run:
    streamlit run app.py

Data flow (matches predict.py):
  - data/analysis_dataset.csv  -> the 8 model features + 'gentrification' target
  - models/gentrification_model.pkl, models/feature_names.pkl
  - data/final_dataset.csv     -> only used to look up tract_id if analysis_dataset.csv lacks it
  - optional: a Census Gazetteer tracts file in data/ (e.g. 2010_Gaz_tracts_national.txt) enables the map
"""
import re
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ----------------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
EDA_DIR = ROOT / "results" / "eda"
ML_DIR = ROOT / "results" / "ml"

ANALYSIS_FILE = DATA_DIR / "analysis_dataset.csv"
FINAL_FILE = DATA_DIR / "final_dataset.csv"
MODEL_FILE = MODEL_DIR / "gentrification_model.pkl"
FEATURES_FILE = MODEL_DIR / "feature_names.pkl"
IMPORTANCE_FILE = ML_DIR / "feature_importance.csv"
COMPARISON_FILE = ML_DIR / "model_comparison.csv"
SUMMARY_FILE = ML_DIR / "final_model_summary.json"
TARGET = "gentrification"

st.set_page_config(
    page_title="California Gentrification Early Warning",
    page_icon="🏙️",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def tract_key(x) -> str:
    """Normalise a tract id to an 11-digit GEOID string."""
    s = str(x).strip()
    if "US" in s:
        s = s.split("US")[-1]
    s = re.sub(r"\D", "", s.split(".")[0])
    return s.zfill(11)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_features():
    return list(joblib.load(FEATURES_FILE))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_gazetteer(path: str) -> pd.DataFrame:
    g = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1")
    g.columns = [c.strip() for c in g.columns]
    lat = next(c for c in g.columns if c.upper().startswith("INTPTLAT"))
    lon = next(c for c in g.columns if c.upper().startswith("INTPTLON"))
    out = pd.DataFrame(
        {
            "tract_key": g["GEOID"].str.strip().str.zfill(11),
            "lat": pd.to_numeric(g[lat], errors="coerce"),
            "lon": pd.to_numeric(g[lon], errors="coerce"),
        }
    )
    return out


def proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else -1
    return model.predict_proba(X)[:, idx]


# ----------------------------------------------------------------------------
# Load core files
# ----------------------------------------------------------------------------
st.title("🏙️ California Gentrification Early Warning System")
st.caption(
    "Predicting whether a California Census tract's inflation-adjusted housing cost "
    "rises over the prediction period (2012 to 2016), from pre-2012 tract data."
)

missing = [p for p in [ANALYSIS_FILE, MODEL_FILE, FEATURES_FILE] if not p.exists()]
if missing:
    st.error("Missing required files:\n\n" + "\n".join(f"- {p}" for p in missing))
    st.info("Run `streamlit run app.py` from the project root.")
    st.stop()

model = load_model()
features = load_features()
df = load_csv(str(ANALYSIS_FILE)).copy()

absent = [f for f in features if f not in df.columns]
if absent:
    st.error(f"analysis_dataset.csv is missing model features: {absent}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in analysis_dataset.csv")
    st.stop()

# Leakage guard: features should only use pre-2012 information
suspicious = [f for f in features if re.search(r"2012|2016|_y$|gentrif", f.lower())]
if suspicious:
    st.warning(
        f"Possible target leakage: these model features look like they use 2012+ data "
        f"or the target itself: {suspicious}"
    )

# Tract id lookup
id_col = None
if "tract_id" in df.columns:
    id_col = "tract_id"
elif FINAL_FILE.exists():
    try:
        fin = load_csv(str(FINAL_FILE))
        common = [c for c in features if c in fin.columns]
        if (
            "tract_id" in fin.columns
            and len(fin) == len(df)
            and common
            and np.allclose(
                df[common].to_numpy(dtype=float),
                fin[common].to_numpy(dtype=float),
                equal_nan=True,
            )
        ):
            df["tract_id"] = fin["tract_id"].values
            id_col = "tract_id"
    except Exception:
        pass

# Features -> numeric, model predictions
X = df[features].apply(pd.to_numeric, errors="coerce")
n_nan = int(X.isnull().sum().sum())
medians = X.median()
if n_nan:
    st.sidebar.warning(f"{n_nan} missing feature values filled with column medians.")
    X = X.fillna(medians)

y = pd.to_numeric(df[TARGET], errors="coerce")
df["pred_proba"] = proba_positive(model, X)

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
df["pred_label"] = (df["pred_proba"] >= threshold).astype(int)

# Optional assets
importance_df = load_csv(str(IMPORTANCE_FILE)) if IMPORTANCE_FILE.exists() else None
comparison_df = load_csv(str(COMPARISON_FILE)) if COMPARISON_FILE.exists() else None
summary = json.loads(SUMMARY_FILE.read_text()) if SUMMARY_FILE.exists() else None


def importance_table():
    if importance_df is None or importance_df.empty:
        return None
    name_col = next(
        (c for c in importance_df.columns if importance_df[c].dtype == object),
        importance_df.columns[0],
    )
    num_cols = list(importance_df.select_dtypes(include=np.number).columns)
    if not num_cols:
        return None
    out = importance_df[[name_col, num_cols[0]]].copy()
    out.columns = ["feature", "importance"]
    return out.sort_values("importance", ascending=False)


imp = importance_table()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Overview",
        "Data Explorer",
        "EDA Gallery",
        "Model Performance",
        "Feature Importance",
        "Predict",
        "Map",
    ]
)
tab_overview, tab_data, tab_eda, tab_perf, tab_imp, tab_predict, tab_map = tabs

# ---- Overview ---------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Census tracts", f"{len(df):,}")
    c2.metric("Model features", len(features))
    c3.metric("Gentrifying tracts (class 1)", f"{y.mean():.1%}")
    if summary:
        c4.metric("Best model", str(summary.get("best_model", "n/a")))

    if summary:
        st.subheader("Held-out test performance (best model)")
        m = st.columns(5)
        m[0].metric("Accuracy", f"{summary.get('best_accuracy', float('nan')):.3f}")
        m[1].metric("Precision", f"{summary.get('best_precision', float('nan')):.3f}")
        m[2].metric("Recall", f"{summary.get('best_recall', float('nan')):.3f}")
        m[3].metric("F1", f"{summary.get('best_f1_score', float('nan')):.3f}")
        m[4].metric("ROC-AUC", f"{summary.get('best_roc_auc', float('nan')):.3f}")
        st.caption(
            f"Train: {summary.get('train_samples', '?'):,} tracts, "
            f"test: {summary.get('test_samples', '?'):,} tracts."
        )

    st.subheader("Model inputs")
    st.write(", ".join(f"`{f}`" for f in features))

    st.info(
        "Reading the numbers honestly: ROC-AUC near 0.70 means the model ranks tracts "
        "better than chance but is far from decisive. Treat predictions as a screening "
        "signal for prioritising where to look, not as a verdict on any single tract."
    )

# ---- Data explorer ----------------------------------------------------------
with tab_data:
    st.subheader("Analysis dataset")
    st.dataframe(df.drop(columns=["pred_label"]).head(500))

    left, right = st.columns(2)
    with left:
        feat = st.selectbox("Feature distribution by class", features)
        fig = px.histogram(
            df,
            x=feat,
            color=df[TARGET].astype(str),
            barmode="overlay",
            opacity=0.65,
            nbins=50,
            labels={"color": TARGET},
        )
        st.plotly_chart(fig)
    with right:
        st.markdown("**Correlation of each feature with the target**")
        corr = X.corrwith(y).dropna().sort_values()
        fig = px.bar(corr, orientation="h", labels={"value": "correlation", "index": "feature"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    st.subheader("Summary statistics")
    st.dataframe(X.describe().T)

# ---- EDA gallery ------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory analysis")
    images = sorted(EDA_DIR.glob("*.png")) if EDA_DIR.exists() else []
    if not images:
        st.info(f"No images found in {EDA_DIR}")
    cols = st.columns(2)
    for i, img in enumerate(images):
        cols[i % 2].image(str(img), caption=img.stem.replace("_", " ").title())

# ---- Model performance ------------------------------------------------------
with tab_perf:
    st.subheader("Model comparison (held-out test set)")
    if comparison_df is not None:
        st.dataframe(comparison_df)
        name_col = comparison_df.columns[0]
        metric_cols = list(comparison_df.select_dtypes(include=np.number).columns)
        chosen = st.multiselect("Metrics to plot", metric_cols, default=metric_cols)
        if chosen:
            long = comparison_df.melt(
                id_vars=name_col, value_vars=chosen, var_name="metric", value_name="score"
            )
            fig = px.bar(long, x=name_col, y="score", color="metric", barmode="group")
            st.plotly_chart(fig)
    else:
        st.info(f"{COMPARISON_FILE} not found.")

    st.subheader("Confusion matrices")
    cms = sorted(ML_DIR.glob("*confusion_matrix*.png")) if ML_DIR.exists() else []
    cols = st.columns(3)
    for i, img in enumerate(cms):
        cols[i % 3].image(str(img), caption=img.stem.replace("_", " ").title())

    st.subheader("Threshold explorer")
    st.caption(
        "Computed on the FULL dataset, which includes training rows, so these are optimistic. "
        "Use it to see the precision/recall trade-off as the threshold moves, not to quote accuracy."
    )
    valid = y.notna()
    yt, yp, pp = y[valid].astype(int), df.loc[valid, "pred_label"], df.loc[valid, "pred_proba"]
    t = st.columns(5)
    t[0].metric("Accuracy", f"{accuracy_score(yt, yp):.3f}")
    t[1].metric("Precision", f"{precision_score(yt, yp, zero_division=0):.3f}")
    t[2].metric("Recall", f"{recall_score(yt, yp, zero_division=0):.3f}")
    t[3].metric("F1", f"{f1_score(yt, yp, zero_division=0):.3f}")
    t[4].metric("ROC-AUC", f"{roc_auc_score(yt, pp):.3f}")

    cm = pd.crosstab(
        pd.Series(yt.values, name="Actual"), pd.Series(yp.values, name="Predicted")
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues"))

# ---- Feature importance -----------------------------------------------------
with tab_imp:
    st.subheader("What drives the prediction?")
    if imp is not None:
        n = st.slider("Top N features", 1, len(imp), min(len(imp), 15))
        top = imp.head(n).iloc[::-1]
        fig = px.bar(top, x="importance", y="feature", orientation="h")
        fig.update_layout(height=max(350, 40 * n))
        st.plotly_chart(fig)
        with st.expander("Full table"):
            st.dataframe(imp)
    else:
        st.info(f"{IMPORTANCE_FILE} not found or has no numeric column.")
    st.caption(
        "Importance shows what the model uses to separate classes, not what causes "
        "gentrification."
    )

# ---- Predict ----------------------------------------------------------------
with tab_predict:
    st.subheader("Predict gentrification risk")
    mode = st.radio(
        "Mode",
        ["Existing tract (what-if)", "Manual entry", "Upload CSV (batch)"],
        horizontal=True,
    )

    def show_result(p: float):
        label = int(p >= threshold)
        pct = float((df["pred_proba"] < p).mean())
        a, b, c = st.columns(3)
        a.metric("P(gentrification = 1)", f"{p:.1%}")
        b.metric("Predicted class", "Gentrifying (1)" if label else "Not gentrifying (0)")
        c.metric("Risk percentile vs all tracts", f"{pct:.0%}")
        st.progress(min(max(p, 0.0), 1.0))

    if mode == "Existing tract (what-if)":
        labels = df[id_col].astype(str) if id_col else df.index.astype(str)
        choice = st.selectbox("Select tract", labels.tolist())
        pos = int(np.where(labels.values == choice)[0][0])
        st.caption("Edit any input to see how the prediction changes.")
        cols = st.columns(4)
        vals = {}
        for i, f in enumerate(features):
            vals[f] = cols[i % 4].number_input(
                f, value=float(X.iloc[pos][f]), key=f"whatif_{choice}_{f}"
            )
        sample = pd.DataFrame([vals], columns=features)
        show_result(float(proba_positive(model, sample)[0]))
        st.write("**Actual label:**", int(y.iloc[pos]) if pd.notna(y.iloc[pos]) else "n/a")

    elif mode == "Manual entry":
        cols = st.columns(4)
        vals = {}
        for i, f in enumerate(features):
            vals[f] = cols[i % 4].number_input(f, value=float(medians[f]), key=f"manual_{f}")
        sample = pd.DataFrame([vals], columns=features)
        show_result(float(proba_positive(model, sample)[0]))

    else:
        st.write("Upload a CSV containing these columns:", ", ".join(f"`{f}`" for f in features))
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            lacking = [f for f in features if f not in new.columns]
            if lacking:
                st.error(f"Missing required columns: {lacking}")
            else:
                Xn = new[features].apply(pd.to_numeric, errors="coerce")
                if Xn.isnull().any().any():
                    st.error("Some feature values are missing or non-numeric. Fix them and re-upload.")
                else:
                    out = new.copy()
                    out["gentrification_probability"] = proba_positive(model, Xn)
                    out["predicted_class"] = (out["gentrification_probability"] >= threshold).astype(int)
                    st.dataframe(out.head(300))
                    st.download_button(
                        "Download predictions",
                        out.to_csv(index=False).encode(),
                        "predictions.csv",
                        "text/csv",
                    )

# ---- Map --------------------------------------------------------------------
with tab_map:
    st.subheader("Geographic view")
    gaz_files = sorted(DATA_DIR.glob("*Gaz*tract*.txt")) + sorted(DATA_DIR.glob("*gaz*tract*.txt"))
    if not id_col:
        st.info(
            "No tract_id could be attached to analysis_dataset.csv, so tracts can't be placed "
            "on a map. Add a tract_id column to it."
        )
    elif not gaz_files:
        st.info(
            "To enable the map, download the Census Gazetteer 'Census Tracts' national file "
            "for 2010 (tab-delimited .txt, from census.gov), unzip it, and place it in the "
            "data/ folder. Its name should contain 'Gaz' and 'tracts'."
        )
    else:
        gaz = load_gazetteer(str(gaz_files[0]))
        mdf = df.copy()
        mdf["tract_key"] = mdf[id_col].map(tract_key)
        mdf = mdf.merge(gaz, on="tract_key", how="inner").dropna(subset=["lat", "lon"])
        st.caption(f"Matched {len(mdf):,} of {len(df):,} tracts using {gaz_files[0].name}.")
        if mdf.empty:
            st.warning("No tract IDs matched. The tract_id format may differ from Census GEOIDs.")
        else:
            color_by = st.radio("Color by", ["pred_proba", TARGET], horizontal=True)
            kwargs = dict(
                lat="lat",
                lon="lon",
                color=color_by,
                color_continuous_scale="RdYlBu_r",
                zoom=5,
                height=650,
                opacity=0.7,
                hover_data={"tract_key": True, "pred_proba": ":.2f"},
            )
            if hasattr(px, "scatter_map"):
                fig = px.scatter_map(mdf, **kwargs)
                fig.update_layout(map_style="open-street-map")
            else:
                fig = px.scatter_mapbox(mdf, **kwargs)
                fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig)
