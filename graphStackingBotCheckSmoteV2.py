# filename: graphStackingBotCheckSmoteV4.py
import os
import gc
import joblib
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import traceback
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
# imblearn combine
from imblearn.combine import SMOTEENN
import plotly.subplots as sp

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from libInternal import (
    variableDump,
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
)

# === THREAD / JOB LIMITS TO AVOID CPU/BLAS OVERLOAD ===
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# === CONFIG ===
SELECTED_SENSOR_ID = "1"
SAMPLE_EDGES = None
DOWNSAMPLE_RATIO = 5 # keep this many majority rows per minority row when downsampling
MAX_MAJOR = 800_000 # cap for majority rows after downsampling

# === PATHS ===
fileTimeStamp, output_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir
os.makedirs(graph_dir, exist_ok=True)

try:
    con = getConnection()
    print("[Init] Using connection from getConnection()")
except Exception as e:
    print(f"[Init] Warning: getConnection() failed ({e}), using direct DuckDB connection.")
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL AND SensorId = '{SELECTED_SENSOR_ID}'
"""
print(f"[Load] Running query for SensorId = {SELECTED_SENSOR_ID} ...")
df = con.sql(query).df()
print(f"[Load] rows: {len(df)}")

if df.empty:
    raise RuntimeError(f"No data found for SensorId = {SELECTED_SENSOR_ID}")

# optional sampling very early to reduce overhead (useful for local dev)
if SAMPLE_EDGES is not None and SAMPLE_EDGES > 0 and len(df) > SAMPLE_EDGES:
    print(f"[Sampling] sampling {SAMPLE_EDGES} rows from {len(df)}")
    df = df.sample(SAMPLE_EDGES, random_state=42).reset_index(drop=True)

df = optimize_dataframe(df)
print(f"[Load] memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# label conversion
df = fast_label_to_binary(df)

print("\n[DEBUG] Label distribution after conversion for Sensor:", SELECTED_SENSOR_ID)
print(df["Label"].value_counts())
print("\n[DEBUG] Group by Label for Sensor:", SELECTED_SENSOR_ID)
print(df[["SensorId", "Label"]].groupby("Label").size())

# if extremely imbalanced: downsample majority to keep training stable
counts = df["Label"].value_counts()
if len(counts) > 1:
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    minority_count = counts.min()
    majority_count = counts.max()
    desired_major = min(max(int(minority_count * DOWNSAMPLE_RATIO), minority_count), MAX_MAJOR)

    if majority_count > desired_major:
        print(f"[Downsample] minority={minority_count:,}, majority={majority_count:,} -> sampling majority to {desired_major:,}")
        df_min = df[df["Label"] == minority_label]
        df_maj = df[df["Label"] == majority_label].sample(n=desired_major, random_state=42)
        df_other = df[~df["Label"].isin([minority_label, majority_label])]
        df = pd.concat([df_min, df_maj, df_other], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
        gc.collect()
        print(f"[Downsample] New dataset size: {len(df):,}")
    else:
        print(f"[Downsample] No downsampling needed (majority {majority_count:,} <= desired {desired_major:,})")

# safety check
if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

# drop rows missing essential features
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

# FEATURE ENGINEERING: add a few informative features
# (safe operations; avoid divide-by-zero)
df["BytesPerPkt"] = df["TotBytes"] / (df["TotPkts"] + 1e-9)
df["DurPerPkt"] = df["Dur"] / (df["TotPkts"] + 1e-9)
df["PktRatio"] = df["SrcBytes"] / (df["TotBytes"] + 1e-9)
df["LogTotBytes"] = np.log1p(df["TotBytes"])
df["LogTotPkts"] = np.log1p(df["TotPkts"])
df["SrcBytesRatio"] = df["SrcBytes"] / (df["TotBytes"] + 1e-9)

# categorical encoding
cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# === SELECT FEATURES ===
features_base = ["Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"]
engineered = ["BytesPerPkt", "DurPerPkt", "PktRatio", "LogTotBytes", "LogTotPkts", "SrcBytesRatio"]
features = [f for f in (features_base + engineered) if f in df.columns]
if not features:
    raise RuntimeError("No valid numeric features found in dataset!")

print(f"[Features] Using features ({len(features)}): {features}")

X = df[features].fillna(df[features].mean())
y = df["Label"].astype(int)

# stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test distribution:", np.bincount(y_test))

# scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# BALANCE: SMOTEENN (SMOTE + ENN) to create clean resampled set
print("\n[SMOTEENN] Balancing minority class with SMOTEENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_scaled, y_train)
print("[SMOTEENN] Before:", np.bincount(y_train), " After:", np.bincount(y_train_res))

# stack setup
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=80, max_depth=10, class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(max_depth=8, random_state=42))
]

if _HAS_XGB:
    meta = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )
    print("[Model] Using XGBoost as meta-estimator")
else:
    meta = ExtraTreesClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=1)
    print("[Model] XGBoost not found; using ExtraTrees as fallback meta-estimator")

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=3,
    n_jobs=1,
    passthrough=False,
    verbose=1
)

gc.collect()

# train
print(f"\n[Train] Training stacking model (SMOTEENN) for Sensor {SELECTED_SENSOR_ID} ...")
try:
    stack.fit(X_train_res, y_train_res)
    print("[Train] Done.")
except Exception:
    print("[Train] Exception during stack.fit():")
    traceback.print_exc()
    raise

gc.collect()

# === EVALUATION: probability on test set & threshold tuning (maximize F1) ===
p_test = stack.predict_proba(X_test_scaled)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Evaluation for Sensor (SMOTEENN):", SELECTED_SENSOR_ID)
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

# evaluation report generator (SMOTEENN) ===
def generate_plotly_evaluation_report_smoteenn(y_true, y_pred, y_prob, sensor_id, best_threshold, output_dir, file_timestamp):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    metrics = {
        "Accuracy": round((y_pred == y_true).mean() * 100, 2),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_val
    }

    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Precision-Recall Curve", "ROC Curve"))

    # PR curve
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve", line=dict(width=2)), row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=1)

    # ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC={roc_auc_val:.3f})", line=dict(width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

    text_metrics = "<br>".join([f"<b>{k}:</b> {v:.4f}" if k != "Accuracy" else f"<b>{k}:</b> {v:.2f}%" for k, v in metrics.items()])
    fig.add_annotation(
        text=f"<b>Sensor {sensor_id} Evaluation Summary (SMOTEENN)</b><br>{text_metrics}<br><br><b>Best Threshold:</b> {best_threshold:.3f}",
        xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False, align="left"
    )

    fig.update_layout(
        title=f"Evaluation Dashboard (SMOTEENN) - Sensor {sensor_id}",
        title_x=0.5,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        width=1100
    )

    html_path = os.path.join(output_dir, f"EvaluationReport_Sensor{sensor_id}_SMOTEENN_{file_timestamp}.html")
    fig.write_html(html_path)
    print(f"[Report] Saved -> {html_path}")
    return html_path

# save report
report_path = generate_plotly_evaluation_report_smoteenn(
    y_true=y_test, y_pred=y_pred_test, y_prob=p_test,
    sensor_id=SELECTED_SENSOR_ID, best_threshold=best_threshold,
    output_dir=graph_dir, file_timestamp=fileTimeStamp
)

train_csv = os.path.join(output_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_SMOTEENN_{fileTimeStamp}.csv")
test_csv = os.path.join(output_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_SMOTEENN_{fileTimeStamp}.csv")
pd.concat([X_train, pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

# apply model (batched) ===
print("[Apply] Scoring entire dataset in batches to avoid memory pressure...")
batch_size = 200_000
df["PredictedProb"] = np.nan
for i in range(0, len(df), batch_size):
    end = min(i + batch_size, len(df))
    Xb = df.loc[i:end-1, features].fillna(df[features].mean())
    Xb_scaled = scaler.transform(Xb)
    df.loc[i:end-1, "PredictedProb"] = stack.predict_proba(Xb_scaled)[:, 1]
    # small explicit GC
    gc.collect()

df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
df["PredictedRole"] = df["PredictedLabel"].apply(lambda x: "Botnet" if x == 1 else "Normal")

# graph render
print(f"[Graph] Generating visualization for Sensor {SELECTED_SENSOR_ID} with {len(df)} edges (sampling up to 5000 for layout)...")
SAMPLE_FOR_GRAPH = 5000
if len(df) > SAMPLE_FOR_GRAPH:
    samp = df.sample(SAMPLE_FOR_GRAPH, random_state=42)
else:
    samp = df

G = nx.from_pandas_edgelist(samp, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

node_roles = {}
for addr in set(samp["SrcAddr"]).union(samp["DstAddr"]):
    subset = df[(df["SrcAddr"] == addr) | (df["DstAddr"] == addr)]
    avg_prob = subset["PredictedProb"].mean()
    if avg_prob > 0.85:
        node_roles[addr] = "C&C"
    elif avg_prob > 0.5:
        node_roles[addr] = "Bot"
    else:
        node_roles[addr] = "Normal"

pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
edge_x, edge_y = [], []
for src, dst in G.edges():
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color="#AAA"), mode="lines", hoverinfo="none")
node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    role = node_roles.get(node, "Normal")
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Role: {role}")
    node_color.append("#007BFF" if role == "C&C" else "#FFB347" if role == "Bot" else "#CCCCCC")
    node_size.append(18 if role == "C&C" else 10)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    hovertext=node_text,
    hoverinfo="text",
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333")),
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Sensor {SELECTED_SENSOR_ID} - Botnet Network (SMOTEENN Model)",
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

html_output = os.path.join(graph_dir, f"BotnetGraph_Sensor{SELECTED_SENSOR_ID}_SMOTEENN_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")

print(f"\nDone. SMOTEENN model trained, report and graph saved for Sensor {SELECTED_SENSOR_ID}.")
