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
    auc
)
from imblearn.over_sampling import SMOTE
import plotly.subplots as sp
from libInternal import (
    variableDump,
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary
)

# job limit
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

SELECTED_SENSOR_ID = "1"
fileTimeStamp, output_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), using direct DuckDB connection.")
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

df = optimize_dataframe(df)
print(f"[Load] memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

df = fast_label_to_binary(df)

print("\n[DEBUG] Label distribution after conversion for Sensor:", SELECTED_SENSOR_ID)
print(df["Label"].value_counts())
print("\n[DEBUG] Group by Label for Sensor:", SELECTED_SENSOR_ID)
print(df[["SensorId", "Label"]].groupby("Label").size())

# Downsample majority
counts = df["Label"].value_counts()
if len(counts) > 1:
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    minority_count = counts.min()
    majority_count = counts.max()
    ratio = 5
    max_major = 800_000
    desired_major = min(max(int(minority_count * ratio), minority_count), max_major)

    if majority_count > desired_major:
        print(f"[Downsample] minority={minority_count:,}, majority={majority_count:,} -> sampling majority to {desired_major:,}")
        df_min = df[df["Label"] == minority_label]
        df_maj = df[df["Label"] == majority_label].sample(n=desired_major, random_state=42)
        df = pd.concat([df_min, df_maj], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
        gc.collect()
        print(f"[Downsample] New dataset size: {len(df):,}")
else:
    print("[Downsample] Only one class present (handled later)")

if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

features = [col for col in ["Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
if not features:
    raise RuntimeError("No valid numeric features found in dataset!")
print(f"[Features] Using features ({len(features)}): {features}")

X = df[features].fillna(df[features].mean())
y = df["Label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test distribution:", np.bincount(y_test))

# scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# apply SMOTE
print("\n[SMOTE] Balancing minority class...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("[SMOTE] Before:", np.bincount(y_train))
print("[SMOTE] After:", np.bincount(y_train_res))

# model setup
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=50, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(max_depth=6, random_state=42))
]
meta = ExtraTreesClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=1)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=3,
    n_jobs=1,
    passthrough=False,
    verbose=1
)

gc.collect()

print(f"\n[Train] Training stacking model (SMOTE) for Sensor {SELECTED_SENSOR_ID} ...")
stack.fit(X_train_res, y_train_res)
print("[Train] Done.")

# evaluation
p_test = stack.predict_proba(X_test_scaled)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Evaluation for Sensor (SMOTE):", SELECTED_SENSOR_ID)
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

report_path = None
try:
    report_path = generate_plotly_evaluation_report_smote(
        y_true=y_test,
        y_pred=y_pred_test,
        y_prob=p_test,
        sensor_id=SELECTED_SENSOR_ID,
        best_threshold=best_threshold,
        output_dir=graph_dir,
        file_timestamp=fileTimeStamp
    )
except Exception as e:
    print("[Report] Failed to generate:", e)

print(f"[Report] -> {report_path}")

# save data splits
train_csv = os.path.join(output_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_SMOTE_{fileTimeStamp}.csv")
test_csv = os.path.join(output_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_SMOTE_{fileTimeStamp}.csv")
pd.concat([X_train, pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

print(f"\nDone. Model (SMOTE) trained and report generated for Sensor {SELECTED_SENSOR_ID}.")

# === NEW: generate_plotly_evaluation_report_smote ===
def generate_plotly_evaluation_report_smote(y_true, y_pred, y_prob, sensor_id, best_threshold, output_dir, file_timestamp):
    """Generate Precision-Recall curve, ROC curve, and metric summary dashboard."""
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
        text=f"<b>Sensor {sensor_id} Evaluation Summary (SMOTE)</b><br>{text_metrics}<br><br><b>Best Threshold:</b> {best_threshold:.3f}",
        xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False, align="left"
    )

    fig.update_layout(
        title=f"Evaluation Dashboard (SMOTE) - Sensor {sensor_id}",
        title_x=0.5,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        width=1100
    )

    html_path = os.path.join(output_dir, f"EvaluationReport_Sensor{sensor_id}_SMOTE_{file_timestamp}.html")
    fig.write_html(html_path)
    print(f"[Report] Saved -> {html_path}")
    return html_path
