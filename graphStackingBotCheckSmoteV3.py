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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from libInternal import (
    variableDump,
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
    generate_plotly_evaluation_report_smote,
)

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# conf
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

# label convert
df = fast_label_to_binary(df)
print("\n[DEBUG] Label distribution after conversion for Sensor:", SELECTED_SENSOR_ID)
print(df["Label"].value_counts())
print("\n[DEBUG] Group by Label for Sensor:", SELECTED_SENSOR_ID)
print(df[["SensorId", "Label"]].groupby("Label").size())

if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

# clean & encode
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])
cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# feature selection
features = [col for col in ["Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
if not features:
    raise RuntimeError("No valid numeric features found in dataset!")
print(f"[Features] Using features ({len(features)}): {features}")

# split train 80/test 20
X = df[features].fillna(df[features].mean())
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train = np.asarray(y_train).astype(int)
y_test = np.asarray(y_test).astype(int)
print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test distribution:", np.bincount(y_test))

# scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# smote balancing
print("\n[SMOTE] Balancing classes for better recall performance...")
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_scaled, y_train)
print("[SMOTE] Before:", np.bincount(y_train))
print("[SMOTE] After :", np.bincount(y_train_balanced))

# esemble model
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(max_depth=10, learning_rate=0.05, random_state=42))
]
meta = LogisticRegression(class_weight="balanced", max_iter=500, solver="liblinear", random_state=42)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=3,
    n_jobs=1,
    passthrough=False,
    verbose=1
)

# train model
print(f"\n[Train] Training stacking model for Sensor {SELECTED_SENSOR_ID} ...")
try:
    stack.fit(X_train_balanced, y_train_balanced)
    print("[Train] Done.")
except Exception:
    print("[Train] Exception during stack.fit():")
    traceback.print_exc()

# evaluation error
p_test = stack.predict_proba(X_test_scaled)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5
best_threshold = max(0.3, best_threshold - 0.05)
y_pred_test = (p_test >= best_threshold).astype(int)

# metric sumary
accuracy = round((y_pred_test == y_test).mean() * 100, 2)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, p_test)

print(f"\n# Evaluation for Sensor: {SELECTED_SENSOR_ID}")
print(f"Best threshold: {best_threshold:.3f}")
print(f"Accuracy: {accuracy}%")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

report_path = generate_plotly_evaluation_report_smote(
    y_true=y_test,
    y_pred=y_pred_test,
    y_prob=p_test,
    sensor_id=SELECTED_SENSOR_ID,
    best_threshold=best_threshold,
    output_dir=graph_dir,
    file_timestamp=fileTimeStamp
)

# export train/test ke csv
train_csv = os.path.join(output_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
test_csv = os.path.join(output_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
pd.concat([X_train, pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

# apply model
df_scaled = scaler.transform(df[features])
df["PredictedProb"] = stack.predict_proba(df_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
df["PredictedRole"] = df["PredictedLabel"].apply(lambda x: "Botnet" if x == 1 else "Normal")

# generate graph
print(f"[Graph] Generating visualization for Sensor {SELECTED_SENSOR_ID} with {len(df)} edges...")
G = nx.from_pandas_edgelist(df, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

node_roles = {}
for addr in set(df["SrcAddr"]).union(df["DstAddr"]):
    subset = df[(df["SrcAddr"] == addr) | (df["DstAddr"] == addr)]
    avg_prob = subset["PredictedProb"].mean()
    if avg_prob > 0.85:
        node_roles[addr] = "C&C"
    elif avg_prob > 0.5:
        node_roles[addr] = "Bot"
    else:
        node_roles[addr] = "Normal"

pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)
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
        title=f"Sensor {SELECTED_SENSOR_ID} - Botnet Network (Global Model)",
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

html_output = os.path.join(graph_dir, f"BotnetGraph_Sensor{SELECTED_SENSOR_ID}_GlobalModel_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")

print(f"\nDone. Model trained and evaluation dashboard generated for Sensor {SELECTED_SENSOR_ID}.")
