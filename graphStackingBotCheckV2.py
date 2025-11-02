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
)
from libInternal import (
    variableDump,
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
    generate_plotly_evaluation_report
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
fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
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

# keep all minority rows, sample majority to (minority_count * ratio) capped at max_major
counts = df["Label"].value_counts()
if len(counts) > 1:
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    minority_count = counts.min()
    majority_count = counts.max()
    ratio = 5                  # how many majority rows per minority row to keep
    max_major = 800_000        # absolute cap for majority rows (tune as needed)
    desired_major = min(max(int(minority_count * ratio), minority_count), max_major)

    if majority_count > desired_major:
        print(f"[Downsample] minority={minority_count:,}, majority={majority_count:,} -> sampling majority to {desired_major:,}")
        df_min = df[df["Label"] == minority_label]
        df_maj = df[df["Label"] == majority_label].sample(n=desired_major, random_state=42)
        
        # if there are more than 2 labels (unlikely), keep other labels as-is
        other = df[~df["Label"].isin([minority_label, majority_label])]
        df = pd.concat([df_min, df_maj, other], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
        gc.collect()
        print(f"[Downsample] New dataset size: {len(df):,}")
    else:
        print(f"[Downsample] No downsampling needed (majority {majority_count:,} <= desired {desired_major:,})")
else:
    print("[Downsample] Only one class present (handled later)")

# Safety check for binary class existence
if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

# drop irrelevant/missing features
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

# encode categorycal
cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# feature
features = [col for col in ["Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
if not features:
    raise RuntimeError("No valid numeric features found in dataset!")
print(f"[Features] Using features ({len(features)}): {features}")

# split data train 80/test 20
X = df[features].fillna(df[features].mean())
y = df["Label"]

y = np.rint(y).astype(int)   # cast labels to discrete ints
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train = np.asarray(y_train).astype(int)
y_test = np.asarray(y_test).astype(int)
print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test distribution:", np.bincount(y_test))

# scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ensure integer labels
y_train = np.rint(y_train).astype(int)
y_test = np.rint(y_test).astype(int)
print("[Check] Unique y_train labels:", np.unique(y_train))

# ensemble model
# base_learners = [
#     ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=1)),
#     ("hgb", HistGradientBoostingClassifier(max_depth=8, random_state=42))
# ]
# meta = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)

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

# clean up memory
gc.collect()

# train
print(f"\n[Train] Training stacking model for Sensor {SELECTED_SENSOR_ID} ...")
try:
    stack.fit(X_train_scaled, y_train)
    print("[Train] Done.")
except Exception:
    print("[Train] Exception during stack.fit():")
    traceback.print_exc()

# clean up memory
gc.collect()

# check score
p_test = stack.predict_proba(X_test_scaled)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Evaluation for Sensor:", SELECTED_SENSOR_ID)
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

report_paths = generate_plotly_evaluation_report(
    y_true=y_test,
    y_pred=y_pred_test,
    y_prob=p_test,
    sensor_id=SELECTED_SENSOR_ID,
    best_threshold=best_threshold,
    output_dir=graph_dir,
    file_timestamp=fileTimeStamp
)

# export train/test for log
train_csv = os.path.join(outputdata_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
test_csv = os.path.join(outputdata_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
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

print(f"\nDone. Model trained and graph generated for Sensor {SELECTED_SENSOR_ID}.")
