import os
import gc
import joblib
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import traceback
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

if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

# Encode categorical
cat_cols = ["Proto", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

features = [col for col in ["Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
print(f"[Features] Using features ({len(features)}): {features}")

X = df[features].fillna(df[features].mean())
y = np.rint(df["Label"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test distribution:", np.bincount(y_test))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
print(f"\n[Train] Training stacking model for Sensor {SELECTED_SENSOR_ID} ...")
stack.fit(X_train_scaled, y_train)
print("[Train] Done.")
gc.collect()

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

train_csv = os.path.join(outputdata_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
test_csv = os.path.join(outputdata_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
pd.concat([X_train, pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

df_scaled = scaler.transform(df[features])
df["PredictedProb"] = stack.predict_proba(df_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)

print(f"[Graph] Generating visualization for Sensor {SELECTED_SENSOR_ID} with {len(df)} edges...")

MAX_EDGES = 5000
df_vis = df.sample(n=MAX_EDGES, random_state=42) if len(df) > MAX_EDGES else df.copy()

# Create graph
G = nx.from_pandas_edgelist(df_vis, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

# Direction-aware inbound/outbound analysis
inbound_counts, outbound_counts, prob_sum, edge_count = {}, {}, {}, {}

for _, row in df_vis.iterrows():
    src, dst, prob, dir_raw = row["SrcAddr"], row["DstAddr"], float(row["PredictedProb"]), str(row["Dir_raw"]).strip()

    for node in [src, dst]:
        inbound_counts.setdefault(node, 0)
        outbound_counts.setdefault(node, 0)
        prob_sum.setdefault(node, 0.0)
        edge_count.setdefault(node, 0)

    prob_sum[src] += prob
    prob_sum[dst] += prob
    edge_count[src] += 1
    edge_count[dst] += 1

    # interpret Dir arrow
    if dir_raw == "->":
        outbound_counts[src] += 1
        inbound_counts[dst] += 1
    elif dir_raw == "<-":
        outbound_counts[dst] += 1
        inbound_counts[src] += 1
    elif dir_raw == "<->":
        outbound_counts[src] += 1
        outbound_counts[dst] += 1
        inbound_counts[src] += 1
        inbound_counts[dst] += 1
    else:
        outbound_counts[src] += 1
        inbound_counts[dst] += 1

# Assign node roles using inbound/outbound behavior + probability
node_roles = {}
for node in set(list(inbound_counts.keys()) + list(outbound_counts.keys())):
    in_ct = inbound_counts.get(node, 0)
    out_ct = outbound_counts.get(node, 0)
    in_ratio = in_ct / (in_ct + out_ct + 1e-9)
    avg_prob = prob_sum.get(node, 0.0) / max(edge_count.get(node, 1), 1)
    deg = edge_count.get(node, 0)

    if avg_prob >= 0.9 and in_ratio >= 0.6 and deg >= 3:
        node_roles[node] = "C&C"
    elif avg_prob >= 0.6 and in_ratio >= 0.5 and deg >= 2:
        node_roles[node] = "C&C"
    elif avg_prob >= 0.5:
        node_roles[node] = "Bot"
    else:
        node_roles[node] = "Normal"

# layout and edges
pos = nx.kamada_kawai_layout(G)
edge_x, edge_y, edge_colors = [], [], []

for _, row in df_vis.iterrows():
    src, dst, dir_raw = row["SrcAddr"], row["DstAddr"], str(row["Dir_raw"]).strip()
    if src not in pos or dst not in pos:
        continue
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

    if dir_raw == "->":
        edge_colors.append("#66B2FF")  # outbound
    elif dir_raw == "<-":
        edge_colors.append("#FF6666")  # inbound
    elif dir_raw == "<->":
        edge_colors.append("#9B59B6")  # bidirectional
    else:
        edge_colors.append("#AAAAAA")

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.6, color="#AAA"),
    mode="lines", hoverinfo="none"
)

# nodes
node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    role = node_roles.get(node, "Normal")
    avg_prob = prob_sum.get(node, 0.0) / max(edge_count.get(node, 1), 1)
    in_ct = inbound_counts.get(node, 0)
    out_ct = outbound_counts.get(node, 0)
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Role: {role}<br>AvgProb: {avg_prob:.3f}<br>In/Out: {in_ct}/{out_ct}")
    if role == "C&C":
        node_color.append("#FF0000")
        node_size.append(28)
    elif role == "Bot":
        node_color.append("#FFB347")
        node_size.append(14)
    else:
        node_color.append("#CCCCCC")
        node_size.append(8)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers",
    hovertext=node_text,
    hoverinfo="text",
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333")),
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Sensor {SELECTED_SENSOR_ID} - Botnet Network (Dir-based Detection, Sampled 5K)",
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

html_output = os.path.join(graph_dir, f"BotnetGraph_Sensor{SELECTED_SENSOR_ID}_DirBased_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")

print(f"\n✅ Done. Model trained and direction-aware graph generated for Sensor {SELECTED_SENSOR_ID}.")
