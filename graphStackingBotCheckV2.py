import os
import gc
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
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
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
)

# thread limits
os.environ.update({
    "JOBLIB_TEMP_FOLDER": "/tmp",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
})

SELECTED_SENSOR_ID = "1"
fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception:
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL AND SensorId = '{SELECTED_SENSOR_ID}'
"""
df = con.sql(query).df()
if df.empty:
    raise RuntimeError(f"No data for SensorId {SELECTED_SENSOR_ID}")
df = optimize_dataframe(df)
df = fast_label_to_binary(df)

# down sample
counts = df["Label"].value_counts()
if len(counts) > 1:
    min_label = counts.idxmin()
    maj_label = counts.idxmax()
    min_ct = counts.min()
    maj_ct = counts.max()
    ratio = 5
    desired_major = min(max(int(min_ct * ratio), min_ct), 800_000)
    if maj_ct > desired_major:
        df_min = df[df["Label"] == min_label]
        df_maj = df[df["Label"] == maj_label].sample(n=desired_major, random_state=42)
        df = pd.concat([df_min, df_maj], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
gc.collect()

# clean & encode
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])
dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)
for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# feature engineering
df["ByteRatio"] = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"] = df["TotPkts"] / (df["Dur"] + 0.1)
df["FlowIntensity"] = df["SrcBytes"] / (df["TotBytes"] + 1)
features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity"
]
print(f"[Features] {len(features)} total: {features}")

# split & scale
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = np.rint(df["Label"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# stacking
base_learners = [
    ("rf", RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_split=4,
        class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.05, random_state=42)),
]
meta = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=3, n_jobs=1, passthrough=True, verbose=1
)

print(f"[Train] Training stacking model for Sensor {SELECTED_SENSOR_ID} ...")
stack.fit(X_train_scaled, y_train)

# evaluate
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
print("F1:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))

# export train/test
train_csv = os.path.join(outputdata_dir, f"Train_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
test_csv = os.path.join(outputdata_dir, f"Test_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
pd.concat([X_train, pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] Train/Test data exported to {outputdata_dir}")

# predict on full dataset (NaN-safe)
print("[Predict] Generating predictions on full dataset...")
df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
df_scaled = scaler.transform(df[features])

try:
    df["PredictedProb"] = stack.predict_proba(df_scaled)[:, 1]
except ValueError as e:
    print(f"[WARN] NaN-safe fallback: {e}")
    meta_hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)
    stack.final_estimator = meta_hgb
    stack.fit(X_train_scaled, y_train)
    df["PredictedProb"] = stack.predict_proba(df_scaled)[:, 1]

df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)

# FULL GRAPH with EDGE AGGREGATION
print(f"[Graph] Generating full visualization (keeping all nodes, aggregated edges)...")

# aggregate duplicate edges by SrcAddr-DstAddr
df_vis = (
    df.groupby(["SrcAddr", "DstAddr"], as_index=False)
      .agg({
          "PredictedProb": "mean",
          "Dir_raw": "first",
          "TotBytes": "sum",
          "TotPkts": "sum"
      })
)
print(f"[Graph] Unique aggregated edges: {len(df_vis):,}")

# build directed graph
G = nx.from_pandas_edgelist(df_vis, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

# inbound/outbound + prob stats
inbound, outbound, prob_sum, edge_ct = {}, {}, {}, {}
for _, r in df_vis.iterrows():
    s, d, p, dr = r["SrcAddr"], r["DstAddr"], float(r["PredictedProb"]), str(r["Dir_raw"]).strip()
    for n in [s, d]:
        inbound.setdefault(n, 0)
        outbound.setdefault(n, 0)
        prob_sum.setdefault(n, 0)
        edge_ct.setdefault(n, 0)
    prob_sum[s] += p; prob_sum[d] += p
    edge_ct[s] += 1; edge_ct[d] += 1
    if dr == "->":
        outbound[s] += 1; inbound[d] += 1
    elif dr == "<-":
        outbound[d] += 1; inbound[s] += 1
    elif dr == "<->":
        outbound[s] += 1; outbound[d] += 1; inbound[s] += 1; inbound[d] += 1
    else:
        outbound[s] += 1; inbound[d] += 1

node_roles = {}
for node in G.nodes():
    in_ct = inbound.get(node, 0)
    out_ct = outbound.get(node, 0)
    in_ratio = in_ct / (in_ct + out_ct + 1e-9)
    avg_prob = prob_sum.get(node, 0) / max(edge_ct.get(node, 1), 1)
    deg = edge_ct.get(node, 0)
    if avg_prob > 0.9 and in_ratio > 0.6 and deg > 3:
        node_roles[node] = "C&C"
    elif avg_prob > 0.6 and in_ratio > 0.5:
        node_roles[node] = "Bot"
    else:
        node_roles[node] = "Normal"

print(f"[Graph] Total nodes: {len(G.nodes())}, Roles assigned: {len(node_roles)}")

# fast layout for large graph
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# edges
edge_x, edge_y, edge_colors = [], [], []
for _, r in df_vis.iterrows():
    s, d, dr = r["SrcAddr"], r["DstAddr"], str(r["Dir_raw"]).strip()
    if s not in pos or d not in pos:
        continue
    x0, y0 = pos[s]; x1, y1 = pos[d]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_colors.append("#66B2FF" if dr == "->" else "#FF6666" if dr == "<-" else "#9B59B6")

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.4, color="#AAA"), mode="lines", hoverinfo="none")

# nodes
node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
for n in G.nodes():
    x, y = pos[n]
    role = node_roles.get(n, "Normal")
    avg_prob = prob_sum.get(n, 0) / max(edge_ct.get(n, 1), 1)
    node_x.append(x); node_y.append(y)
    node_text.append(f"{n}<br>Role:{role}<br>Prob:{avg_prob:.3f}")
    node_color.append("#FF0000" if role == "C&C" else "#FFB347" if role == "Bot" else "#CCCCCC")
    node_size.append(22 if role == "C&C" else 12 if role == "Bot" else 6)

node_trace = go.Scatter(
    x=node_x, y=node_y, mode="markers",
    hovertext=node_text, hoverinfo="text",
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333"))
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Sensor {SELECTED_SENSOR_ID} - Full Graph (All Nodes, Aggregated Edges)",
        title_x=0.5, showlegend=False, hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

html_output = os.path.join(output_dir, f"BotnetGraph_Sensor{SELECTED_SENSOR_ID}_FullAgg_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")
print("\nDone. Full graph with all nodes and aggregated edges rendered successfully.")
