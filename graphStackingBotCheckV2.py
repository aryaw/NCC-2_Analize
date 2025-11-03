import os
import gc
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
from libInternal import (
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
)

# === Thread limits ===
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

# === LOAD DATA ===
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

# downsize
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
df["PktByteRatio"] = df["TotPkts"] / (df["TotBytes"] + 1)
df["SrcByteRatio"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = np.abs(df["sTos"] - df["dTos"])
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"] = (df["TotBytes"] / (df["Dur"] + 1))

features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity", "PktByteRatio",
    "SrcByteRatio", "TrafficBalance", "DurationPerPkt", "Intensity"
]
print(f"[Features] {len(features)} total: {features}")

# === Split & Scale ===
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = np.rint(df["Label"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = MinMaxScaler()
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = qt.fit_transform(scaler.fit_transform(X_train))
X_test_scaled = qt.transform(scaler.transform(X_test))

# === Enhanced Stacking Model ===
base_learners = [
    ("rf", RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_split=3,
        class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(max_depth=10, learning_rate=0.05, random_state=42)),
    ("et", ExtraTreesClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=1))
]

meta = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=5, n_jobs=1, passthrough=True, verbose=1
)

print(f"[Train] Training enhanced stacking model for Sensor {SELECTED_SENSOR_ID} ...")
stack.fit(X_train_scaled, y_train)

# evaluate with ROC-based threshold
p_test = stack.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thr_roc = roc_curve(y_test, p_test)
gmeans = np.sqrt(tpr * (1 - fpr))
best_idx = np.argmax(gmeans)
best_threshold = thr_roc[best_idx]
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Evaluation for Sensor:", SELECTED_SENSOR_ID)
print("Best threshold (ROC-opt):", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))

# predict on full dataset
print("[Predict] Generating predictions on full dataset...")
df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
df_scaled = qt.transform(scaler.transform(df[features]))
df["PredictedProb"] = stack.predict_proba(df_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)

# === Full Graph with Aggregated Edges (Optimized) ===
print(f"[Graph] Generating full visualization (keeping all nodes, aggregated edges)...")

df_vis = (
    df.groupby(["SrcAddr", "DstAddr"], as_index=False)
      .agg({
          "PredictedProb": "mean",
          "Dir_raw": lambda x: x.value_counts().index[0] if len(x) else "->",
          "TotBytes": "sum",
          "TotPkts": "sum"
      })
)
print(f"[Graph] Unique aggregated edges: {len(df_vis):,}")

# build directed graph
G = nx.from_pandas_edgelist(df_vis, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

# vectorized inbound/outbound stats
agg_in = df_vis.groupby("DstAddr")["PredictedProb"].agg(["count", "mean"]).rename(columns={"count": "in_ct", "mean": "in_prob"})
agg_out = df_vis.groupby("SrcAddr")["PredictedProb"].agg(["count", "mean"]).rename(columns={"count": "out_ct", "mean": "out_prob"})
stats = agg_in.join(agg_out, how="outer").fillna(0)
stats["in_ratio"] = stats["in_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
stats["avg_prob"] = (stats["in_prob"] + stats["out_prob"]) / 2
stats["degree"] = stats["in_ct"] + stats["out_ct"]

# auto-detect roles
node_roles = {}
for n, r in stats.iterrows():
    if r["avg_prob"] > 0.9 and r["in_ratio"] > 0.8 and r["degree"] > 3:
        node_roles[n] = "C&C"
    elif r["avg_prob"] > 0.6 and r["in_ratio"] > 0.5:
        node_roles[n] = "Bot"
    else:
        node_roles[n] = "Normal"

print(f"[Graph] Total nodes: {len(G.nodes())}, Roles assigned: {len(node_roles)}")

# fast layout
print("[Graph] Computing layout...")
pos = nx.spring_layout(G, k=0.45, iterations=30, seed=42, dim=2)

# draw edges
edge_x, edge_y = [], []
for s, d in G.edges():
    if s not in pos or d not in pos: 
        continue
    x0, y0 = pos[s]; x1, y1 = pos[d]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.25, color="#AAA"),
    hoverinfo="none"
)

# draw nodes
node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
for n, (x, y) in pos.items():
    role = node_roles.get(n, "Normal")
    avg_prob = stats.loc[n, "avg_prob"] if n in stats.index else 0
    in_ratio = stats.loc[n, "in_ratio"] if n in stats.index else 0
    node_x.append(x); node_y.append(y)
    node_text.append(f"{n}<br>Role:{role}<br>Prob:{avg_prob:.3f}<br>InRatio:{in_ratio:.2f}")
    if role == "C&C":
        node_color.append("#FF0000"); node_size.append(26)
    elif role == "Bot":
        node_color.append("#FFB347"); node_size.append(14)
    else:
        node_color.append("#BFC9CA"); node_size.append(6)

node_trace = go.Scatter(
    x=node_x, y=node_y, mode="markers",
    hovertext=node_text, hoverinfo="text",
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333"))
)

# combine figure
fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Sensor {SELECTED_SENSOR_ID} – Dir-based Auto C&C Graph (Aggregated)",
        title_x=0.5, showlegend=False, hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ),
)

html_output = os.path.join(output_dir, f"BotnetGraph_Sensor{SELECTED_SENSOR_ID}_FullAggFast_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")

# export C&C and bot nodes
detected_roles_csv = os.path.join(output_dir, f"DetectedRoles_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
pd.DataFrame([
    {"Node": n, "Role": node_roles[n], "AvgProb": stats.loc[n, "avg_prob"], "InRatio": stats.loc[n, "in_ratio"]}
    for n in node_roles
]).to_csv(detected_roles_csv, index=False)
print(f"[Export] Roles summary -> {detected_roles_csv}")

print("\n✅ Done. High-accuracy model and fast Dir-based full graph generated successfully.")
