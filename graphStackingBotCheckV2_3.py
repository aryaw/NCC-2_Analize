import os
import gc
import re
import math
import duckdb
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# Plotly only for per-sensor charts & 3D graphs
import plotly.express as px
import plotly.graph_objects as go

# Safer threading on CPU-only boxes
try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits=1):
        yield

from libInternal import (
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
)

RANDOM_STATE = 42
MAX_ROWS_FOR_STACKING = 8_000_000
SAFE_THREADS = "1"
os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
    "NUMEXPR_NUM_THREADS": SAFE_THREADS,
    "MKL_THREADING_LAYER": "GNU",
    "JOBLIB_TEMP_FOLDER": "/tmp",
})

fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

#---- load------
try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception:
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL
"""

print("[Load] Reading dataset...")
df = con.sql(query).df()
if df.empty:
    raise RuntimeError("No labeled data found in dataset.")

df = optimize_dataframe(df)

# robust binary labels
df = fast_label_to_binary(df)
print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

# removes all rows from the DataFrame that have a NaN
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"]).copy()

# Dir mapping
dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

# Categorical encodings
for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# Feature engineering
df["ByteRatio"]      = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"]   = df["TotPkts"]  / (df["Dur"] + 0.1)
df["FlowIntensity"]  = df["SrcBytes"] / (df["TotBytes"] + 1)
df["PktByteRatio"]   = df["TotPkts"]  / (df["TotBytes"] + 1)
df["SrcByteRatio"]   = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = (df["sTos"] - df["dTos"]).abs()
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"]      = df["TotBytes"] / (df["Dur"] + 1)

features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity", "PktByteRatio",
    "SrcByteRatio", "TrafficBalance", "DurationPerPkt", "Intensity"
]

# replace ifinity & negatif infinity with NaN and fill with 0
X_full = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

# cast as int value
y_full = df["Label"].astype(int)

# sample guard for stacking memory
if len(df) > MAX_ROWS_FOR_STACKING:
    print(f"[Sample] Dataset too large ({len(df):,}), using {MAX_ROWS_FOR_STACKING:,} for model training...")
    df_sample = df.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE)
    X = df_sample[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_sample["Label"].astype(int)
else:
    X, y = X_full, y_full

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # float64
X_test_scaled  = scaler.transform(X_test)        # float64

#training: stacking (passthrough=False)
trained_model = None
print("\n[Train] Starting model training (Stacking, passthrough=False)...")

with threadpool_limits(limits=1):  # critical for stability on older CPUs
    try:
        base_learners = [
            ("rf", RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1, verbose=0
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=100, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=1, verbose=0
            )),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE, early_stopping=False
            )),
        ]

        meta = LogisticRegression(solver="lbfgs", max_iter=1000)

        stack = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            stack_method="predict_proba",
            passthrough=False,
            cv=2, n_jobs=1, verbose=1
        )

        stack.fit(X_train_scaled, y_train)
        trained_model = stack
        print("[Train] Stacking model trained successfully.")

    except Exception as e:
        print(f"[WARN] Stacking failed: {type(e).__name__}: {e}")
        print("[Fallback] Using RandomForest instead...")
        trained_model = RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        ).fit(X_train_scaled, y_train)
        print("[Fallback] RandomForest model trained.")

#evaluation (printed only)
p_test = trained_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thr_roc = roc_curve(y_test, p_test)
best_threshold = thr_roc[np.argmax(np.sqrt(tpr * (1 - fpr)))]
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Global Model Evaluation")
print("Best threshold:", round(float(best_threshold), 4))
print("Accuracy:", round(float((y_pred_test == y_test).mean() * 100), 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:",    recall_score(y_test, y_pred_test))
print("F1:",        f1_score(y_test, y_pred_test))
print("ROC-AUC:",   roc_auc_score(y_test, p_test))

# cleanup global split arrays (no global Plotly)
del X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
del p_test, fpr, tpr, thr_roc, y_pred_test
gc.collect()

# full inference
X_all_scaled = scaler.transform(X_full)     # float64
df["PredictedProb"] = trained_model.predict_proba(X_all_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
del X_all_scaled, X_full, y_full
gc.collect()

# 3D C&C GRAPH (C&C-only, cap 100 nodes)
def make_3d_cnc_graph_cnc_only(df_sensor, cnc_nodes, out_dir, sid, ts, max_nodes=100):
    if len(cnc_nodes) == 0:
        print(f"[3D] Sensor {sid}: no C&C nodes to render plot.")
        return

    cnc_set = set(map(str, cnc_nodes))

    # Keep edges where BOTH endpoints are C&C nodes
    edges_agg = (
        df_sensor.groupby(["SrcAddr", "DstAddr"])
        .size().reset_index(name="weight")
    )
    edges_agg["SrcAddr"] = edges_agg["SrcAddr"].astype(str)
    edges_agg["DstAddr"] = edges_agg["DstAddr"].astype(str)
    cnc_edges = edges_agg[
        edges_agg["SrcAddr"].isin(cnc_set) & edges_agg["DstAddr"].isin(cnc_set)
    ].copy()

    if cnc_edges.empty:
        print(f"[3D] Sensor {sid}: no C&C edges; skipping 3D graph.")
        return

    # Weighted degree among C&C-only subgraph
    deg_df = pd.concat([
        cnc_edges.groupby("SrcAddr")["weight"].sum().rename("out_w"),
        cnc_edges.groupby("DstAddr")["weight"].sum().rename("in_w")
    ], axis=1).fillna(0.0)
    deg_df["degree_w"] = deg_df["in_w"] + deg_df["out_w"]

    # Select top-N C&C nodes by degree_w
    top_nodes = deg_df.sort_values("degree_w", ascending=False).head(max_nodes).index.tolist()
    top_set = set(top_nodes)

    # Subset edges to only top-N nodes
    sub_edges = cnc_edges[
        cnc_edges["SrcAddr"].isin(top_set) & cnc_edges["DstAddr"].isin(top_set)
    ].copy()

    if sub_edges.empty:
        print(f"[3D] Sensor {sid}: top-{max_nodes} C&C nodes have no mutual edges.")
        return

    # Build graph
    G = nx.DiGraph()
    for n in top_nodes:
        degw = float(deg_df.loc[n, "degree_w"]) if n in deg_df.index else 0.0
        G.add_node(n, role="C&C", degree_w=degw)

    for _, row in sub_edges.iterrows():
        G.add_edge(row["SrcAddr"], row["DstAddr"], weight=float(row["weight"]))

    # Layout & figure
    pos3d = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=100)

    def edge_trace():
        xs, ys, zs = [], [], []
        for u, v, d in G.edges(data=True):
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=2),
            opacity=0.7,
            name="Edges (C&C↔C&C)"
        )

    def node_trace():
        xs, ys, zs, texts, sizes = [], [], [], [], []
        for n, data in G.nodes(data=True):
            x, y, z = pos3d[n]
            xs.append(x); ys.append(y); zs.append(z)
            degw = float(data["degree_w"])
            sizes.append(6 + 4 * math.log1p(degw))
            texts.append(f"{n}<br>weighted_degree={degw:.0f}")
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=sizes),
            name="C&C nodes",
            text=texts, hoverinfo="text"
        )

    fig = go.Figure(data=[edge_trace(), node_trace()])
    fig.update_layout(
        title=f"Sensor {sid} – 3D C&C Network (C&C-only, top {min(max_nodes, len(top_nodes))} nodes)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    out_path = os.path.join(out_dir, f"Sensor{sid}_3D_CNC_onlyTop{min(max_nodes, len(top_nodes))}_{ts}.html")
    fig.write_html(out_path)
    print(f"[3D] Saved 3D C&C-only network -> {out_path}")

#-- per-sensor C&C detection--
detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== [Sensor {sid}] Auto C&C Detection ===")
    df_s = df[df["SensorId"] == sid].copy()

    # per-node stats (in/out)
    agg_in = df_s.groupby("DstAddr")["PredictedProb"].agg(["count", "mean"]) \
                 .rename(columns={"count": "in_ct", "mean": "in_prob"})
    agg_out = df_s.groupby("SrcAddr")["PredictedProb"].agg(["count", "mean"]) \
                  .rename(columns={"count": "out_ct", "mean": "out_prob"})
    stats = agg_in.join(agg_out, how="outer").fillna(0)

    stats["in_ratio"]  = stats["in_ct"]  / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["out_ratio"] = stats["out_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["avg_prob"]  = (stats["in_prob"] + stats["out_prob"]) / 2
    stats["degree"]    = stats["in_ct"] + stats["out_ct"]

    node_roles = {}
    for n, r in stats.iterrows():
        # deg_thr = max(15, stats["degree"].quantile(0.90))
        # prob_thr = max(0.55, stats["avg_prob"].quantile(0.80))
        # ratio_thr = 0.65

        # if (r["avg_prob"] > prob_thr) and (r["degree"] > deg_thr) and ((r["in_ratio"] > ratio_thr) or (r["out_ratio"] > ratio_thr)):
        #     node_roles[n] = "C&C"

        # if (r["avg_prob"] > 0.60) and (r["degree"] > 100) and ((r["in_ratio"] > 0.60) or (r["out_ratio"] > 0.70)):
        if (r["avg_prob"] > 0.60) and (r["degree"] > 100) and (r["out_ratio"] > 0.70):
            node_roles[n] = "C&C"
        else:
            node_roles[n] = "Normal"

    cc_nodes = [n for n, role in node_roles.items() if role == "C&C"]
    print(f"[Detected] {len(cc_nodes)} potential C&C nodes")

    if cc_nodes:
        cnc_df = stats.loc[cc_nodes].copy()
        cnc_df["SensorId"] = sid
        cnc_df["cnc_score"] = cnc_df["avg_prob"] * (1 + cnc_df["in_ratio"] + cnc_df["out_ratio"]) * np.log1p(cnc_df["degree"])
        cnc_df = cnc_df.sort_values("cnc_score", ascending=False)
        detected_summary.append(cnc_df)

        cnc_top = cnc_df.sort_values("cnc_score", ascending=False).head(20).copy()
        cnc_top["ip"] = cnc_top.index.astype(str)
        bar_fig = px.bar(
            cnc_top, x="ip", y="cnc_score",
            hover_data=["avg_prob", "degree", "in_ratio", "out_ratio"],
            title=f"Sensor {sid} – Top C&C Candidates (Top 20)"
        )
        bar_fig.update_layout(xaxis_tickangle=-45)
        bar_fig.write_html(os.path.join(output_dir, f"Sensor{sid}_CNC_Score_{fileTimeStamp}.html"))

        # Per-sensor scatter (all C&C)
        scatter_fig = px.scatter(
            cnc_df.reset_index().rename(columns={"index": "ip"}),
            x="degree", y="avg_prob", size="cnc_score",
            hover_name="ip",
            hover_data=["in_ratio", "out_ratio"],
            title=f"Sensor {sid} – AvgProb vs Degree (C&C)"
        )
        scatter_fig.write_html(os.path.join(output_dir, f"Sensor{sid}_CNC_Scatter_{fileTimeStamp}.html"))

        # 3D C&C-only network (cap 100 nodes)
        make_3d_cnc_graph_cnc_only(
            df_sensor=df_s,
            cnc_nodes=cnc_df.index.tolist(),
            out_dir=output_dir,
            sid=sid,
            ts=fileTimeStamp,
            max_nodes=100
        )
    else:
        print("[Info] No C&C nodes detected — printing top suspicious candidates")
        temp = stats.copy()
        temp["score"] = temp["avg_prob"] * np.log1p(temp["degree"])
        print(temp.sort_values("score", ascending=False).head(10)[
            ["avg_prob", "degree", "in_ratio", "out_ratio", "score"]
        ])


# export CSV summary
if detected_summary:
    summary_df = pd.concat(detected_summary, ignore_index=True)
    summary_csv = os.path.join(output_dir, f"CNC_AutoDetected_SafeAll_{fileTimeStamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[Export] Saved -> {summary_csv}")
else:
    print("\n[Summary] No C&C detected in any sensor.")

print("\nDone. Stacking (passthrough=False) + per-sensor Top-20 bar + scatter + C&C-only 3D (top 100 nodes) complete.")
