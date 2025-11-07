import os
import gc
import re
import duckdb
import pandas as pd
import numpy as np

# Network graph imports
import math
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

# Plotly for exports
import plotly.graph_objects as go
import plotly.express as px

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

#-- config--
RANDOM_STATE = 42
MAX_ROWS_FOR_STACKING = 7_000_000
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

#-- load----
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

df = fast_label_to_binary(df)
print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

# preprocessing & features
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"]).copy()

dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

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

X_full = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_full = df["Label"].astype(int)

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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# training
trained_model = None
print("\n[Train] Starting model training (Stacking, passthrough=False)...")

with threadpool_limits(limits=1):
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

# evaluation
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

# plot: ROC
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             name='Random', line=dict(dash='dash')))
roc_fig.update_layout(
    title=f"ROC Curve (AUC = {roc_auc_score(y_test, p_test):.4f})",
    xaxis_title="FPR", yaxis_title="TPR"
)
roc_fig.write_html(os.path.join(output_dir, f"Global_ROC_{fileTimeStamp}.html"))

# plot: PR curve
precisions, recalls = [], []
for t in thr_roc:
    preds = (p_test >= t).astype(int)
    prc = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    precisions.append(prc)
    recalls.append(rec)

pr_fig = go.Figure()
pr_fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name='PR'))
pr_fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
pr_fig.write_html(os.path.join(output_dir, f"Global_PR_{fileTimeStamp}.html"))

# plot: confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=["Benign", "Malicious"], y=["Benign", "Malicious"])
cm_fig.update_layout(title="Confusion Matrix")
cm_fig.write_html(os.path.join(output_dir, f"Global_CM_{fileTimeStamp}.html"))

del X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, p_test, fpr, tpr, thr_roc, y_pred_test, cm
gc.collect()

# full inference
X_all_scaled = scaler.transform(X_full)
df["PredictedProb"] = trained_model.predict_proba(X_all_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
del X_all_scaled, X_full, y_full
gc.collect()

# 3D C&C GRAPH FUNCTION
def make_3d_cnc_graph(df_sensor, cnc_nodes, out_dir, sid, ts, normal_edges_cap=100):
    edges_agg = (
        df_sensor.groupby(["SrcAddr", "DstAddr"])
        .size().reset_index(name="weight")
    )

    cnc_set = set(map(str, cnc_nodes))
    edges_agg["is_cnc_edge"] = (
        edges_agg["SrcAddr"].astype(str).isin(cnc_set) |
        edges_agg["DstAddr"].astype(str).isin(cnc_set)
    )

    cnc_edges = edges_agg[edges_agg["is_cnc_edge"]].copy()
    normal_pool = edges_agg[~edges_agg["is_cnc_edge"]].copy()

    if len(normal_pool) > 0:
        normal_sample = normal_pool.sample(
            n=min(normal_edges_cap, len(normal_pool)),
            random_state=RANDOM_STATE
        )
    else:
        normal_sample = normal_pool

    edges_keep = pd.concat([cnc_edges, normal_sample], ignore_index=True)

    node_ids = pd.unique(pd.concat([
        edges_keep["SrcAddr"].astype(str),
        edges_keep["DstAddr"].astype(str)
    ], ignore_index=True))

    deg_df = pd.concat([
        edges_keep.groupby("SrcAddr")["weight"].sum().rename("out_w"),
        edges_keep.groupby("DstAddr")["weight"].sum().rename("in_w")
    ], axis=1).fillna(0.0)
    deg_df["degree_w"] = deg_df["in_w"] + deg_df["out_w"]

    G = nx.DiGraph()
    for n in node_ids:
        role = "C&C" if n in cnc_set else "Normal"
        degw = float(deg_df.loc[n, "degree_w"]) if n in deg_df.index else 0.0
        G.add_node(n, role=role, degree_w=degw)

    for _, row in edges_keep.iterrows():
        G.add_edge(str(row["SrcAddr"]),
                   str(row["DstAddr"]),
                   weight=float(row["weight"]),
                   is_cnc_edge=bool(row["is_cnc_edge"]))

    if len(G.nodes) == 0:
        print(f"[3D] Sensor {sid}: nothing to plot.")
        return

    pos3d = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=100)
    import plotly.graph_objects as go

    def edge_trace(mask_is_cnc):
        xs, ys, zs = [], [], []
        for u, v, d in G.edges(data=True):
            if bool(d["is_cnc_edge"]) != mask_is_cnc:
                continue
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=3 if mask_is_cnc else 1),
            opacity=0.9 if mask_is_cnc else 0.3,
            name="C&C edges" if mask_is_cnc else "Normal edges"
        )

    def node_trace(role):
        xs, ys, zs, texts, sizes = [], [], [], [], []
        for n, data in G.nodes(data=True):
            if data["role"] != role:
                continue
            x, y, z = pos3d[n]
            xs.append(x); ys.append(y); zs.append(z)
            degw = float(data["degree_w"])
            size = 6 + 4 * math.log1p(degw)
            sizes.append(size)
            texts.append(f"{n}<br>deg={degw:.0f}")
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes),
            name=f"{role} nodes",
            text=texts, hoverinfo="text"
        )

    fig = go.Figure(data=[
        edge_trace(mask_is_cnc=True),
        edge_trace(mask_is_cnc=False),
        node_trace("C&C"),
        node_trace("Normal"),
    ])

    fig.update_layout(
        title=f"Sensor {sid} – 3D C&C Network",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    out_path = os.path.join(out_dir, f"Sensor{sid}_3D_CNC_{ts}.html")
    fig.write_html(out_path)
    print(f"[3D] Saved 3D network -> {out_path}")


# per-sensor C&C detection
detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== [Sensor {sid}] Auto C&C Detection ===")
    df_s = df[df["SensorId"] == sid].copy()

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
        if (r["avg_prob"] > 0.70) and (r["degree"] > 110) and ((r["in_ratio"] > 0.70) or (r["out_ratio"] > 0.70)):
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
            title=f"Sensor {sid} – Top C&C Candidates"
        )
        bar_fig.update_layout(xaxis_tickangle=-45)
        bar_fig.write_html(os.path.join(output_dir, f"Sensor{sid}_CNC_Score_{fileTimeStamp}.html"))

        scatter_fig = px.scatter(
            cnc_df.reset_index().rename(columns={"index": "ip"}),
            x="degree", y="avg_prob", size="cnc_score",
            hover_name="ip",
            title=f"Sensor {sid} – AvgProb vs Degree"
        )
        scatter_fig.write_html(os.path.join(output_dir, f"Sensor{sid}_CNC_Scatter_{fileTimeStamp}.html"))

        make_3d_cnc_graph(
            df_sensor=df_s,
            cnc_nodes=cnc_df.index.tolist(),
            out_dir=output_dir,
            sid=sid,
            ts=fileTimeStamp,
            normal_edges_cap=100
        )

    else:
        print("[Info] No C&C nodes detected in this sensor.")

print("\nDone. Memory-safe stacking + per-sensor C&C + 3D graph export complete.")
