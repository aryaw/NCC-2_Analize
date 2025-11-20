import os
import gc
import re
import duckdb
import pandas as pd
import numpy as np
import psutil
from datetime import datetime

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
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

import plotly.graph_objects as go
import networkx as nx

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
    detect_cnc_from_label
)



RANDOM_STATE = 42
MAX_ROWS_FOR_STACKING = 8_500_000
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

memory_log_path = os.path.join(output_dir, f"memory_trace_{fileTimeStamp}.csv")
with open(memory_log_path, "w") as f:
    f.write("timestamp,tag,current_mb,vms_mb\n")

def log_ram(tag=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024)
    vms = getattr(process.memory_info(), "vms", 0) / (1024 * 1024)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[RAM] {tag:<25} Current={rss:8.2f} MB | VMS={vms:8.2f} MB")
    with open(memory_log_path, "a") as f:
        f.write(f"{now},{tag},{rss:.2f},{vms:.2f}\n")



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
    AND REGEXP_MATCHES(SrcAddr, '^[0-9.]+$')
"""

print("[Load] Reading dataset...")
df = con.sql(query).df()
log_ram("After Load CSV")

df = detect_cnc_from_label(df)
df = optimize_dataframe(df)
df = fast_label_to_binary(df)

df = df.dropna(subset=[
    "SrcAddr","DstAddr","Dir","Proto","Dur","TotBytes","TotPkts","Label"
]).copy()
log_ram("After DropNA")

dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

df["ByteRatio"] = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"] = df["TotPkts"] / (df["Dur"] + 0.1)
df["FlowIntensity"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["PktByteRatio"] = df["TotPkts"] / (df["TotBytes"] + 1)
df["SrcByteRatio"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = (df["sTos"] - df["dTos"]).abs()
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"] = df["TotBytes"] / (df["Dur"] + 1)

features = [
    "Dir","Dur","Proto","TotBytes","TotPkts","sTos","dTos","SrcBytes",
    "ByteRatio","DurationRate","FlowIntensity","PktByteRatio",
    "SrcByteRatio","TrafficBalance","DurationPerPkt","Intensity"
]

X_full = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_full = df["Label"].astype(int)
log_ram("After Feature Select")

if len(df) > MAX_ROWS_FOR_STACKING:
    df_sample = df.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE)
    X = df_sample[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_sample["Label"].astype(int)
    print(f"[Data] Data too big, {MAX_ROWS_FOR_STACKING} used")
else:
    print(f"[Data] All data, {len(df)} used")
    X, y = X_full, y_full
log_ram("After Sampling")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
log_ram("After Train/Test Split")

# EXPORT split to CSV (existing feature)
pd.DataFrame(X_train).to_csv(os.path.join(outputdata_dir, f"train_features_{fileDataTimeStamp}.csv"), index=False)
pd.DataFrame(y_train, columns=["Label"]).to_csv(os.path.join(outputdata_dir, f"train_labels_{fileDataTimeStamp}.csv"), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(outputdata_dir, f"test_features_{fileDataTimeStamp}.csv"), index=False)
pd.DataFrame(y_test, columns=["Label"]).to_csv(os.path.join(outputdata_dir, f"test_labels_{fileDataTimeStamp}.csv"), index=False)
print(f"[Export] Train/Test split exported to CSV in {outputdata_dir}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_ram("After Scaling")
print("\n[Train] Training model...")

with threadpool_limits(limits=1):
    try:
        base_learners = [
            ("rf", RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
            )),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE
            )),
        ]

        meta = LogisticRegression(solver="lbfgs", max_iter=1000)
        model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            stack_method="predict_proba",
            passthrough=False, cv=2, n_jobs=1
        )
        model.fit(X_train_scaled, y_train)
        print("[Train] Stacking model OK")

    except Exception as e:
        print("[Fallback] Using RandomForest:", e)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        ).fit(X_train_scaled, y_train)
log_ram("After Model Train")


p_test = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thr = roc_curve(y_test, p_test)
best_threshold = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]

print("\n# Global Model Evaluation")
y_pred_test = (p_test >= best_threshold).astype(int)
print("Best threshold:", round(float(best_threshold), 4))
print("Accuracy:", round(float((y_pred_test == y_test).mean() * 100), 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:",    recall_score(y_test, y_pred_test))
print("F1:",        f1_score(y_test, y_pred_test))
print("ROC-AUC:",   roc_auc_score(y_test, p_test))

df["PredictedProb"]  = model.predict_proba(scaler.transform(X_full))[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
log_ram("After Full Inference")
gc.collect()


detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== Sensor {sid} ===")
    df_s = df[df["SensorId"] == sid].copy()

    # Aggregate inbound/outbound
    agg_in  = df_s.groupby("DstAddr")["PredictedProb"].agg(["count","mean"])
    agg_out = df_s.groupby("SrcAddr")["PredictedProb"].agg(["count","mean"])

    agg_in.columns  = ["in_ct","in_prob"]
    agg_out.columns = ["out_ct","out_prob"]

    stats = agg_in.join(agg_out, how="outer").fillna(0)
    stats["degree"] = stats["in_ct"] + stats["out_ct"]
    stats["in_ratio"]  = stats["in_ct"]  / (stats["degree"] + 1e-9)
    stats["out_ratio"] = stats["out_ct"] / (stats["degree"] + 1e-9)
    stats["avg_prob"]  = (stats["in_prob"] + stats["out_prob"]) / 2

    # Existing C&C Logic (untouched)
    node_roles = {}
    for n, r in stats.iterrows():
        if (r["avg_prob"] > 0.70) and (r["out_ct"] > 120) and (r["out_ratio"] > 0.55):
            node_roles[n] = "C&C"
        else:
            node_roles[n] = "Normal"

    # cc_nodes = [n for n, role in node_roles.items() if role == "C&C"]
    # stats["Role"] = stats.index.map(lambda x: node_roles[x])

    real_cnc_nodes = df_s[df_s["LabelCNC"] == 1]["DstAddr"].unique().tolist()
    cc_nodes = sorted(set([
        *[n for n, role in node_roles.items() if role == "C&C"], 
        *real_cnc_nodes
    ]))
    stats["Role"] = stats.index.map(lambda x: "C&C" if x in cc_nodes else "Normal")

    cnc_path   = os.path.join(outputdata_dir, f"Sensor{sid}_CNC_Detected_{fileDataTimeStamp}.csv")
    normal_path= os.path.join(outputdata_dir, f"Sensor{sid}_NormalNodes_{fileDataTimeStamp}.csv")

    stats[stats["Role"] == "C&C"].to_csv(cnc_path)
    stats[stats["Role"] == "Normal"].to_csv(normal_path)
    print(f"[Export] Sensor {sid} C&C + Normal exported")

    # C&C Score (existing)
    if len(cc_nodes) > 0:
        cnc_df = stats.loc[cc_nodes].copy()
        cnc_df["SensorId"] = sid
        cnc_df["cnc_score"] = cnc_df["avg_prob"] * \
                              (1 + cnc_df["in_ratio"] + cnc_df["out_ratio"]) * \
                              np.log1p(cnc_df["degree"])
        detected_summary.append(cnc_df)
        log_ram(f"Sensor {sid} After C&C Score")

        # Graph Generation (existing)
        print("[Plot] Generating 3D interactive network graph")

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 4:
            MAX_NODES, MAX_EDGES = 200, 400
        elif available_gb < 8:
            MAX_NODES, MAX_EDGES = 300, 600
        else:
            MAX_NODES, MAX_EDGES = 500, 800

        log_ram(f"Sensor {sid} Graph Limits: {MAX_NODES} nodes, {MAX_EDGES} edges")

        G_full = nx.from_pandas_edgelist(df_s, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

        cnc_neighbors = set()
        for cnc in cc_nodes:
            if cnc in G_full:
                cnc_neighbors.update(G_full.predecessors(cnc))
                cnc_neighbors.update(G_full.successors(cnc))

        nodes_keep = set(cc_nodes) | cnc_neighbors
        if len(nodes_keep) > MAX_NODES:
            normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
            sample_normal = np.random.choice(normal_nodes, size=min(MAX_NODES, len(normal_nodes)), replace=False)
            nodes_keep = set(cc_nodes) | set(sample_normal)

        G = G_full.subgraph(nodes_keep).copy()
        del G_full

        all_edges = list(G.edges())
        if len(all_edges) > MAX_EDGES:
            edges_sample = np.random.choice(len(all_edges), size=MAX_EDGES, replace=False)
            edges_keep = [all_edges[i] for i in edges_sample]
            G = nx.DiGraph(edges_keep)

        print(f"[Plot] Drawing {len(G.nodes())} nodes and {len(G.edges())} edges")

        node_color = ["red" if n in cc_nodes else "blue" for n in G.nodes()]
        node_size  = [10   if n in cc_nodes else 4     for n in G.nodes()]

        pos = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=60)

        x_nodes = [pos[k][0] for k in G.nodes()]
        y_nodes = [pos[k][1] for k in G.nodes()]
        z_nodes = [pos[k][2] for k in G.nodes()]

        edge_x, edge_y, edge_z = [], [], []
        for e in G.edges():
            x0, y0, z0 = pos[e[0]]
            x1, y1, z1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none'
        )

        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            marker=dict(size=node_size, color=node_color, opacity=0.9),
            text=[f"{n}" for n in G.nodes()],
            hoverinfo='text'
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Sensor {sid} – 3D Network Graph (Safe, {len(G.nodes())} nodes)",
                showlegend=False,
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(showbackground=False),
                    yaxis=dict(showbackground=False),
                    zaxis=dict(showbackground=False)
                ),
                annotations=[dict(
                    text="Red = C&C Node | Blue = Normal Node (sampled)",
                    showarrow=False,
                    xref="paper", yref="paper", x=0, y=-0.05
                )]
            )
        )

        graph_path = os.path.join(output_dir, f"Sensor{sid}_3DGraphSafe_{fileTimeStamp}.html")
        fig.write_html(graph_path)
        print(f"[Plot] 3D safe network graph saved -> {graph_path}")
        log_ram(f"Sensor {sid} After Safe Plot")

    else:
        print("[Info] No C&C nodes detected in this sensor.")

    gc.collect()
    log_ram(f"Sensor {sid} End (Post-GC)")

print("\nDone. All CSV exported successfully.")
log_ram("Script End")