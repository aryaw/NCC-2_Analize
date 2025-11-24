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
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# Plotly for exports
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

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

# -------------------- helper --------------------
memory_log_path = os.path.join(output_dir, f"memory_trace_{fileTimeStamp}.csv")
# create header
with open(memory_log_path, "w") as f:
    f.write("timestamp,tag,current_mb,vms_mb\n")

def log_ram(tag=""):
    """Print and record current + virtual memory usage (MB)"""
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024)
    vms = getattr(process.memory_info(), "vms", 0) / (1024 * 1024)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[RAM] {tag:<25} Current={rss:8.2f} MB | VMS={vms:8.2f} MB")

    # Append to CSV
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

if df.empty:
    raise RuntimeError("No labeled data found in dataset.")

df = optimize_dataframe(df)
df = fast_label_to_binary(df)
log_ram("After Optimize+Label")

print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

# delete row contains NaN
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"]).copy()
log_ram("After DropNA")

dir_map_num = {"->": 1, "<-": -1, "<->": 0}

# if Dir missing, usually considered “forward flow” or standard direction >>> set as ->
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")

# map Dir to number
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

# set Proto & State as number, ML (Tree Based Model) can't process text, only numbers
for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# creating new features from the original features so that the machine learning model can detect botnets with higher accuracy
df["ByteRatio"] = df["TotBytes"] / (df["TotPkts"] + 1) # byte per packet, botnet had samll packet

# packet per second
# Botnet behavior:
# DDoS > very high rate
# Beaconing > short burst, low rate
# Scanning > fast packets but short duration
df["DurationRate"]   = df["TotPkts"]  / (df["Dur"] + 0.1)

# How many bytes come from source
# Bot (client) > usually more outbound traffic
# C&C server > usually more outbound, small inbound traffic
# Normal host > relatively balanced
df["FlowIntensity"]  = df["SrcBytes"] / (df["TotBytes"] + 1)

# Get ratio
# Some botnets:
# Send many small packets > high ratio
# Send few large packets > low ratio
df["PktByteRatio"]   = df["TotPkts"]  / (df["TotBytes"] + 1)

# Get how dominant traffic is from the source
# Bots > typically send outbound traffic > high ratio
# DDoS victims > small srcBytes > low ratio
df["SrcByteRatio"]   = df["SrcBytes"] / (df["TotBytes"] + 1)

# Type of Service / Quality of Service > packet prio
# Botnet traffic often does not follow normal priorities (ToS & QoS mismatch)
# Normal traffic is usually symmetrical/homogeneous
df["TrafficBalance"] = (df["sTos"] - df["dTos"]).abs()

# Bot scanning/automated traffic > very short duration per packet
# Normal human users > higher average
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)

# Byte/second
# DDoS UDP/TCP flood > very high intensity
# Small beaconing > low intensity
df["Intensity"] = df["TotBytes"] / (df["Dur"] + 1)
log_ram("After Feature Eng")

# select field in df
features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity", "PktByteRatio",
    "SrcByteRatio", "TrafficBalance", "DurationPerPkt", "Intensity"
]

# replace positif infiiny & negative infinity with NaN, an replace NaN to 0
X_full = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_full = df["Label"].astype(int) #if not int, cause error
log_ram("After Feature Select")

# reduce if df to big, 16gb memory bro!
if len(df) > MAX_ROWS_FOR_STACKING:
    print(f"[Sample] Dataset too large ({len(df):,}), using {MAX_ROWS_FOR_STACKING:,} for model training...")
    # get random sampling, dataset to big!
    df_sample = df.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE)

    # replace positif infiiny & negative infinity with NaN, an replace NaN to 0
    X = df_sample[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_sample["Label"].astype(int) #if not int, cause error
else:
    X, y = X_full, y_full
log_ram("After Sampling")

# X_train > training features
# X_test > testing features
# y_train > training labels
# y_test > testing labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
log_ram("After Train/Test Split")

# Logistic Regression is a linear model that is sensitive to the scale of features > use StandardScaler
scaler = StandardScaler()
# normalize scale train features
X_train_scaled = scaler.fit_transform(X_train)

# normalize scale test feature
# dont fit it, fit = learn from data
X_test_scaled  = scaler.transform(X_test)
log_ram("After Scaling")

# train
trained_model = None
print("\n[Train] Starting model training (Stacking, passthrough=False)...")

# RandomForest > stable, robust against noise, good for generalization
# ExtraTrees > more random, faster, and increases diversity
# HistGradientBoosting > very powerful for small/subtle patterns, suitable for large datasets
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

        # Logistic Regression meta-model for combining probabilities from multiple modes
        meta = LogisticRegression(solver="lbfgs", max_iter=1000)
        stack = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            stack_method="predict_proba",
            passthrough=False,
            cv=2,
            n_jobs=1,
            verbose=1
        )

        # RF_prob  ─┐
        # ET_prob  ─┼──➤ Logistic Regression ➤ Final Prediction
        # HGB_prob ─┘
        # stack becomes a fully trained model
        stack.fit(X_train_scaled, y_train)
        trained_model = stack
        print("[Train] Stacking model trained successfully.")
        log_ram("After Stacking Train")

    except Exception as e:
        print(f"[WARN] Stacking failed: {type(e).__name__}: {e}")
        print("[Fallback] Using RandomForest instead...")
        trained_model = RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        ).fit(X_train_scaled, y_train)
        print("[Fallback] RandomForest model trained.")
        log_ram("After Fallback RF")

# evaluation--

# predict probability for the test set
p_test = trained_model.predict_proba(X_test_scaled)[:, 1]

# calculate the ROC curve
fpr, tpr, thr_roc = roc_curve(y_test, p_test)

# find the best threshold
# selecting the best threshold for binary classification based on ROC curve
# The ROC curve is used because it provides an objective, fair way to select the best threshold, especially in cases such as class imbalance
best_threshold = thr_roc[np.argmax(np.sqrt(tpr * (1 - fpr)))]
# because the default threshold = 0.5 is not suitable for unbalanced datasets
# if using a threshold of 0.5, the model may allow botnets to pass through or trigger too many false alerts > mostly blue

# changes the probability of the model result to a 0/1 label based on the best threshold
y_pred_test = (p_test >= best_threshold).astype(int)
log_ram("After Eval")

print("\n# Global Model Evaluation")
print("Best threshold:", round(float(best_threshold), 4))
print("Accuracy:", round(float((y_pred_test == y_test).mean() * 100), 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:",    recall_score(y_test, y_pred_test))
print("F1:",        f1_score(y_test, y_pred_test))
print("ROC-AUC:",   roc_auc_score(y_test, p_test))

# full inference--

# convert all X_full features to standard scale (mean=0, std=1) > consistent with training, no refitting allowed
X_all_scaled = scaler.transform(X_full)

# probability prediction for all data
# rowValue nearby 0 >>> normal
# rowValue nearby 1 >>> bot
df["PredictedProb"] = trained_model.predict_proba(X_all_scaled)[:, 1]

# label prediction (0 = normal, 1 = botnet)
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)

# ram cleanup
del X_all_scaled, X_full, y_full
gc.collect()
log_ram("After Full Inference")

# per-sensor C&C detection
detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== [Sensor {sid}] Auto C&C Detection")
    log_ram(f"Sensor {sid} Start")

    df_s = df[df["SensorId"] == sid].copy()

    # inbound traffic per node
    agg_in = df_s.groupby("DstAddr")["PredictedProb"].agg(["count", "mean"]) \
                 .rename(columns={"count": "in_ct", "mean": "in_prob"})
    
    # outbound traffic per node
    agg_out = df_s.groupby("SrcAddr")["PredictedProb"].agg(["count", "mean"]) \
                  .rename(columns={"count": "out_ct", "mean": "out_prob"})
    
    # combine inbound & outbound statistics
    stats = agg_in.join(agg_out, how="outer").fillna(0)

    # high in_ratio > frequently receives traffic (server-like)
    stats["in_ratio"] = stats["in_ct"]  / (stats["in_ct"] + stats["out_ct"] + 1e-9) # 1e-9 prevents division by zero
    
    # high out_ratio > frequently sends traffic (client-like bot)
    stats["out_ratio"] = stats["out_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9) # 1e-9 prevents division by zero

    # degree = total number of connections (edges) that a node (IP) has in the network
    # degree is an important measure in graph analysis
    # C&C servers often have high degree, bot clients have low-medium degree
    stats["avg_prob"] = (stats["in_prob"] + stats["out_prob"]) / 2 # get average prob, by taking the average, we get a general idea of ​​how “malicious” the node is overall
    
    # degree = total number of connections (edges) that a node (IP) has in the network
    stats["degree"] = stats["in_ct"] + stats["out_ct"]

    node_roles = {}
    for n, r in stats.iterrows():
        if (r["avg_prob"] > 0.70) and (r["degree"] > 110) and (r["out_ratio"] > 0.70):
            node_roles[n] = "C&C"
        else:
            node_roles[n] = "Normal"

    # map as [], [IP:Role]
    cc_nodes = [n for n, role in node_roles.items() if role == "C&C"]
    print(f"[Detected] {len(cc_nodes)} potential C&C nodes")

    if cc_nodes:
        cnc_df = stats.loc[cc_nodes].copy()
        cnc_df["SensorId"] = sid
        
        cnc_df["cnc_score"] = cnc_df["avg_prob"] * (1 + cnc_df["in_ratio"] + cnc_df["out_ratio"]) * np.log1p(cnc_df["degree"])

        cnc_df = cnc_df.sort_values("cnc_score", ascending=False)
        detected_summary.append(cnc_df)
        log_ram(f"Sensor {sid} After C&C Score")

        print("[Plot] Generating 3D interactive network graph")

        # Dynamically set limits based on available RAM
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 4:
            MAX_NODES, MAX_EDGES = 200, 400
        elif available_gb < 8:
            MAX_NODES, MAX_EDGES = 300, 600
        else:
            MAX_NODES, MAX_EDGES = 500, 800

        print(f"[Info] Available RAM: {available_gb:.2f} GB -> Using limits: "
            f"{MAX_NODES} nodes, {MAX_EDGES} edges")
        log_ram(f"Sensor {sid} Graph Limits: {MAX_NODES} nodes, {MAX_EDGES} edges")

        # build directed graph from flows
        G_full = nx.from_pandas_edgelist(
            df_s, source="SrcAddr", target="DstAddr", create_using=nx.DiGraph()
        )

        # prioritize: all C&C nodes + their neighbors
        cnc_neighbors = set()
        for cnc in cc_nodes:
            if cnc in G_full:
                cnc_neighbors.update(G_full.predecessors(cnc))
                cnc_neighbors.update(G_full.successors(cnc))

        nodes_keep = set(cc_nodes) | cnc_neighbors
        if len(nodes_keep) > MAX_NODES:
            
            # sample neighbors to limit total nodes
            normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
            sample_normal = np.random.choice(normal_nodes, size=min(MAX_NODES, len(normal_nodes)), replace=False)
            nodes_keep = set(cc_nodes) | set(sample_normal)

        # subgraph for plotting
        G = G_full.subgraph(nodes_keep).copy()
        del G_full

        # limit edges
        all_edges = list(G.edges())
        if len(all_edges) > MAX_EDGES:
            np.random.seed(RANDOM_STATE)
            edges_sample = np.random.choice(len(all_edges), size=MAX_EDGES, replace=False)
            edges_keep = [all_edges[i] for i in edges_sample]
            G = nx.DiGraph(edges_keep)

        print(f"[Plot] Drawing {len(G.nodes())} nodes and {len(G.edges())} edges")

        node_color = []
        node_size = []
        for n in G.nodes():
            if n in cc_nodes:
                node_color.append("red")
                node_size.append(10)
            else:
                node_color.append("blue")
                node_size.append(4)

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
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.9
            ),
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
                    xref="paper", yref="paper",
                    x=0, y=-0.05
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

print("\nDone. Memory-safe stacking + per-sensor 3D network visualization complete.")
log_ram("Script End")
print(f"[Info] RAM usage log saved -> {memory_log_path}")
