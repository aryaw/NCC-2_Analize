# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# GraphSAGE-based
# CPU-optimized, preserves original ingestion, features, scoring, and plotting.

import os
import gc
import re
import duckdb
import pandas as pd
import numpy as np
import psutil
from datetime import datetime
import sys
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

import plotly.graph_objects as go
import networkx as nx

try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits=1):
        yield

# internal utilities (assumed available)
from libInternal import (
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
    detect_cnc_from_label
)

import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader

RANDOM_STATE = 42
MAX_ROWS_FOR_STACKING = 13_000_000
SAFE_THREADS = "1"

os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
    "NUMEXPR_NUM_THREADS": SAFE_THREADS,
    "MKL_THREADING_LAYER": "GNU",
    "JOBLIB_TEMP_FOLDER": "/tmp",
})

# GNN config (GraphSAGE)
GNN_CONFIG = {
    "model": "graphsage",
    # "hidden_dim": 64,
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    # "epochs": 30,
    "epochs": 20,
    "batch_size": 4,
    "patience": 6
}

fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()

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

edge_weights = df.groupby(["SrcAddr", "DstAddr"]).size().reset_index(name="EdgeWeight")
df = df.merge(edge_weights, on=["SrcAddr", "DstAddr"], how="left")
df["EdgeWeight"] = df["EdgeWeight"].fillna(1).astype(int)

src_total = df.groupby("SrcAddr")["EdgeWeight"].sum().rename("SrcTotalWeight")
dst_total = df.groupby("DstAddr")["EdgeWeight"].sum().rename("DstTotalWeight")

df = df.merge(src_total, left_on="SrcAddr", right_index=True, how="left")
df = df.merge(dst_total, left_on="DstAddr", right_index=True, how="left")
df["SrcTotalWeight"] = df["SrcTotalWeight"].fillna(0)
df["DstTotalWeight"] = df["DstTotalWeight"].fillna(0)

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
    "SrcByteRatio","TrafficBalance","DurationPerPkt","Intensity",
    "EdgeWeight","SrcTotalWeight","DstTotalWeight"
]

# feature matrix for fallback stacking (if ever needed)
X_full = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_full = df["Label"].astype(int)
log_ram("After Feature Select")

# sampling logic to limit training size
if len(df) > MAX_ROWS_FOR_STACKING:
    df_used = df.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE).copy()
    print(f"[Data] Data too big, {MAX_ROWS_FOR_STACKING} used")
else:
    df_used = df.copy()
    print(f"[Data] All data, {len(df)} used")
log_ram("After Sampling (df_used)")

USE_PYG = True
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SAGEConv
except Exception as e:
    USE_PYG = False
    _PYG_ERR = e

def build_graph_data_from_df(df_sensor, features_list):
    src_nodes = df_sensor["SrcAddr"].unique().tolist()
    dst_nodes = df_sensor["DstAddr"].unique().tolist()
    all_nodes = list(dict.fromkeys(src_nodes + dst_nodes))
    if len(all_nodes) == 0:
        return None
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    # Edge list
    ew = df_sensor.groupby(["SrcAddr", "DstAddr"])["EdgeWeight"].sum().reset_index()
    if ew.empty:
        edge_index = np.empty((2,0), dtype=int)
        edge_weight = np.empty((0,), dtype=float)
    else:
        edge_index = np.array([[node_to_idx[r["SrcAddr"]], node_to_idx[r["DstAddr"]]] for _, r in ew.iterrows()]).T
        edge_weight = np.array([float(r["EdgeWeight"]) for _, r in ew.iterrows()], dtype=float)

    # node level aggregation for features_list
    # compute mean and count when node acts as src and as dst.
    agg_src = df_sensor.groupby("SrcAddr")[features_list].agg(["mean","count"])
    agg_src.columns = [f"src_{f}_{agg}" for f, agg in agg_src.columns]
    agg_dst = df_sensor.groupby("DstAddr")[features_list].agg(["mean","count"])
    agg_dst.columns = [f"dst_{f}_{agg}" for f, agg in agg_dst.columns]

    nf = pd.DataFrame(index=all_nodes)
    if not agg_src.empty:
        agg_src = agg_src.rename_axis(index="node")
        nf = nf.join(agg_src, how="left")
    if not agg_dst.empty:
        agg_dst = agg_dst.rename_axis(index="node")
        nf = nf.join(agg_dst, how="left")
    nf = nf.fillna(0.0)

    # additional scalar features
    in_count = df_sensor.groupby("DstAddr").size().rename("in_count")
    out_count = df_sensor.groupby("SrcAddr").size().rename("out_count")
    nf = nf.join(in_count.rename("in_count"), how="left")
    nf = nf.join(out_count.rename("out_count"), how="left")
    nf = nf.fillna(0.0)

    # aggregate src/dst total weights
    src_w = df_sensor.groupby("SrcAddr")["SrcTotalWeight"].mean().rename("src_total_weight_agg")
    dst_w = df_sensor.groupby("DstAddr")["DstTotalWeight"].mean().rename("dst_total_weight_agg")
    nf = nf.join(src_w, how="left").join(dst_w, how="left")
    nf = nf.fillna(0.0)

    # label mapping: node is CNC if it appears with LabelCNC==1 in either src or dst
    labels = pd.Series(0, index=all_nodes, dtype=int)
    if "LabelCNC" in df_sensor.columns:
        cnc_src = set(df_sensor[df_sensor["LabelCNC"] == 1]["SrcAddr"].unique().tolist())
        cnc_dst = set(df_sensor[df_sensor["LabelCNC"] == 1]["DstAddr"].unique().tolist())
        cnc_nodes = cnc_src.union(cnc_dst)
        for n in cnc_nodes:
            if n in labels.index:
                labels.loc[n] = 1

    # build tensors if PyG available
    if USE_PYG:
        import torch
        x = torch.tensor(nf.values, dtype=torch.float)
        y = torch.tensor(labels.values, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long) if edge_index.size != 0 else torch.empty((2,0), dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight.size != 0 else torch.empty((0,), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.edge_weight = edge_weight
        data.node_mapping = all_nodes
        return data
    else:
        return None

# Build graphs per sensor
print("[GNN] Building sensor graphs...")
sensor_graphs = []
sids = sorted(df_used["SensorId"].unique())
for sid in sids:
    df_s = df_used[df_used["SensorId"] == sid].copy()
    if df_s.empty:
        continue
    data = build_graph_data_from_df(df_s, features)
    if data is None:
        continue
    data.sensor_id = sid
    sensor_graphs.append(data)

print(f"[GNN] Built {len(sensor_graphs)} sensor graphs")
log_ram("After building sensor graphs")

if not USE_PYG or len(sensor_graphs) == 0:
    print("[GNN] PyG not available or no graphs built. Exiting GNN pipeline.")
    if not USE_PYG:
        print("[GNN] Import error:", _PYG_ERR)
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[GNN] Running on device:", device)

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin(x).squeeze(-1)
        return logits

# instantiate model
sample_in_dim = sensor_graphs[0].num_node_features
model = GraphSAGENet(
    in_channels=sample_in_dim,
    hidden_channels=GNN_CONFIG["hidden_dim"],
    num_layers=GNN_CONFIG["num_layers"],
    dropout=GNN_CONFIG["dropout"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=GNN_CONFIG["lr"], weight_decay=GNN_CONFIG["weight_decay"])
loss_fn = torch.nn.BCEWithLogitsLoss()

# dataLoader over graphs
loader = DataLoader(sensor_graphs, batch_size=GNN_CONFIG["batch_size"], shuffle=True)

best_loss = float("inf")
best_state = None
patience_cnt = 0
print("[GNN] Training started...")
for epoch in range(1, GNN_CONFIG["epochs"] + 1):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)  # node-level logits
        
        y = batch.y.float()
        mask = (y == 0) | (y == 1)   # in our construction all present nodes are labeled 0/1
        
        if mask.sum() == 0:
            continue
        loss = loss_fn(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(mask.sum().item())
        total_nodes += int(mask.sum().item())
    avg_loss = total_loss / max(1, total_nodes)
    print(f"[GNN][Epoch {epoch}/{GNN_CONFIG['epochs']}] avg_node_loss={avg_loss:.6f}")
    # early stopping
    if avg_loss + 1e-6 < best_loss:
        best_loss = avg_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= GNN_CONFIG["patience"]:
            print("[GNN] Early stopping triggered.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print("[GNN] Training finished.")
log_ram("After GNN Train")
gc.collect()

model.eval()
all_pred_rows = []
with torch.no_grad():
    for data in sensor_graphs:
        data = data.to(device)
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()
        node_ips = data.node_mapping
        for ip, p in zip(node_ips, probs):
            all_pred_rows.append((data.sensor_id, ip, float(p)))

preds_df = pd.DataFrame(all_pred_rows, columns=["SensorId", "IP", "PredictedProb_Node"])

# split into src/dst preds for merging
preds_src = preds_df.rename(columns={"IP": "SrcAddr", "PredictedProb_Node": "Src_PredProb"})
preds_dst = preds_df.rename(columns={"IP": "DstAddr", "PredictedProb_Node": "Dst_PredProb"})

# merge predictions into original df (use df, but ensure merging with SensorId)
df = df.merge(preds_src[["SensorId", "SrcAddr", "Src_PredProb"]], on=["SensorId", "SrcAddr"], how="left")
df = df.merge(preds_dst[["SensorId", "DstAddr", "Dst_PredProb"]], on=["SensorId", "DstAddr"], how="left")
df["Src_PredProb"] = df["Src_PredProb"].fillna(0.0)
df["Dst_PredProb"] = df["Dst_PredProb"].fillna(0.0)
df["PredictedProb"] = df[["Src_PredProb", "Dst_PredProb"]].max(axis=1)
# placeholder PredictedLabel (actual selection via per-sensor rules)
df["PredictedLabel"] = (df["PredictedProb"] >= 0.5).astype(int)

log_ram("After GNN Inference and merge")
gc.collect()

detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== Sensor {sid} ===")
    df_s = df[df["SensorId"] == sid].copy()

    # inbound/outbound (counts and mean predicted prob)
    agg_in  = df_s.groupby("DstAddr")["PredictedProb"].agg(["count","mean"])
    agg_out = df_s.groupby("SrcAddr")["PredictedProb"].agg(["count","mean"])

    agg_in.columns  = ["in_ct","in_prob"]
    agg_out.columns = ["out_ct","out_prob"]

    stats = agg_in.join(agg_out, how="outer").fillna(0)

    # add unique peer counts
    out_unique = df_s.groupby("SrcAddr")["DstAddr"].nunique().rename("out_unique_dests")
    in_unique  = df_s.groupby("DstAddr")["SrcAddr"].nunique().rename("in_unique_srcs")
    stats = stats.join(out_unique, how="left").join(in_unique, how="left").fillna(0)

    # aggregated weight features
    src_total_w = df_s.groupby("SrcAddr")["EdgeWeight"].sum().rename("src_total_weight")
    dst_total_w = df_s.groupby("DstAddr")["EdgeWeight"].sum().rename("dst_total_weight")
    stats = stats.join(src_total_w, how="left").join(dst_total_w, how="left").fillna(0)

    stats["degree"] = stats["in_ct"] + stats["out_ct"]
    stats["in_ratio"]  = stats["in_ct"]  / (stats["degree"] + 1e-9)
    stats["out_ratio"] = stats["out_ct"] / (stats["degree"] + 1e-9)

    # AUTO weighted CNC probability (sensor-adaptive)
    mal_out = stats["out_prob"].mean()
    mal_in  = stats["in_prob"].mean()
    dominance = mal_out / (mal_out + mal_in + 1e-9)

    w_out = 0.5 + 0.5 * dominance
    w_in  = 1.0 - w_out
    print(f"[Sensor {sid}] Auto C&C Weights: w_out={w_out:.3f}, w_in={w_in:.3f}")

    stats["cnc_prob"] = (stats["out_prob"] * w_out) + (stats["in_prob"] * w_in)

    stats["cnc_score"] = stats["cnc_prob"] * (
        1 + stats["out_ratio"] * 1.8 + stats["in_ratio"] * 0.8
    ) * np.log1p(stats["degree"]) * np.log1p(stats.get("out_unique_dests", 0) + 1) * np.log1p(stats.get("src_total_weight", 0) + 1)

    # top 5 candidates (sorted by cnc_score)
    top5_any = stats.sort_values("cnc_score", ascending=False).head(5)
    print(f"\n[Sensor {sid}] Top 5 C&C Candidates (regardless of threshold):")
    for idx, row in top5_any.iterrows():
        print(
            f"  - {idx} | score={row['cnc_score']:.4f} | "
            f"cnc_prob={row['cnc_prob']:.3f} | "
            f"out_ct={int(row['out_ct'])} | out_ratio={row['out_ratio']:.2f}"
        )

    auto_strict_thr = stats["cnc_prob"].mean() + 1.5 * stats["cnc_prob"].std()
    auto_strict_thr = min(auto_strict_thr, 0.95)
    auto_strict_thr = max(auto_strict_thr, 0.40)

    print(f"[Sensor {sid}] Auto strict CNC threshold = {auto_strict_thr:.3f}")

    strict_nodes = set()
    for n, r in stats.iterrows():
        if (r["cnc_prob"] > auto_strict_thr) and (r["out_ct"] > 120) and (r["out_ratio"] > 0.70):
            strict_nodes.add(n)

    percentile_cut = 95
    cutoff_score = max(0.0, np.percentile(stats["cnc_score"].values, percentile_cut))

    min_cnc_prob = 0.40
    min_out_ct = 20

    percentile_nodes = set(stats[
        (stats["cnc_score"] >= cutoff_score) &
        (stats["cnc_prob"] >= min_cnc_prob) &
        (stats["out_ct"] >= min_out_ct)
    ].index.tolist())

    real_cnc_nodes = df_s[df_s.get("LabelCNC", pd.Series(dtype=int)) == 1]["DstAddr"].unique().tolist() if "LabelCNC" in df_s.columns else []
    real_cnc_set = set(real_cnc_nodes)

    cc_nodes = sorted(set().union(strict_nodes, percentile_nodes, real_cnc_set))

    reasons = {}
    for n in cc_nodes:
        r = stats.loc[n] if n in stats.index else None
        reason_parts = []
        if n in strict_nodes:
            reason_parts.append("strict_rule")
        if n in percentile_nodes:
            reason_parts.append(f"score>=p{percentile_cut} (>= {cutoff_score:.4f})")
        if n in real_cnc_set:
            reason_parts.append("label_cnc")
        if r is not None:
            if r.get("out_ratio", 0) > 0.8:
                reason_parts.append("high_out_ratio")
            if r.get("out_ct", 0) > 200:
                reason_parts.append("very_high_out_ct")
            if r.get("src_total_weight", 0) > 500:
                reason_parts.append("high_src_total_weight")
        reasons[n] = ";".join(reason_parts) if reason_parts else "candidate"

    stats["Role"] = stats.index.map(lambda x: "C&C" if x in cc_nodes else "Normal")
    stats["Reason"] = stats.index.map(lambda x: reasons.get(x, ""))

    candidates_path = os.path.join(outputdata_dir, f"Sensor{sid}_CNC_Candidates_{fileDataTimeStamp}_gnn.csv")
    stats.sort_values("cnc_score", ascending=False).to_csv(candidates_path)
    print(f"[Export] Sensor {sid} candidates exported -> {candidates_path}")

    if len(cc_nodes) > 0:
        cnc_df = stats.loc[[n for n in cc_nodes if n in stats.index]].copy()
        cnc_df["SensorId"] = sid
        detected_summary.append(cnc_df)
        log_ram(f"Sensor {sid} After C&C Score")
        print(f"[Detect] Sensor {sid} flagged {len(cc_nodes)} nodes (strict={len(strict_nodes)}, percentile={len(percentile_nodes)}, ground_truth={len(real_cnc_set)})")

        # show top 5 flagged nodes for quick debug
        top5 = cnc_df.sort_values("cnc_score", ascending=False).head(5)
        for idx, row in top5.iterrows():
            print(
                f"  - {idx} score={row['cnc_score']:.4f} cnc_prob={row['cnc_prob']:.3f} "
                f"out_ct={int(row['out_ct'])} out_ratio={row['out_ratio']:.2f} reason={row['Reason']}"
            )

        # graph Generation (weighted edges, safe sampling & preserve node set)
        print("[Plot] Generating 3D interactive network graph")

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 4:
            MAX_NODES, MAX_EDGES = 200, 400
        elif available_gb < 8:
            MAX_NODES, MAX_EDGES = 300, 600
        else:
            MAX_NODES, MAX_EDGES = 500, 800

        log_ram(f"Sensor {sid} Graph Limits: {MAX_NODES} nodes, {MAX_EDGES} edges")

        # build weighted directed graph for this sensor using aggregated edge weights
        ew = df_s.groupby(["SrcAddr", "DstAddr"]).size().reset_index(name="weight")
        G_full = nx.from_pandas_edgelist(
            ew,
            source="SrcAddr",
            target="DstAddr",
            edge_attr="weight",
            create_using=nx.DiGraph()
        )

        # determine neighbor expansion: include predecessors and successors up to 1-hop
        cnc_neighbors = set()
        for cnc in cc_nodes:
            if cnc in G_full:
                cnc_neighbors.update(G_full.predecessors(cnc))
                cnc_neighbors.update(G_full.successors(cnc))

        nodes_keep = set(cc_nodes) | cnc_neighbors

        # if still too large, sample non-cc nodes by degree (prefer high-degree)
        if len(nodes_keep) > MAX_NODES:
            normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
            normal_nodes_scores = [(n, G_full.degree(n)) for n in normal_nodes]
            normal_nodes_scores.sort(key=lambda x: x[1], reverse=True)
            keep_normals = [n for n, _ in normal_nodes_scores[: max(0, MAX_NODES - len(cc_nodes))]]
            nodes_keep = set(cc_nodes) | set(keep_normals)

        # subgraph and preserve original graph connectivity among chosen nodes
        G = G_full.subgraph(nodes_keep).copy()
        del G_full

        # if edges exceed limit, prioritize heavy edges adjacent to cc_nodes first, then fill with remaining heavy edges
        all_edges = list(G.edges(data=True))
        if len(all_edges) > MAX_EDGES:
            prioritized = [e for e in all_edges if (e[0] in cc_nodes or e[1] in cc_nodes)]
            prioritized.sort(key=lambda x: x[2].get("weight", 1), reverse=True)
            remaining = [e for e in all_edges if e not in prioritized]
            remaining.sort(key=lambda x: x[2].get("weight", 1), reverse=True)
            keep_edges = []
            keep_edges.extend(prioritized)
            if len(keep_edges) < MAX_EDGES:
                needed = MAX_EDGES - len(keep_edges)
                keep_edges.extend(remaining[:needed])
            keep_edges = keep_edges[:MAX_EDGES]
            G = nx.DiGraph()
            G.add_nodes_from(nodes_keep)
            for u, v, d in keep_edges:
                G.add_edge(u, v, **(d if isinstance(d, dict) else {}))

        print(f"[Plot] Drawing {len(G.nodes())} nodes and {len(G.edges())} edges")
        pos = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=60)

        top4_nodes = []
        if 'cnc_df' in locals() and not cnc_df.empty:
            cnc_present = cnc_df[cnc_df.index.isin(G.nodes())]
            top4_nodes = cnc_present.sort_values("cnc_score", ascending=False).head(4).index.tolist()

        top4_set = set(top4_nodes)

        # build edge traces
        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            try:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
            except Exception:
                continue
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color="#B0B0B0", width=1.2),
            hoverinfo='none',
            name='edges'
        )

        x_nodes = [pos[n][0] for n in G.nodes()]
        y_nodes = [pos[n][1] for n in G.nodes()]
        z_nodes = [pos[n][2] for n in G.nodes()]

        node_color = []
        node_size = []
        node_text = []

        for n in G.nodes():
            if n in top4_set:
                node_color.append("#F11F1F")
                node_size.append(22)
                node_text.append(n)
            elif n in cc_nodes:
                node_color.append("#E4A41A")
                node_size.append(12)
                node_text.append("")
            else:
                node_color.append("#1F77B4")
                node_size.append(6)
                node_text.append("")

        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.92,
                line=dict(width=1.5, color="black")
            ),
            text=node_text,
            textfont=dict(size=14, color="white"),
            textposition="top center",
            hoverinfo='text',
            name='nodes'
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Sensor {sid} – GNN Network Graph (C&C Highlighted)",
                showlegend=False,
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(showbackground=False, visible=False),
                    yaxis=dict(showbackground=False, visible=False),
                    zaxis=dict(showbackground=False, visible=False)
                ),
                annotations=[dict(
                    text="Red = C&C | Blue = Normal | Label = Top-4 C&C",
                    showarrow=False,
                    xref="paper", yref="paper", x=0, y=-0.06,
                    font=dict(size=13, color="#AAAAAA")
                )]
            )
        )

        graph_path = os.path.join(output_dir, f"Sensor{sid}_3DGraphGNN_{fileTimeStamp}.html")
        fig.write_html(graph_path)
        print(f"[Plot] GNN graph saved -> {graph_path}")
        log_ram(f"Sensor {sid} After GNN Plot")

    else:
        print("[Info] No C&C nodes detected in this sensor (GNN stage).")

    gc.collect()
    log_ram(f"Sensor {sid} End (Post-GC)")

print("\nDone. All CSV exported successfully.")
log_ram("Script End")
