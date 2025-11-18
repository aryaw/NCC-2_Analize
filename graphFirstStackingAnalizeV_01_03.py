"""
graphFirstStackingAnalizeV_01_3.py
- adaptive traffic mode detection (sporadic / periodic / simultaneous)
- adaptive grouping threshold G
- graph features extraction
- stacking model (RF + ET + HGB)
- auto threshold
- C&C detection rule
- per sensor inference + 3D graph
"""

import os
import gc
import re
import duckdb
import pandas as pd
import numpy as np
import psutil
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score
)

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelmin
from scipy.stats import skew

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
)

RANDOM_STATE = 42
MAX_ROWS_FOR_STACKING = 6_000_000
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

def detect_traffic_mode(df, src_col='SrcAddr', dst_col='DstAddr', time_col='StartTime'):
    gaps = []
    for (s, d), grp in df.groupby([src_col, dst_col]):
        times = grp[time_col].sort_values().values
        if len(times) <= 1:
            continue
        deltas = np.diff(times).astype('timedelta64[s]').astype(int)
        if len(deltas) > 50:
            deltas = np.random.choice(deltas, size=50, replace=False)
        gaps.extend(deltas.tolist())
        if len(gaps) > 20000:
            break

    if len(gaps) == 0:
        return "sporadic", {"reason": "no_gaps"}

    arr = np.array(gaps)
    frac_le_1 = np.mean(arr <= 1)
    small_frac = np.mean(arr <= 3)
    median = np.median(arr)
    sk = float(skew(arr))

    if frac_le_1 > 0.30 or small_frac > 0.45:
        mode = "simultaneous"
    elif median < 10 and sk > 1.0:
        mode = "periodic"
    else:
        mode = "sporadic"

    return mode, {
        "frac_le_1": float(frac_le_1),
        "small_frac": float(small_frac),
        "median_gap": float(median),
        "skew": sk,
        "gaps_sample": len(arr)
    }

def compute_timegap_threshold(df, src_col='SrcAddr', dst_col='DstAddr', time_col='StartTime',
                              enforce_mode=None):
    if enforce_mode is None:
        mode, _ = detect_traffic_mode(df)
    else:
        mode = enforce_mode

    if mode == "simultaneous":
        return 1

    gaps_seconds = []
    for (s, d), grp in df.groupby([src_col, dst_col]):
        times = grp[time_col].sort_values().values
        if len(times) <= 1:
            continue
        deltas = np.diff(times).astype('timedelta64[s]').astype(int)
        if len(deltas) > 500:
            deltas = np.random.choice(deltas, size=500, replace=False)
        gaps_seconds.extend(deltas.tolist())
        if len(gaps_seconds) > 200000:
            break

    if len(gaps_seconds) == 0:
        return 3600

    arr = np.array(gaps_seconds)
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    G = int(median + 2 * iqr)
    return int(np.clip(G, 1, 300))

def auto_threshold_tuner(stats_df, sensor_id=None, mode="sporadic"):
    stats = stats_df.copy()
    if stats.empty:
        return {
            "prob_threshold": 0.25,
            "degree_threshold": 10.0,
            "ratio_threshold": 0.6,
            "min_weighted_out": 5.0,
        }

    probs = np.array(stats["avg_prob"].fillna(0.0))
    degrees = np.array(stats["degree"].fillna(0.0))
    ratios = np.array(stats["out_ratio"].fillna(0.0))

    p75 = float(np.percentile(probs, 75)) if len(probs) else 0.2
    d70 = float(np.percentile(degrees, 70)) if len(degrees) else 5.0
    r65 = float(np.percentile(ratios, 65)) if len(ratios) else 0.5

    if mode == "simultaneous":
        prob_thr = max(0.30, p75)
        degree_thr = max(12.0, d70 * 1.5)
        ratio_thr = max(0.65, r65)
        min_wout = 8.0
    elif mode == "periodic":
        prob_thr = max(0.22, p75 * 0.9)
        degree_thr = max(8.0, d70)
        ratio_thr = max(0.55, r65)
        min_wout = 5.0
    else:
        prob_thr = max(0.15, p75 * 0.8)
        degree_thr = max(4.0, d70 * 0.8)
        ratio_thr = max(0.35, r65 * 0.9)
        min_wout = 3.0

    prob_thr = float(np.round(np.clip(prob_thr, 0.05, 0.95), 4))
    degree_thr = float(np.round(np.clip(degree_thr, 1.0, 500), 2))
    ratio_thr = float(np.round(np.clip(ratio_thr, 0.05, 0.99), 3))

    if sensor_id:
        print(f"[AutoTuneV2][Sensor {sensor_id} | {mode}] "
              f"prob>={prob_thr} | degree>={degree_thr} | out_ratio>={ratio_thr}")

    return {
        "prob_threshold": prob_thr,
        "degree_threshold": degree_thr,
        "ratio_threshold": ratio_thr,
        "min_weighted_out": min_wout,
    }

def is_cnc_node(addr, r, th):
    if r["avg_prob"] < th["prob_threshold"]:
        return False
    if r["degree"] < th["degree_threshold"]:
        return False
    if r["out_ratio"] < th["ratio_threshold"]:
        return False

    w_out = r["out_prob"] * r["out_ct"]
    if w_out < th["min_weighted_out"]:
        return False

    if not (r["out_ct"] >= 2 * max(1, r["in_ct"]) or r["out_ct"] >= 8):
        return False

    return True

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception:
    con = duckdb.connect()

# load only required columns + StartTime exists
query = f"""
SELECT StartTime, SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL
"""

print("[Load] Reading dataset...")
df = con.sql(query).df()
log_ram("After Load CSV")

if len(df) > MAX_ROWS_FOR_STACKING:
    print(f"[Sample] Raw rows {len(df):,} too large → sampling to {MAX_ROWS_FOR_STACKING:,}")
    df = df.sample(n=MAX_ROWS_FOR_STACKING, random_state=42).copy()

if df.empty:
    raise RuntimeError("No labeled data found in dataset.")

# optimize + binary label (use your helpers)
df = optimize_dataframe(df)
df = fast_label_to_binary(df)   # assumes adds/standardizes 'Label' column binary (0/1)
log_ram("After Optimize+Label")

print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

# enforce required non-null columns
df = df.dropna(subset=["StartTime","SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"]).copy()
log_ram("After DropNA")

# --- keep Dir mapping + basic encoding as before ---
dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# basic per-flow features (kept from your script)
df["ByteRatio"]      = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"]   = df["TotPkts"]  / (df["Dur"] + 0.1)
df["FlowIntensity"]  = df["SrcBytes"] / (df["TotBytes"] + 1)
df["PktByteRatio"]   = df["TotPkts"]  / (df["TotBytes"] + 1)
df["SrcByteRatio"]   = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = (df["sTos"] - df["dTos"]).abs()
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"]      = df["TotBytes"] / (df["Dur"] + 1)
log_ram("After Feature Eng")

global_mode, global_diag = detect_traffic_mode(df)
print("[GlobalMode] dataset mode:", global_mode, global_diag)
G = compute_timegap_threshold(df, enforce_mode=global_mode)
print(f"[Group] Threshold G (seconds): {G}")
log_ram("After Compute G")

def assign_activity_groups(df_in, G, src_col='SrcAddr', dst_col='DstAddr', time_col='StartTime'):
    df_local = df_in.sort_values(time_col).copy()
    df_local['GroupID'] = None
    for (s,d), grp in df_local.groupby([src_col,dst_col], sort=False):
        grp = grp.sort_values(time_col)
        current_gid = f"{s}->{d}_g0"
        counter = 0
        last_time = None
        for idx, row in grp.iterrows():
            t = row[time_col]
            if last_time is None:
                current_gid = f"{s}->{d}_g{counter}"
                counter += 1
                df_local.at[idx, 'GroupID'] = current_gid
                last_time = t
                continue
            gap = (t - last_time).total_seconds()
            if gap <= G:
                df_local.at[idx, 'GroupID'] = current_gid
            else:
                current_gid = f"{s}->{d}_g{counter}"
                counter += 1
                df_local.at[idx, 'GroupID'] = current_gid
            last_time = t
    return df_local

print("[Group] Assigning activity groups...")
df = assign_activity_groups(df, G)
log_ram("After Assign Groups")
print(f"[Group] Total groups: {df['GroupID'].nunique():,}")

def build_group_graph_edges(df_group):
    rows = []
    for gid, grp in df_group.groupby('GroupID'):
        gstart = grp['StartTime'].min()
        edge_counts = grp.groupby(['SrcAddr','DstAddr']).size().reset_index(name='weight')
        g_label = 'botnet' if (grp['Label'] == 1).any() else 'normal'
        for _, r in edge_counts.iterrows():
            rows.append({
                'GroupID': gid,
                'GroupStartTime': gstart,
                'SrcAddr': r['SrcAddr'],
                'DstAddr': r['DstAddr'],
                'Weight': int(r['weight']),
                'GroupLabel': g_label
            })
    return pd.DataFrame(rows)

print("[Graph] Building group edges (weighted)...")
group_edges = build_group_graph_edges(df)
log_ram("After Build Group Edges")
print(f"[Graph] Group edges rows: {len(group_edges):,}")

def extract_address_group_features(group_edges_df):
    records = []
    for gid, grp in group_edges_df.groupby('GroupID'):
        Gg = nx.DiGraph()
        for _, r in grp.iterrows():
            s = r['SrcAddr']; d = r['DstAddr']; w = int(r['Weight'])
            if Gg.has_edge(s,d):
                Gg[s][d]['weight'] += w
            else:
                Gg.add_edge(s,d, weight=w)
        gstart = grp['GroupStartTime'].iloc[0]
        g_label = grp['GroupLabel'].iloc[0]
        nodes = list(Gg.nodes())
        for n in nodes:
            out_deg = Gg.out_degree(n)
            in_deg  = Gg.in_degree(n)
            w_out = Gg.out_degree(n, weight='weight') or 0
            w_in  = Gg.in_degree(n, weight='weight') or 0
            records.append({
                'GroupID': gid,
                'GroupStartTime': gstart,
                'Address': n,
                'OutDegree': int(out_deg),
                'InDegree': int(in_deg),
                'WeightedOutDegree': float(w_out),
                'WeightedInDegree': float(w_in),
                'Label': g_label
            })
    return pd.DataFrame(records)

print("[Extract] Extracting address-group features...")
feat_ag = extract_address_group_features(group_edges)
log_ram("After Extract Features")
print(f"[Extract] Address-group feature rows: {len(feat_ag):,}")

if feat_ag.empty:
    raise RuntimeError("No address-group features extracted. Check grouping or input data.")

def split_80_20_per_class(df_features):
    df_bot = df_features[df_features['Label'] == 'botnet']
    df_norm = df_features[df_features['Label'] == 'normal']
    if len(df_bot) == 0 or len(df_norm) == 0:
        raise RuntimeError("Need both classes present in extracted features for split.")
    bot_tr, bot_te = train_test_split(df_bot, test_size=0.2, random_state=RANDOM_STATE)
    nor_tr, nor_te = train_test_split(df_norm, test_size=0.2, random_state=RANDOM_STATE)
    train = pd.concat([bot_tr, nor_tr]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test  = pd.concat([bot_te,  nor_te]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return train, test

print("[Split] Splitting features 80/20 per class...")
train_feat, test_feat = split_80_20_per_class(feat_ag)
log_ram("After Split")
print(f"[Split] Train rows: {len(train_feat):,} | Test rows: {len(test_feat):,}")

# features to train: degree features only (paper)
feat_cols = ['OutDegree','InDegree','WeightedOutDegree','WeightedInDegree']

# sample if too large
if len(train_feat) > MAX_ROWS_FOR_STACKING:
    print(f"[Sample] Too many rows for stacking train ({len(train_feat):,}) -> sampling {MAX_ROWS_FOR_STACKING:,}")
    train_feat_sample = train_feat.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE)
else:
    train_feat_sample = train_feat.copy()

X_train_graph = train_feat_sample[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
y_train_graph = (train_feat_sample['Label'] == 'botnet').astype(int)

X_test_graph = test_feat[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
y_test_graph = (test_feat['Label'] == 'botnet').astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_graph)
X_test_scaled  = scaler.transform(X_test_graph)
log_ram("After Scaling Graph Features")

print("\n[Train] Starting stacking model training on graph-based features (passthrough=False) ...")
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
            cv=2,
            n_jobs=1,
            verbose=1
        )

        stack.fit(X_train_scaled, y_train_graph)
        trained_model = stack
        print("[Train] Stacking model trained successfully on graph features.")
        log_ram("After Stacking Train Graph")

    except Exception as e:
        print(f"[WARN] Stacking failed: {type(e).__name__}: {e}")
        print("[Fallback] Using RandomForest instead...")
        trained_model = RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        ).fit(X_train_scaled, y_train_graph)
        print("[Fallback] RandomForest model trained on graph features.")
        log_ram("After Fallback RF Graph")

p_test = trained_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thr_roc = roc_curve(y_test_graph, p_test)

# Best threshold = Youden-like sqrt(tpr * (1 - fpr))
best_threshold = thr_roc[np.argmax(np.sqrt(tpr * (1 - fpr)))]
y_pred_test = (p_test >= best_threshold).astype(int)
log_ram("After Eval Graph")

print("\n# Graph-based Model Evaluation (per Address-Group rows)")
print("Best threshold:", round(float(best_threshold), 4))
print("Accuracy:", round(float((y_pred_test == y_test_graph).mean() * 100), 2), "%")
print("Precision:", precision_score(y_test_graph, y_pred_test, zero_division=0))
print("Recall:",    recall_score(y_test_graph, y_pred_test, zero_division=0))
print("F1:",        f1_score(y_test_graph, y_pred_test, zero_division=0))
print("ROC-AUC:",   roc_auc_score(y_test_graph, p_test))

X_all_graph = feat_ag[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
X_all_scaled = scaler.transform(X_all_graph)

feat_ag['PredictedProb'] = trained_model.predict_proba(X_all_scaled)[:,1]
feat_ag['PredictedLabel'] = (feat_ag['PredictedProb'] >= best_threshold).astype(int)
log_ram("After Full Inference Graph")
gc.collect()

addr_avg = feat_ag.groupby('Address')['PredictedProb'].agg(
    ['count','mean','median','max']
).rename(columns={
    'count':'group_ct',
    'mean':'addr_prob_mean',
    'median':'addr_prob_median',
    'max':'addr_prob_max'
}).reset_index()

# Use robust median instead of mean (reduces FP in NCC-2)
addr_prob_map = dict(zip(addr_avg['Address'], addr_avg['addr_prob_median']))

# attach addr_prob to original flow-level df for per-sensor aggregation
src_prob = df['SrcAddr'].map(addr_prob_map)
dst_prob = df['DstAddr'].map(addr_prob_map)

# More stable: maximum between src/dst probabilities
df['AddrProb'] = np.maximum(src_prob.fillna(0), dst_prob.fillna(0))

log_ram("After AddrProb Merge")


detected_summary = []
print("\n[Mode] Global traffic mode =", global_mode)

for sid in sorted(df["SensorId"].unique()):
    print(f"=== [Sensor {sid}] Auto C&C Detection (Graph-based) ===")
    log_ram(f"Sensor {sid} Start")

    df_s = df[df["SensorId"] == sid].copy()
    if df_s.empty:
        print("[Info] Sensor empty, skipping...")
        continue

    # Compute stats per address inside this sensor
    agg_in = df_s.groupby("DstAddr")["AddrProb"].agg(["count","mean"]).rename(
        columns={"count":"in_ct","mean":"in_prob"}
    )
    agg_out = df_s.groupby("SrcAddr")["AddrProb"].agg(["count","mean"]).rename(
        columns={"count":"out_ct","mean":"out_prob"}
    )

    stats = agg_in.join(agg_out, how="outer").fillna(0)
    stats["in_ratio"]  = stats["in_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["out_ratio"] = stats["out_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["avg_prob"]  = (stats["in_prob"] + stats["out_prob"]) / 2
    stats["degree"]    = stats["in_ct"] + stats["out_ct"]

    thresholds = auto_threshold_tuner(stats, sensor_id=sid, mode=global_mode)
    prob_thr = thresholds["prob_threshold"]
    degree_thr = thresholds["degree_threshold"]
    ratio_thr = thresholds["ratio_threshold"]
    min_wout = thresholds["min_weighted_out"]

    print(f"[Thresholds] prob>={prob_thr} | degree>={degree_thr} | ratio>={ratio_thr} | w_out>={min_wout}")

    # per-node classification
    node_roles = {}
    for n, r in stats.iterrows():
        if is_cnc_node(n, r, thresholds):
            node_roles[n] = "C&C"
        else:
            node_roles[n] = "Normal"

    cc_nodes = [n for n, role in node_roles.items() if role == "C&C"]
    print(f"[Detected] {len(cc_nodes)} potential C&C nodes in sensor {sid}")

    if len(cc_nodes) == 0:
        print("[Info] No C&C nodes detected in this sensor.")
        gc.collect()
        log_ram(f"Sensor {sid} End (no C&C)")
        continue

    cnc_df = stats.loc[cc_nodes].copy()
    cnc_df["SensorId"] = sid
    cnc_df["cnc_score"] = cnc_df["avg_prob"] * (
        1 + cnc_df["in_ratio"] + cnc_df["out_ratio"]
    ) * np.log1p(cnc_df["degree"])

    cnc_df = cnc_df.sort_values("cnc_score", ascending=False)
    detected_summary.append(cnc_df)

    log_ram(f"Sensor {sid} After C&C Score Computation")

    # Get all addresses that appear in flows of this sensor
    addrs_in_sensor = set(df_s['SrcAddr']).union(set(df_s['DstAddr']))

    # Select group-edges that involve these addresses
    ge_s = group_edges[
        (group_edges['SrcAddr'].isin(addrs_in_sensor)) |
        (group_edges['DstAddr'].isin(addrs_in_sensor))
    ].copy()

    if ge_s.empty:
        print("[Info] No group-edges for this sensor -> skip graph drawing")
        gc.collect()
        log_ram(f"Sensor {sid} End (no edges)")
        continue

    # Collapse weights across groups for same (src, dst)
    agg_edges = ge_s.groupby(['SrcAddr','DstAddr'])['Weight'].sum().reset_index()

    # Build DiGraph
    G_full = nx.from_pandas_edgelist(
        agg_edges,
        source='SrcAddr',
        target='DstAddr',
        edge_attr='Weight',
        create_using=nx.DiGraph()
    )

    cnc_neighbors = set()
    for cnc in cc_nodes:
        if cnc in G_full:
            cnc_neighbors.update(G_full.predecessors(cnc))
            cnc_neighbors.update(G_full.successors(cnc))

    nodes_keep = set(cc_nodes) | cnc_neighbors

    # Keep only nodes that exist in the graph
    nodes_keep = nodes_keep.intersection(G_full.nodes())

    # If still empty, fallback to top-degree nodes
    if not nodes_keep:
        degree_sorted = sorted(G_full.degree(), key=lambda x: x[1], reverse=True)
        fallback_cnt = min(40, len(degree_sorted))
        nodes_keep = set([n for n,_ in degree_sorted[:fallback_cnt]])
        print(f"[Fallback] Using {fallback_cnt} top-degree nodes.")

    available_gb = psutil.virtual_memory().available / (1024 ** 3)

    if available_gb < 4:
        MAX_NODES, MAX_EDGES = 150, 350
    elif available_gb < 8:
        MAX_NODES, MAX_EDGES = 250, 500
    else:
        MAX_NODES, MAX_EDGES = 450, 800

    # If too many nodes, sample only a subset
    if len(nodes_keep) > MAX_NODES:
        normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
        keep_norm = min(MAX_NODES - len(cc_nodes), len(normal_nodes))

        if keep_norm > 0:
            sample_norm = np.random.choice(
                normal_nodes, size=keep_norm, replace=False
            )
            nodes_keep = set(cc_nodes) | set(sample_norm)
        else:
            nodes_keep = set(cc_nodes)

    # Subgraph with selected nodes
    G = G_full.subgraph(nodes_keep).copy()

    all_edges = list(G.edges(data=True))

    if len(all_edges) > MAX_EDGES:
        print(f"[PruneEdges] {len(all_edges)} -> {MAX_EDGES}")

        np.random.seed(RANDOM_STATE)
        chosen_idx = np.random.choice(
            len(all_edges), size=MAX_EDGES, replace=False
        )
        chosen_edges = [all_edges[i] for i in chosen_idx]

        # Rebuild graph with chosen edges only
        G_temp = nx.DiGraph()
        for u, v, attr in chosen_edges:
            G_temp.add_edge(u, v, **attr)
        G = G_temp

    print(f"[PlotPrep] Sensor {sid}: {len(G.nodes())} nodes, {len(G.edges())} edges")

    node_color = []
    node_size  = []
    for n in G.nodes():
        if n in cc_nodes:
            node_color.append("red")
            node_size.append(15)
        else:
            node_color.append("blue")
            node_size.append(6)

    print("[Layout] Computing 3D layout (spring_layout) ...")
    pos = nx.spring_layout(
        G,
        dim=3,
        seed=RANDOM_STATE,
        iterations=60
    )

    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]
    z_nodes = [pos[k][2] for k in G.nodes()]

    # Edge coordinates for 3D plot
    edge_x, edge_y, edge_z = [], [], []
    for (u, v) in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    print("[Plot] Building 3D graph...")
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
            title=f"Sensor {sid} – 3D Degree Graph (Group-based, {len(G.nodes())} nodes)",
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            annotations=[
                dict(
                    text="Red = C&C Node | Blue = Normal Node (sampled)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=-0.05
                )
            ]
        )
    )

    # save graph
    graph_path = os.path.join(
        output_dir,
        f"Sensor{sid}_3DGraph_GroupDegree_{fileTimeStamp}.html"
    )
    fig.write_html(graph_path)
    print(f"[Plot] 3D graph saved: {graph_path}")
    log_ram(f"Sensor {sid} After Plot")

    # Cleanup before next sensor
    gc.collect()
    log_ram(f"Sensor {sid} End (Post-GC)")

print(" ALL SENSORS COMPLETED SUCCESSFULLY ")

# Merge all detected summaries (if any)
if len(detected_summary) > 0:
    all_cnc = pd.concat(detected_summary, ignore_index=True)
    cnc_summary_path = os.path.join(
        outputdata_dir,
        f"CNC_Detection_Summary_{fileDataTimeStamp}.csv"
    )
    all_cnc.to_csv(cnc_summary_path, index=False)
    print(f"[Summary] C&C Detection Summary saved -> {cnc_summary_path}")
else:
    print("[Summary] No C&C nodes detected across all sensors.")

log_ram("Script End")

print("\n[INFO] graphFirstStackingAnalizeV_01_3.py finished successfully.")
print(f"[INFO] RAM usage log saved -> {memory_log_path}")
