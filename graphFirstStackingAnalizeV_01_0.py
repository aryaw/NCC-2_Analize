#!/usr/bin/env python3
"""
graphFirstStackingAnalizeV_01_0_tuner.py
Pipeline: grouping -> graph features -> stacking -> auto-threshold tuning (KDE+Elbow+IForest) -> per-sensor detection + 3D graphs
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
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
)

# density / signal utils
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelmin

# Plotly for exports
import plotly.graph_objects as go
import networkx as nx

# Safer threading on CPU-only boxes
try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits=1):
        yield

# keep your internal helpers if present (setFileLocation, getConnection, etc.)
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

# -------------------- helper memory logging --------------------
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

def otsu_threshold(data):
    data = np.array([x for x in data if not np.isnan(x)])
    if len(data) < 3:
        return 0.5  # fallback

    hist, bin_edges = np.histogram(data, bins=50)
    total = data.size
    sum_total = np.dot(bin_edges[:-1], hist)

    sumB = 0.0
    wB = 0
    max_var = 0
    threshold = bin_edges[0]

    for i in range(len(hist)):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += bin_edges[i] * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = bin_edges[i]

    return threshold

# -------------------- Auto-threshold tuner --------------------
def auto_threshold_tuner(stats_df, sensor_id=None):
    stats = stats_df.copy()
    if stats.empty:
        return {"prob_threshold": 0.15, "degree_threshold": 5.0, "ratio_threshold": 0.5}

    probs = np.array(stats["avg_prob"].fillna(0.0))
    degrees = np.array(stats["degree"].fillna(0.0))
    ratios = np.array(stats["out_ratio"].fillna(0.0))

    # ---------- prob threshold via KDE valley + percentile ----------
    try:
        # scale bandwidth relative to spread
        bw = max(0.01, min(0.2, np.std(probs) if np.std(probs) > 0 else 0.05))
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(probs.reshape(-1, 1))
        prob_range = np.linspace(0.0, 1.0, 300)
        log_dens = kde.score_samples(prob_range.reshape(-1, 1))
        dens = np.exp(log_dens)
        mins = argrelmin(dens)[0]
        if len(mins) > 0:
            prob_thr_kde = float(prob_range[mins[0]])
        else:
            prob_thr_kde = float(np.percentile(probs, 70))
    except Exception:
        prob_thr_kde = float(np.percentile(probs, 70))

    prob_thr_pct = float(np.percentile(probs, 75))
    prob_thr = max(prob_thr_kde, prob_thr_pct, 0.10)
    #
    # clamp
    prob_thr = min(max(prob_thr, 0.05), 0.9)

    # ---------- degree threshold via elbow (biggest jump) ----------
    deg_sorted = np.sort(degrees)
    if len(deg_sorted) > 5:
        diff_deg = np.diff(deg_sorted)
        elbow_idx = int(np.argmax(diff_deg))
        degree_thr_elbow = float(deg_sorted[min(elbow_idx, len(deg_sorted)-1)])
    else:
        degree_thr_elbow = float(np.percentile(deg_sorted, 70) if len(deg_sorted)>0 else 5.0)
    degree_thr = max(3.0, degree_thr_elbow)

    # ---------- out_ratio threshold via percentile ----------
    ratio_thr = float(np.percentile(ratios, 65)) if len(ratios) > 0 else 0.5
    ratio_thr = max(0.35, ratio_thr)

    # ---------- Isolation Forest refinement ----------
    try:
        iso_input = np.vstack([probs, degrees, ratios]).T
        # if few samples, skip heavy IF
        if iso_input.shape[0] >= 8:
            cont = min(0.12, max(0.02, 6.0 / iso_input.shape[0]))  # adaptive contamination
            iso = IsolationForest(n_estimators=256, contamination=cont, random_state=RANDOM_STATE)
            preds = iso.fit_predict(iso_input)  # -1 anomaly
            mask_anom = (preds == -1)
            if mask_anom.sum() > 0:
                # get minimal values inside anomaly set to refine thresholds (conservative)
                prob_anom_min = float(np.min(probs[mask_anom]))
                deg_anom_min = float(np.min(degrees[mask_anom]))
                ratio_anom_min = float(np.min(ratios[mask_anom]))

                # refine thresholds if anomalies indicate lower cutoffs
                prob_thr = min(prob_thr, max(prob_anom_min, 0.08))
                degree_thr = min(degree_thr, max(deg_anom_min, 3.0))
                ratio_thr = min(ratio_thr, max(ratio_anom_min, 0.30))
    except Exception:
        pass

    # final safety clamps
    prob_thr = float(np.round(prob_thr, 4))
    degree_thr = float(np.round(max(1.0, degree_thr), 2))
    ratio_thr = float(np.round(min(max(ratio_thr, 0.2), 0.95), 3))

    if sensor_id is not None:
        print(f"[AutoTune][Sensor {sensor_id}] prob>={prob_thr} | degree>={degree_thr} | out_ratio>={ratio_thr}")

    return {"prob_threshold": prob_thr, "degree_threshold": degree_thr, "ratio_threshold": ratio_thr}


# -------------------- load connection --------------------
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

if df.empty:
    raise RuntimeError("No labeled data found in dataset.")

# optimize + binary label (use your helpers)
df = optimize_dataframe(df)
df = fast_label_to_binary(df)   # assumes adds/standardizes 'Label' column binary (0/1)
log_ram("After Optimize+Label")

print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

df = df.dropna(subset=["StartTime","SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"]).copy()
log_ram("After DropNA")

# --- keep Dir mapping + basic encoding as you had ---
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

# ------------------------------
# 3.1 Activity grouping: compute threshold G from time-gaps
# ------------------------------
def compute_timegap_threshold(df, src_col='SrcAddr', dst_col='DstAddr', time_col='StartTime'):
    gaps_seconds = []
    # group by src,dst
    for (s,d), grp in df.groupby([src_col, dst_col]):
        times = grp[time_col].sort_values().values
        if len(times) <= 1:
            continue
        deltas = np.diff(times).astype('timedelta64[s]').astype(int)
        gaps_seconds.extend(deltas.tolist())
    if len(gaps_seconds) == 0:
        return 3600  # fallback 1 hour
    arr = np.array(gaps_seconds)
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    G = int(median + 2 * iqr)
    return max(1, G)

print("[Group] Computing time-gap threshold G ...")
G = compute_timegap_threshold(df)
print(f"[Group] Threshold G (seconds): {G}")
log_ram("After Compute G")

# ------------------------------
# assign GroupID per (src,dst) using threshold G
# ------------------------------
def assign_activity_groups(df, G, src_col='SrcAddr', dst_col='DstAddr', time_col='StartTime'):
    df = df.sort_values(time_col).copy()
    df['GroupID'] = None
    for (s,d), grp in df.groupby([src_col,dst_col], sort=False):
        grp = grp.sort_values(time_col)
        current_gid = f"{s}->{d}_g0"
        counter = 0
        last_time = None
        for idx, row in grp.iterrows():
            t = row[time_col]
            if last_time is None:
                current_gid = f"{s}->{d}_g{counter}"
                counter += 1
                df.at[idx, 'GroupID'] = current_gid
                last_time = t
                continue
            gap = (t - last_time).total_seconds()
            if gap <= G:
                df.at[idx, 'GroupID'] = current_gid
            else:
                current_gid = f"{s}->{d}_g{counter}"
                counter += 1
                df.at[idx, 'GroupID'] = current_gid
            last_time = t
    return df

print("[Group] Assigning activity groups...")
df = assign_activity_groups(df, G)
log_ram("After Assign Groups")
print(f"[Group] Total groups: {df['GroupID'].nunique():,}")

# ------------------------------
# 3.2 - 3.4 Build group graphs and compute edge weights
# ------------------------------
def build_group_graph_edges(df_group):
    # returns dataframe of edges per group: GroupID, src, dst, weight, group_start_time, Label (if group has botnet)
    rows = []
    for gid, grp in df_group.groupby('GroupID'):
        # group start time
        gstart = grp['StartTime'].min()
        # edge weights: count occurrences of src->dst inside group
        edge_counts = grp.groupby(['SrcAddr','DstAddr']).size().reset_index(name='weight')
        # label per group: if any row has Label==1 then group is botnet
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

# ------------------------------
# 3.5 Feature extraction per Address-per-Group
# ------------------------------
def extract_address_group_features(group_edges_df):
    records = []
    # group by GroupID, compute directed graph metrics per group
    for gid, grp in group_edges_df.groupby('GroupID'):
        # build DiGraph for this group
        Gg = nx.DiGraph()
        for _, r in grp.iterrows():
            s = r['SrcAddr']; d = r['DstAddr']; w = int(r['Weight'])
            if Gg.has_edge(s,d):
                Gg[s][d]['weight'] += w
            else:
                Gg.add_edge(s,d, weight=w)
        # group start
        gstart = grp['GroupStartTime'].iloc[0]
        # label for addresses in group: if group labeled botnet -> all addresses that appear get label botnet if any row had botnet
        g_label = grp['GroupLabel'].iloc[0]
        # ensure nodes include any src/dst even if isolated (should exist)
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

# If no features extracted, abort
if feat_ag.empty:
    raise RuntimeError("No address-group features extracted. Check grouping or input data.")

# ------------------------------
# 3.6 Data splitting 80/20 per class (on feat_ag)
# ------------------------------
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

# If dataset too large, sample for stacking training (respect MAX_ROWS_FOR_STACKING)
if len(train_feat) > MAX_ROWS_FOR_STACKING:
    print(f"[Sample] Too many rows for stacking train ({len(train_feat):,}) -> sampling {MAX_ROWS_FOR_STACKING:,}")
    train_feat_sample = train_feat.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE)
else:
    train_feat_sample = train_feat.copy()

X_train_graph = train_feat_sample[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
y_train_graph = (train_feat_sample['Label'] == 'botnet').astype(int)

X_test_graph = test_feat[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
y_test_graph = (test_feat['Label'] == 'botnet').astype(int)

# ------------------------------
# scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_graph)
X_test_scaled  = scaler.transform(X_test_graph)
log_ram("After Scaling Graph Features")

# ------------------------------
# 3.7 Classification using stacking (KEEP architecture)
# ------------------------------
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

# ------------------------------
# evaluation on test graph rows
# ------------------------------
p_test = trained_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thr_roc = roc_curve(y_test_graph, p_test)
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

# ------------------------------
# Predict probabilities for ALL address-group rows
# ------------------------------
X_all_graph = feat_ag[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
X_all_scaled = scaler.transform(X_all_graph)
feat_ag['PredictedProb'] = trained_model.predict_proba(X_all_scaled)[:,1]
feat_ag['PredictedLabel'] = (feat_ag['PredictedProb'] >= best_threshold).astype(int)
log_ram("After Full Inference Graph")
gc.collect()

# ------------------------------
# Map predicted per address: average across groups -> address-level score
# ------------------------------
addr_avg = feat_ag.groupby('Address')['PredictedProb'].agg(['count','mean']).rename(columns={'count':'group_ct','mean':'addr_prob'}).reset_index()
# merge addr_prob into original df by Address when needed
addr_prob_map = dict(zip(addr_avg['Address'], addr_avg['addr_prob']))

# attach addr_prob to original flow-level df for per-sensor aggregation
src_prob = df['SrcAddr'].map(addr_prob_map)
dst_prob = df['DstAddr'].map(addr_prob_map)
df['AddrProb'] = np.maximum(src_prob.fillna(0), dst_prob.fillna(0))

# ------------------------------
# Per-sensor C&C detection + 3D plotting based on degree-graph (paper) using auto-threshold tuner
# ------------------------------
detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== [Sensor {sid}] Auto C&C Detection (graph-based, auto-threshold) ===")
    log_ram(f"Sensor {sid} Start")

    df_s = df[df["SensorId"] == sid].copy()
    if df_s.empty:
        print("[Info] sensor empty, skipping")
        continue

    # compute in/out counts for addresses within this sensor (use flows)
    agg_in = df_s.groupby("DstAddr")["AddrProb"].agg(["count","mean"]).rename(columns={"count":"in_ct","mean":"in_prob"})
    agg_out = df_s.groupby("SrcAddr")["AddrProb"].agg(["count","mean"]).rename(columns={"count":"out_ct","mean":"out_prob"})
    stats = agg_in.join(agg_out, how="outer").fillna(0)
    stats["in_ratio"]  = stats["in_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["out_ratio"] = stats["out_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["avg_prob"]  = (stats["in_prob"] + stats["out_prob"]) / 2
    stats["degree"]    = stats["in_ct"] + stats["out_ct"]

    # run auto threshold tuner
    thresholds = auto_threshold_tuner(stats, sensor_id=sid)
    prob_threshold = thresholds["prob_threshold"]
    degree_threshold = thresholds["degree_threshold"]
    ratio_threshold = thresholds["ratio_threshold"]

    print(f"thresholds: {thresholds}")
    print(f"prob_threshold: {prob_threshold}")
    print(f"degree_threshold: {degree_threshold}")
    print(f"ratio_threshold: {ratio_threshold}")

    # apply detection rule using thresholds
    node_roles = {}
    for n, r in stats.iterrows():
        if (r["avg_prob"] >= prob_threshold) and \
           (r["degree"] >= degree_threshold) and \
           (r["out_ratio"] >= ratio_threshold):
            node_roles[n] = "C&C"
        else:
            node_roles[n] = "Normal"

    cc_nodes = [n for n, role in node_roles.items() if role == "C&C"]
    print(f"[Detected] {len(cc_nodes)} potential C&C nodes in sensor {sid}")

    if cc_nodes:
        cnc_df = stats.loc[cc_nodes].copy()
        cnc_df["SensorId"] = sid
        cnc_df["cnc_score"] = cnc_df["avg_prob"] * (1 + cnc_df["in_ratio"] + cnc_df["out_ratio"]) * np.log1p(cnc_df["degree"])
        cnc_df = cnc_df.sort_values("cnc_score", ascending=False)
        detected_summary.append(cnc_df)
        log_ram(f"Sensor {sid} After C&C Score")

        # Build degree-graph for this sensor based on grouped edges (only groups that involve addresses in this sensor)
        # Filter group_edges rows where either src or dst appears in sensor flows
        addrs_in_sensor = set(df_s['SrcAddr']).union(set(df_s['DstAddr']))
        # select relevant group edges
        ge_s = group_edges[(group_edges['SrcAddr'].isin(addrs_in_sensor)) | (group_edges['DstAddr'].isin(addrs_in_sensor))].copy()
        if ge_s.empty:
            print("[Info] No group-edges for this sensor's addresses -> skip plot")
            continue

        # Build DiGraph from ge_s but collapse weights aggregated across groups for same pair
        agg_edges = ge_s.groupby(['SrcAddr','DstAddr'])['Weight'].sum().reset_index()
        G_full = nx.from_pandas_edgelist(agg_edges, source='SrcAddr', target='DstAddr', edge_attr='Weight', create_using=nx.DiGraph())

        # neighbors of C&C nodes
        cnc_neighbors = set()
        for cnc in cc_nodes:
            if cnc in G_full:
                cnc_neighbors.update(G_full.predecessors(cnc))
                cnc_neighbors.update(G_full.successors(cnc))

        nodes_keep = set(cc_nodes) | cnc_neighbors
        # ensure at least C&C nodes present in graph
        nodes_keep = nodes_keep.intersection(set(G_full.nodes()))
        # if still empty, fallback to top-degree nodes
        if not nodes_keep:
            degree_sorted = sorted(G_full.degree(), key=lambda x: x[1], reverse=True)
            nodes_keep = set([n for n,_ in degree_sorted[:min(50, len(degree_sorted))]])

        # Cap nodes by memory heuristics (reuse your limits)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 4:
            MAX_NODES, MAX_EDGES = 200, 400
        elif available_gb < 8:
            MAX_NODES, MAX_EDGES = 300, 600
        else:
            MAX_NODES, MAX_EDGES = 500, 800

        if len(nodes_keep) > MAX_NODES:
            # keep C&C nodes + sample neighbors
            normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
            sample_normal = np.random.choice(normal_nodes, size=min(MAX_NODES, len(normal_nodes)), replace=False)
            nodes_keep = set(cc_nodes) | set(sample_normal)

        G = G_full.subgraph(nodes_keep).copy()

        # prune edges if too many
        all_edges = list(G.edges(data=True))
        if len(all_edges) > MAX_EDGES:
            np.random.seed(RANDOM_STATE)
            edges_sample = np.random.choice(len(all_edges), size=MAX_EDGES, replace=False)
            edges_keep = [all_edges[i] for i in edges_sample]
            G = nx.DiGraph()
            for u,v,attr in edges_keep:
                G.add_edge(u,v, **attr)

        print(f"[Plot] Drawing {len(G.nodes())} nodes and {len(G.edges())} edges for sensor {sid}")
        node_color = []
        node_size = []
        for n in G.nodes():
            if n in cc_nodes:
                node_color.append("red")
                node_size.append(15)
            else:
                node_color.append("blue")
                node_size.append(6)

        # 3D layout
        pos = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=60)
        x_nodes = [pos[k][0] for k in G.nodes()]
        y_nodes = [pos[k][1] for k in G.nodes()]
        z_nodes = [pos[k][2] for k in G.nodes()]

        # edges for plot
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
                title=f"Sensor {sid} – 3D Degree Graph (Group-based, {len(G.nodes())} nodes)",
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

        graph_path = os.path.join(output_dir, f"Sensor{sid}_3DGraph_GroupDegree_{fileTimeStamp}.html")
        fig.write_html(graph_path)
        print(f"[Plot] 3D degree-graph saved -> {graph_path}")
        log_ram(f"Sensor {sid} After GroupDegree Plot")

    else:
        print("[Info] No C&C nodes detected in this sensor.")

    gc.collect()
    log_ram(f"Sensor {sid} End (Post-GC)")

print("\nDone. Graph-based pipeline + stacking model + per-sensor auto-threshold detection + 3D degree-graph exports complete.")
log_ram("Script End")
print(f"[Info] RAM usage log saved -> {memory_log_path}")
