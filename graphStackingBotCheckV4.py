"""
what we do here?
- train one global model, detect per sensor
- Stacking RF + ET + HGB stacking, with fallback algo
- graph shows C&C + neighbors
- auto time_gap threshold (median + 2*IQR) per sensor

expect output:
- global model metrics
- per-sensor C&C activity-group summary in CSV
- per-sensor graph HTML C&C + neighbors only
"""

import os
import gc
import re
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

RANDOM_STATE = 42
SAFE_THREADS = "1"
MAX_RENDER_NODES = 1000 
MODEL_TEST_SIZE = 0.30 # 70/30 split
os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
    "NUMEXPR_NUM_THREADS": SAFE_THREADS,
    "JOBLIB_TEMP_FOLDER": "/tmp",
})

from libInternal import (
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
)

fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

# labelin func using regex, 1 = bot; 0 = normal
def fast_label_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    labels_str = df["Label"].astype(str).fillna("").str.lower()
    bot_pattern = re.compile(
        r"\b(bot|botnet|cnc|c&c|malware|infected|attack|spam|ddos|dos|trojan|worm|"
        r"zombie|backdoor|virus|phish|miner|exploit|bruteforce|scanner|adware|suspicious)\b",
        re.IGNORECASE,
    )
    normal_pattern = re.compile(
        r"\b(normal|benign|background|legit|clean|regular|safe|harmless)\b",
        re.IGNORECASE,
    )

    def classify_label(text: str):
        if bot_pattern.search(text):
            return 1
        if normal_pattern.search(text):
            return 0
        return np.nan

    result = labels_str.apply(classify_label)

    # numeric fallback
    numeric = pd.to_numeric(df["Label"], errors="coerce")
    idx_num = numeric.notna()
    result.loc[idx_num] = (numeric.loc[idx_num] >= 0.5).astype(int)

    before = len(df)
    df = df.copy()
    df["Label"] = result
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"[Label] dropped {dropped:,} rows with undetermined Label")
    print("[Label] value counts:\n", df["Label"].value_counts())
    return df

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception:
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId, StartTime
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL
"""
print("[Load] Reading dataset...")
df = con.sql(query).df()
if df.empty:
    raise RuntimeError("No labeled data found in dataset.")

df = optimize_dataframe(df)
print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

# keep a clean working copy
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "StartTime"]).copy()
df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
df = df.dropna(subset=["StartTime"])

# label binary (regex + numeric)
# sets df["Label"] to 0/1
df = fast_label_to_binary(df)

# direction mapping
dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

# categorical encoding
for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# feature engineering
df["ByteRatio"] = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"] = df["TotPkts"] / (df["Dur"] + 0.1)
df["FlowIntensity"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["PktByteRatio"] = df["TotPkts"] / (df["TotBytes"] + 1)
df["SrcByteRatio"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = (df["sTos"] - df["dTos"]).abs()
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"] = df["TotBytes"] / (df["Dur"] + 1)

features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity", "PktByteRatio",
    "SrcByteRatio", "TrafficBalance", "DurationPerPkt", "Intensity"
]
print(f"[Features] {len(features)} prepared.")

# train/test split
X_all = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_all = df["Label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=MODEL_TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
)

scaler = MinMaxScaler()
qt = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)

X_train_scaled = qt.fit_transform(scaler.fit_transform(X_train)).astype(np.float32)
X_test_scaled  = qt.transform(scaler.transform(X_test)).astype(np.float32)

# train model w/ fallback
trained_model = None
print("\n[Train] training stacking model with fallback")
try:
    # base_learners = [
    #     ("rf", RandomForestClassifier(
    #         n_estimators=120, max_depth=12, class_weight="balanced",
    #         random_state=RANDOM_STATE, n_jobs=1
    #     )),
    #     ("et", ExtraTreesClassifier(
    #         n_estimators=120, max_depth=12,
    #         random_state=RANDOM_STATE, n_jobs=1
    #     )),
    #     ("hgb", HistGradientBoostingClassifier(
    #         max_iter=120, max_depth=8, learning_rate=0.05,
    #         random_state=RANDOM_STATE
    #     )),
    # ]
    base_learners = [
        ("rf", RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=8,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )),
    ]
    print("[Train] Finish base conf.")
    meta = HistGradientBoostingClassifier(
        max_iter=100, max_depth=8, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    print("[Train] Finish meta conf.")
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=3, passthrough=True, n_jobs=1, verbose=1
    )
    print("[Train] Start stack fit.")
    stack.fit(X_train_scaled, y_train)
    trained_model = stack
    print("[Train] Full stacking model trained successfully.")
except Exception as e:
    print(f"[WARN] Stacking failed: {type(e).__name__}: {e}")
    print("[Fallback] Training RandomForestClassifier...")
    try:
        fallback_rf = RandomForestClassifier(
            n_estimators=150, max_depth=14, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        )
        fallback_rf.fit(X_train_scaled, y_train)
        trained_model = fallback_rf
        print("[Fallback] RandomForest trained successfully.")
    except Exception as e2:
        print(f"[WARN] RandomForest failed: {type(e2).__name__}: {e2}")
        fallback_hgb = HistGradientBoostingClassifier(
            max_iter=120, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE
        )
        fallback_hgb.fit(X_train_scaled, y_train)
        trained_model = fallback_hgb
        print("[Fallback] HistGradientBoosting trained successfully.")

# evaluate
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

del X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, p_test
gc.collect()

# inference for all rows
X_scaled_all = qt.transform(scaler.transform(X_all)).astype(np.float32)
df["PredictedProb"] = trained_model.predict_proba(X_scaled_all)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
del X_scaled_all, X_all, y_all
gc.collect()

# per-sensor activity grouping
def compute_activity_groups(sensor_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    sdf = sensor_df.sort_values(["SrcAddr", "DstAddr", "StartTime"]).copy()
    sdf["PrevTime"] = sdf.groupby(["SrcAddr", "DstAddr"])["StartTime"].shift(1)
    sdf["TimeGap"]  = (sdf["StartTime"] - sdf["PrevTime"]).dt.total_seconds().fillna(0)

    pos = sdf.loc[sdf["TimeGap"] > 0, "TimeGap"]
    if len(pos) > 0:
        median_gap = pos.median()
        iqr = pos.quantile(0.75) - pos.quantile(0.25)
        G = median_gap + 2 * iqr
        if not np.isfinite(G) or G <= 0:
            G = 30.0
    else:
        G = 30.0
    sdf["ActivityGroup"] = sdf.groupby(["SrcAddr", "DstAddr"])["TimeGap"].apply(lambda x: (x > G).cumsum())
    return sdf, float(G)

# per-sensor detect
detected_summary = []
sensors = sorted(df["SensorId"].unique())
print(f"\n[Detect] Running per-sensor C&C detection for {len(sensors)} sensors...")

for sid in sensors:
    print(f"\n=== [Sensor {sid}] Activity-group aware C&C detection ===")
    df_s_raw = df[df["SensorId"] == sid].copy()
    if df_s_raw.empty:
        print("No data for this sensor.")
        continue

    # activity groups inside this sensor
    df_s, G = compute_activity_groups(df_s_raw)
    print(f"[Activity Grouping] Sensor {sid}: G = {G:.2f}s, groups = {df_s['ActivityGroup'].nunique():,}")

    # aggregate per (Src, Dst, ActivityGroup)
    agg = df_s.groupby(["SrcAddr", "DstAddr", "ActivityGroup"]).agg(
        avg_prob=("PredictedProb", "mean"),
        count=("PredictedProb", "size"),
        start_time=("StartTime", "min"),
        end_time=("StartTime", "max")
    ).reset_index()
    agg["duration"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds()
    agg["intensity"] = agg["count"] / (agg["duration"] + 1)

    # group rule: high avg_prob + some density
    agg["is_cnc_group"] = ((agg["avg_prob"] > 0.70) & (agg["intensity"] > 0.20))

    cnc_groups = agg[agg["is_cnc_group"]].copy()
    print(f"[Detected] {len(cnc_groups)} C&C activity groups in sensor {sid}")

    # if none, take top 1 by score when avg_prob reasonably high > avoid empty graphs
    if cnc_groups.empty and not agg.empty:
        agg["score"] = agg["avg_prob"] * (agg["intensity"] + 1) * np.log1p(agg["count"])
        top_candidate = agg.sort_values("score", ascending=False).head(1)
        if (len(top_candidate) == 1) and (top_candidate["avg_prob"].iloc[0] > 0.60):
            cnc_groups = top_candidate.copy()
            cnc_groups["is_cnc_group"] = True
            print("[Fallback] Promoted top activity group as candidate C&C (no strict matches).")

    if not cnc_groups.empty:
        cnc_groups["SensorId"] = sid
        detected_summary.append(cnc_groups)

    # graph
    df_vis = df_s.groupby(["SrcAddr", "DstAddr"], as_index=False).agg({
        "PredictedProb": "mean",
        "Dir_raw": lambda x: x.value_counts().index[0] if len(x) else "->",
        "TotBytes": "sum",
        "TotPkts": "sum"
    })

    G_all = nx.from_pandas_edgelist(df_vis, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    # must-include nodes, sources from C&C groups
    cnc_sources = cnc_groups["SrcAddr"].unique().tolist() if not cnc_groups.empty else []
    must_include = set(cnc_sources)

    # expand with neighbors
    neighbors = set()
    for n in list(must_include):
        if n in G_all:
            neighbors.update(G_all.predecessors(n))
            neighbors.update(G_all.successors(n))

    important_nodes = set(list(must_include) + list(neighbors))

    # pad with more nodes
    if len(important_nodes) < MAX_RENDER_NODES:
        for cnode in G_all.nodes():
            if len(important_nodes) >= MAX_RENDER_NODES:
                break
            if cnode not in important_nodes:
                important_nodes.add(cnode)

    # filter edges to those touching important_nodes
    df_vis_sub = df_vis[
        df_vis["SrcAddr"].isin(important_nodes) | df_vis["DstAddr"].isin(important_nodes)
    ].copy()

    G_sub = nx.from_pandas_edgelist(df_vis_sub, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    # Small stats for coloring/size
    agg_in_sub = df_vis_sub.groupby("DstAddr")["PredictedProb"].agg(["count", "mean"]).rename(columns={"count":"in_ct","mean":"in_prob"})
    agg_out_sub = df_vis_sub.groupby("SrcAddr")["PredictedProb"].agg(["count", "mean"]).rename(columns={"count":"out_ct","mean":"out_prob"})
    stats_sub = agg_in_sub.join(agg_out_sub, how="outer").fillna(0)
    stats_sub["avg_prob"] = (stats_sub.get("in_prob", 0) + stats_sub.get("out_prob", 0)) / 2
    stats_sub["degree"] = stats_sub.get("in_ct", 0) + stats_sub.get("out_ct", 0)

    print(f"[Graph] Sensor {sid}: nodes = {len(G_sub.nodes())}, edges = {len(G_sub.edges())}")

    # Layout and rendering (interactive .html)
    if len(G_sub.nodes()) > 0:
        pos = nx.spring_layout(G_sub, k=0.5, iterations=20, seed=RANDOM_STATE)
    else:
        pos = {}

    # edges
    edge_x, edge_y = [], []
    for _, row in df_vis_sub.iterrows():
        s, d = row["SrcAddr"], row["DstAddr"]
        if s not in pos or d not in pos:
            continue
        x0, y0 = pos[s]; x1, y1 = pos[d]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.25, color="#AAA"), hoverinfo="none"
    )

    # nodes
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    cnc_source_set = set(cnc_sources)
    for n, (x, y) in pos.items():
        role = "Normal"
        if n in cnc_source_set:
            role = "C&C"
        elif n in must_include:
            role = "Candidate"

        srow = stats_sub.loc[n] if n in stats_sub.index else None
        avg_prob = float(srow["avg_prob"]) if srow is not None else 0.0
        deg = int(srow["degree"]) if srow is not None else 0

        node_x.append(x); node_y.append(y)
        node_text.append(f"{n}<br>Role:{role}<br>AvgProb:{avg_prob:.4f}<br>Degree:{deg}")

        if role == "C&C":
            node_color.append("#FF0000"); node_size.append(56 if n in must_include else 36)
        elif role == "Candidate":
            node_color.append("#FFA500"); node_size.append(16)
        else:
            node_color.append("#BFC9CA"); node_size.append(6)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        hovertext=node_text, hoverinfo="text",
        marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333"))
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Sensor {sid} – Auto C&C Graph (C&C + neighbors ~{MAX_RENDER_NODES})",
            title_x=0.5, showlegend=False, hovermode="closest",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )

    html_graph = os.path.join(output_dir, f"Sensor{sid}_CConlyGraph_{fileTimeStamp}.html")
    fig.write_html(html_graph)
    print(f"[Export] Graph -> {html_graph}")

# export CSV
if detected_summary:
    summary_df = pd.concat(detected_summary, ignore_index=True)
    summary_csv = os.path.join(output_dir, f"CNC_AutoDetected_ActivityGroup_{fileTimeStamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[Export] Activity-grouped C&C summary saved -> {summary_csv}")
else:
    print("\n[Summary] No C&C groups detected across sensors.")

print("\nDone. Global training (A2) + per-sensor detection (C1) with auto time grouping (D1) complete.")
