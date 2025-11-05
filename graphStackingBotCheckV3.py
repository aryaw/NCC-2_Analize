"""
what we do?
 - load NCC dataset (all sensors)
 - fast_label_to_binary() > final label
 - activity grouping (auto-threshold G)
 - global model training, with fallback
 - per-sensor C&C detection using activity groups
 - graph visualization (C&C + up to 1000 neighbors)
 - CSV output
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
from libInternal import getConnection, setFileLocation, setExportDataLocation, optimize_dataframe

RANDOM_STATE = 42
SAFE_THREADS = "1"
MAX_RENDER_NODES = 1000
os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
    "NUMEXPR_NUM_THREADS": SAFE_THREADS,
    "JOBLIB_TEMP_FOLDER": "/tmp",
})

def fast_label_to_binary(df):
    """Convert Label → 0/1 using regex + numeric fallback."""
    labels_str = df["Label"].astype(str).str.lower().fillna("")

    bot_pattern = re.compile(
        r"\b(bot|botnet|cnc|c&c|malware|infected|attack|spam|ddos|trojan|worm|"
        r"zombie|backdoor|exploit|bruteforce|miner)\b",
        re.IGNORECASE
    )

    normal_pattern = re.compile(
        r"\b(normal|benign|background|legit|clean|regular|safe|harmless)\b",
        re.IGNORECASE
    )

    def classify(label):
        text = str(label)
        if bot_pattern.search(text):
            return 1
        elif normal_pattern.search(text):
            return 0
        return np.nan

    result = df["Label"].apply(classify)

    # numeric fallback
    numeric = pd.to_numeric(df["Label"], errors="coerce")
    result.loc[numeric.notna()] = (numeric.loc[numeric.notna()] >= 0.5).astype(int)

    before = len(df)
    df["Label"] = result
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)

    dropped = before - len(df)
    if dropped:
        print(f"[Label] Dropped {dropped:,} rows (undetermined labels)")

    print("[Label] value counts:\n", df["Label"].value_counts())
    return df

fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception:
    con = duckdb.connect()

print("[Load] Reading dataset...")
query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId, StartTime
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL
"""
df = con.sql(query).df()
if df.empty:
    raise RuntimeError("Dataset empty.")

df = optimize_dataframe(df)

df = fast_label_to_binary(df)
df["Label_bin"] = df["Label"]

df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "StartTime"])
df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
df = df.dropna(subset=["StartTime"])

dir_map = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str)
df["Dir"] = df["Dir_raw"].map(dir_map).fillna(0).astype(int)

for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

df["ByteRatio"] = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"] = df["TotPkts"] / (df["Dur"] + 0.1)
df["FlowIntensity"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["PktByteRatio"] = df["TotPkts"] / (df["TotBytes"] + 1)
df["SrcByteRatio"] = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = np.abs(df["sTos"] - df["dTos"])
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"] = df["TotBytes"] / (df["Dur"] + 1)

features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity", "PktByteRatio",
    "SrcByteRatio", "TrafficBalance", "DurationPerPkt", "Intensity"
]

print("\n[Activity Grouping] Computing time gaps...")
df = df.sort_values(["SrcAddr", "DstAddr", "StartTime"])
df["PrevTime"] = df.groupby(["SrcAddr", "DstAddr"])["StartTime"].shift(1)
df["TimeGap"] = (df["StartTime"] - df["PrevTime"]).dt.total_seconds().fillna(0)

positive = df.loc[df["TimeGap"] > 0, "TimeGap"]
if len(positive) > 0:
    median_gap = positive.median()
    iqr_gap = positive.quantile(0.75) - positive.quantile(0.25)
    G = median_gap + 2 * iqr_gap
    if G <= 0 or not np.isfinite(G):
        G = 30
else:
    G = 30
print(f"[Activity Grouping] G = {G:.2f} sec")

df["ActivityGroup"] = df.groupby(["SrcAddr", "DstAddr"])["TimeGap"].apply(lambda x: (x > G).cumsum())
print(f"[Activity Grouping] Groups: {df['ActivityGroup'].nunique():,}")

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["Label_bin"].astype(int)

scaler = MinMaxScaler()
qt = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)
X_scaled = qt.fit_transform(scaler.fit_transform(X)).astype(np.float32)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

print("\n[Train] Training stacking model...")
try:
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=120, max_depth=12, class_weight="balanced", n_jobs=1)),
            ("et", ExtraTreesClassifier(n_estimators=120, max_depth=12, n_jobs=1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=120, max_depth=8)),
        ],
        final_estimator=HistGradientBoostingClassifier(max_iter=80),
        passthrough=True,
        cv=3,
        n_jobs=1,
    )
    stack.fit(X_train, y_train)
    model = stack
    print("[Train] Stacking model OK.")
except Exception as e:
    print("[WARN] Stacking failed:", e)
    model = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced", n_jobs=1)
    model.fit(X_train, y_train)
    print("[Fallback] Using RandomForest.")

# evaluate
p = model.predict_proba(X_test)[:, 1]
fpr, tpr, thr = roc_curve(y_test, p)
best_threshold = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]
y_pred = (p >= best_threshold).astype(int)

print("\n# Evaluation")
print("Threshold:", best_threshold)
print("Accuracy:", (y_pred == y_test).mean())
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, p))

del X_train, X_test, y_train, y_test
gc.collect()

print("\n[Predict] Scoring full dataset...")
df["PredictedProb"] = model.predict_proba(X_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
del X_scaled
gc.collect()

detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== Sensor {sid} ===")
    df_s = df[df["SensorId"] == sid].copy()

    # group-level aggregation
    agg = df_s.groupby(["SrcAddr", "DstAddr", "ActivityGroup"]).agg(
        avg_prob=("PredictedProb", "mean"),
        count=("PredictedProb", "size"),
        start=("StartTime", "min"),
        end=("StartTime", "max")
    ).reset_index()
    agg["duration"] = (agg["end"] - agg["start"]).dt.total_seconds()
    agg["intensity"] = agg["count"] / (agg["duration"] + 1)

    agg["is_cnc"] = (agg["avg_prob"] > 0.7) & (agg["intensity"] > 0.2)
    cnc = agg[agg["is_cnc"]]

    print(f"[Detected] {len(cnc)} C&C activity groups")

    if len(cnc):
        cnc["SensorId"] = sid
        detected_summary.append(cnc)

    print("[Graph] Building graph…")

    df_vis = df_s.groupby(["SrcAddr", "DstAddr"], as_index=False).agg({
        "PredictedProb": "mean",
        "Dir_raw": lambda x: x.value_counts().index[0],
        "TotBytes": "sum",
        "TotPkts": "sum"
    })
    G_all = nx.from_pandas_edgelist(df_vis, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    top_srcs = cnc["SrcAddr"].unique().tolist()
    must_include = set(top_srcs)

    neighbors = set()
    for n in must_include:
        if n in G_all:
            neighbors.update(G_all.predecessors(n))
            neighbors.update(G_all.successors(n))

    important = set(list(must_include) + list(neighbors))

    # pad with normals until MAX_RENDER_NODES
    if len(important) < MAX_RENDER_NODES:
        for node in list(G_all.nodes()):
            if len(important) >= MAX_RENDER_NODES:
                break
            important.add(node)

    df_sub = df_vis[
        df_vis["SrcAddr"].isin(important) | df_vis["DstAddr"].isin(important)
    ]

    G = nx.from_pandas_edgelist(df_sub, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    # recompute small stats
    agg_in = df_sub.groupby("DstAddr")["PredictedProb"].agg(["mean", "count"]).rename(columns={"mean": "in_prob", "count": "in_ct"})
    agg_out = df_sub.groupby("SrcAddr")["PredictedProb"].agg(["mean", "count"]).rename(columns={"mean": "out_prob", "count": "out_ct"})
    stats = agg_in.join(agg_out, how="outer").fillna(0)
    stats["avg_prob"] = (stats["in_prob"] + stats["out_prob"]) / 2
    stats["degree"] = stats["in_ct"] + stats["out_ct"]

    print(f"[Graph] Nodes={len(G.nodes())}, Edges={len(G.edges())}")

    if len(G) == 0:
        continue

    pos = nx.spring_layout(G, k=0.5, iterations=20, seed=RANDOM_STATE)

    # edges
    edge_x, edge_y = [], []
    for s, d in G.edges():
        if s not in pos or d not in pos:
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[d]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.25, color="#AAA"),
        hoverinfo="none"
    )

    # nodes
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []

    cnc_srcs = set(cnc["SrcAddr"].unique().tolist())

    for n, (x, y) in pos.items():
        s = stats.loc[n] if n in stats.index else None
        avg_prob = float(s["avg_prob"]) if s is not None else 0
        deg = int(s["degree"]) if s is not None else 0

        # role
        role = "Normal"
        if n in cnc_srcs:
            role = "C&C"
        elif n in must_include:
            role = "Candidate"

        node_x.append(x); node_y.append(y)
        node_text.append(f"{n}<br>{role}<br>Prob:{avg_prob:.3f}<br>Deg:{deg}")

        if role == "C&C":
            node_color.append("#FF0000"); node_size.append(48)
        elif role == "Candidate":
            node_color.append("#FFA500"); node_size.append(14)
        else:
            node_color.append("#BFC9CA"); node_size.append(6)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        hoverinfo="text", hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#333"))
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=f"Sensor {sid} – C&C Graph (Top sources + {MAX_RENDER_NODES} neighbors)",
        title_x=0.5,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white", plot_bgcolor="white"
    ))

    html_graph = os.path.join(output_dir, f"Sensor{sid}_CNC_Graph_{fileTimeStamp}.html")
    fig.write_html(html_graph)
    print(f"[Export] Graph → {html_graph}")

if detected_summary:
    summary_df = pd.concat(detected_summary, ignore_index=True)
    csv_out = os.path.join(output_dir, f"CNC_AutoDetected_{fileTimeStamp}.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\n[Export] Summary saved → {csv_out}")
else:
    print("\n[Summary] No C&C detected across sensors.")

print("\nDone.")
