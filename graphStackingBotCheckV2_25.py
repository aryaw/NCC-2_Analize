# ---
# cek ombak
# Flow-Based Command-and-Control Detection Using Ensemble Machine Learning and Network Graph Feature Analysis
# ---
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
# MAX_ROWS_FOR_STACKING = 10_500_000
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

edge_weights = df.groupby(["SrcAddr", "DstAddr"]).size().reset_index(name="EdgeWeight")
df = df.merge(edge_weights, on=["SrcAddr", "DstAddr"], how="left")
df["EdgeWeight"] = df["EdgeWeight"].fillna(1).astype(int)

src_total = df.groupby("SrcAddr")["EdgeWeight"].sum().rename("SrcTotalWeight")
dst_total = df.groupby("DstAddr")["EdgeWeight"].sum().rename("DstTotalWeight")

df = df.merge(src_total, left_on="SrcAddr", right_index=True, how="left")
df = df.merge(dst_total, left_on="DstAddr", right_index=True, how="left")
df["SrcTotalWeight"] = df["SrcTotalWeight"].fillna(0)
df["DstTotalWeight"] = df["DstTotalWeight"].fillna(0)

# end of edge-weight feature block

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
pd.DataFrame(X_train, columns=features).to_csv(os.path.join(outputdata_dir, f"train_features_{fileDataTimeStamp}.csv"), index=False)
pd.DataFrame(y_train, columns=["Label"]).to_csv(os.path.join(outputdata_dir, f"train_labels_{fileDataTimeStamp}.csv"), index=False)
pd.DataFrame(X_test, columns=features).to_csv(os.path.join(outputdata_dir, f"test_features_{fileDataTimeStamp}.csv"), index=False)
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

# FULL DATA INFERENCE
df["PredictedProb"]  = model.predict_proba(scaler.transform(X_full))[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
log_ram("After Full Inference")
gc.collect()


detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== Sensor {sid} ===")
    df_s = df[df["SensorId"] == sid].copy()

    # Aggregate inbound/outbound (counts and mean predicted prob)
    agg_in  = df_s.groupby("DstAddr")["PredictedProb"].agg(["count","mean"])
    agg_out = df_s.groupby("SrcAddr")["PredictedProb"].agg(["count","mean"])

    agg_in.columns  = ["in_ct","in_prob"]
    agg_out.columns = ["out_ct","out_prob"]

    stats = agg_in.join(agg_out, how="outer").fillna(0)

    # Add unique peer counts: how many distinct srcs contacted this dest, and how many distinct dsts a src contacted
    out_unique = df_s.groupby("SrcAddr")["DstAddr"].nunique().rename("out_unique_dests")
    in_unique  = df_s.groupby("DstAddr")["SrcAddr"].nunique().rename("in_unique_srcs")
    stats = stats.join(out_unique, how="left").join(in_unique, how="left").fillna(0)

    # Add aggregated weight features at stats level (use per-node sums)
    # total out-weight per src (how many flows outgoing from a given src)
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

    # cnc_score using weighted graph-informed features (degree, ratios, unique-peers, node weight sums)
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
    auto_strict_thr = min(auto_strict_thr, 0.95)  # cap high end
    auto_strict_thr = max(auto_strict_thr, 0.40)  # do not drop below reasonable level

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

    real_cnc_nodes = df_s[df_s["LabelCNC"] == 1]["DstAddr"].unique().tolist()
    real_cnc_set = set(real_cnc_nodes)

    cc_nodes = sorted(set().union(strict_nodes, percentile_nodes, real_cnc_set))

    reasons = {}
    for n in cc_nodes:
        # defensive lookup
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

    candidates_path = os.path.join(outputdata_dir, f"Sensor{sid}_CNC_Candidates_{fileDataTimeStamp}.csv")
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

        # Graph Generation (weighted edges, safe sampling & preserve node set)
        print("[Plot] Generating 3D interactive network graph")

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 4:
            MAX_NODES, MAX_EDGES = 200, 400
        elif available_gb < 8:
            MAX_NODES, MAX_EDGES = 300, 600
        else:
            MAX_NODES, MAX_EDGES = 500, 800

        log_ram(f"Sensor {sid} Graph Limits: {MAX_NODES} nodes, {MAX_EDGES} edges")

        # Build weighted directed graph for this sensor using aggregated edge weights
        ew = df_s.groupby(["SrcAddr", "DstAddr"]).size().reset_index(name="weight")
        G_full = nx.from_pandas_edgelist(
            ew,
            source="SrcAddr",
            target="DstAddr",
            edge_attr="weight",
            create_using=nx.DiGraph()
        )

        # Determine neighbor expansion: include predecessors and successors up to 1-hop
        cnc_neighbors = set()
        for cnc in cc_nodes:
            if cnc in G_full:
                cnc_neighbors.update(G_full.predecessors(cnc))
                cnc_neighbors.update(G_full.successors(cnc))

        nodes_keep = set(cc_nodes) | cnc_neighbors

        # If still too large, sample non-cc nodes by degree (prefer high-degree)
        if len(nodes_keep) > MAX_NODES:
            normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
            normal_nodes_scores = [(n, G_full.degree(n)) for n in normal_nodes]
            normal_nodes_scores.sort(key=lambda x: x[1], reverse=True)
            keep_normals = [n for n, _ in normal_nodes_scores[: max(0, MAX_NODES - len(cc_nodes))]]
            nodes_keep = set(cc_nodes) | set(keep_normals)

        # Subgraph and preserve original graph connectivity among chosen nodes
        G = G_full.subgraph(nodes_keep).copy()
        del G_full

        # If edges exceed limit, prioritize heavy edges adjacent to cc_nodes first, then fill with remaining heavy edges
        all_edges = list(G.edges(data=True))
        if len(all_edges) > MAX_EDGES:
            # prioritized edges: adjacent to cc_nodes, sorted by weight desc
            prioritized = [e for e in all_edges if (e[0] in cc_nodes or e[1] in cc_nodes)]
            prioritized.sort(key=lambda x: x[2].get("weight", 1), reverse=True)

            remaining = [e for e in all_edges if e not in prioritized]
            remaining.sort(key=lambda x: x[2].get("weight", 1), reverse=True)

            keep_edges = []
            keep_edges.extend(prioritized)
            if len(keep_edges) < MAX_EDGES:
                needed = MAX_EDGES - len(keep_edges)
                keep_edges.extend(remaining[:needed])

            # Ensure we don't exceed MAX_EDGES
            keep_edges = keep_edges[:MAX_EDGES]

            # rebuild graph with original nodes (so isolated nodes remain visible) and weighted edges
            G = nx.DiGraph()
            G.add_nodes_from(nodes_keep)
            for u, v, d in keep_edges:
                G.add_edge(u, v, **(d if isinstance(d, dict) else {}))

        print(f"[Plot] Drawing {len(G.nodes())} nodes and {len(G.edges())} edges")

        pos = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=60)

        top4_nodes = []
        if 'cnc_df' in locals() and not cnc_df.empty:
            # keep only nodes that are in graph G
            cnc_present = cnc_df[cnc_df.index.isin(G.nodes())]
            top4_nodes = cnc_present.sort_values("cnc_score", ascending=False).head(4).index.tolist()

        # fallback ensure list
        top4_set = set(top4_nodes)

        # --- build edge traces (all grey, clean)
        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color="#B0B0B0", width=1.2),  # soft grey
            hoverinfo='none',
            name='edges'
        )

        # --- PREPARE NODE APPEARANCE
        x_nodes = [pos[n][0] for n in G.nodes()]
        y_nodes = [pos[n][1] for n in G.nodes()]
        z_nodes = [pos[n][2] for n in G.nodes()]

        node_color = []
        node_size = []
        node_text = []

        for n in G.nodes():
            if n in top4_set:
                node_color.append("#F11F1F")         # bright red
                node_size.append(22)
                node_text.append(n)
            elif n in cc_nodes:
                node_color.append("#E4A41A")         # yellow
                node_size.append(12)
                node_text.append("")                 # no label
            else:
                node_color.append("#1F77B4")         # blue
                node_size.append(6)
                node_text.append("")

        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.92,
                line=dict(width=1.5, color="black")  # subtle outline for clarity
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
                title=f"Sensor {sid} – Clean 3D Network Graph (C&C Highlighted)",
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

        graph_path = os.path.join(output_dir, f"Sensor{sid}_3DGraphAgencyStyle_{fileTimeStamp}.html")
        fig.write_html(graph_path)
        print(f"[Plot] Clean digital-agency style graph saved -> {graph_path}")
        log_ram(f"Sensor {sid} After Clean Plot")


    else:
        print("[Info] No C&C nodes detected in this sensor.")

    gc.collect()
    log_ram(f"Sensor {sid} End (Post-GC)")

print("\nDone. All CSV exported successfully.")
log_ram("Script End")
