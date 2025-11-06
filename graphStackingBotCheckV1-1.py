"""
What we do?
 - read NCC dataset for 1 SensorId
 - convert Label -> binary via fast_label_to_binary()
 - train compact stacking model
 - pick decision threshold
 - predict on all rows, for the selected sensor
 - direction aware C&C detection using Dir field (->, <-, <->)
 - only "C&C" or "Normal" > no Bot / suspicious
 - rendergraph
 - export train/test and detected C&C CSV
"""

import os
import gc
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from libInternal import (
    variableDump,
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
    generate_plotly_evaluation_report
)


os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

SELECTED_SENSOR_ID = "1"
MAX_EDGES = 5_000 
RANDOM_STATE = 42
CNC_MIN_AVG_PROB   = 0.75   
CNC_MIN_DEGREE     = 3      
CNC_MIN_IN_RATIO   = 0.60   

fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), using direct DuckDB connection.")
    con = duckdb.connect()

query = f"""
SELECT
  SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
  sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL AND SensorId = '{SELECTED_SENSOR_ID}'
"""
print(f"[Load] Running query for SensorId = {SELECTED_SENSOR_ID} ...")
df = con.sql(query).df()
print(f"[Load] rows: {len(df):,}")

if df.empty:
    raise RuntimeError(f"No data found for SensorId = {SELECTED_SENSOR_ID}")

df = optimize_dataframe(df)
print(f"[Load] memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


df = fast_label_to_binary(df)
print("\n[Label] Distribution after conversion:")
print(df["Label"].value_counts(dropna=False))

if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")   
df["Dir"]     = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)  

for c in ["Proto", "State"]:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts",
    "sTos", "dTos", "SrcBytes",
]
missing = [c for c in features if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required features in dataframe: {missing}")

print(f"[Features] Using features ({len(features)}): {features}")


X = df[features].fillna(df[features].mean())
y = np.rint(df["Label"]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test  distribution:", np.bincount(y_test))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

base_learners = [
    ("rf",  RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=1
    )),
    ("hgb", HistGradientBoostingClassifier(
        max_depth=6, random_state=RANDOM_STATE
    )),
]
meta = ExtraTreesClassifier(
    n_estimators=30, max_depth=6,
    random_state=RANDOM_STATE, n_jobs=1
)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=3,
    n_jobs=1,
    passthrough=False,
    verbose=1
)

gc.collect()
print(f"\n[Train] Training stacking model for Sensor {SELECTED_SENSOR_ID} ...")
stack.fit(X_train_scaled, y_train)
print("[Train] Done.")
gc.collect()

p_test = stack.predict_proba(X_test_scaled)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n===Evaluation for Sensor:", SELECTED_SENSOR_ID)
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
try:
    print("ROC-AUC:", roc_auc_score(y_test, p_test))
except Exception:
    print("ROC-AUC: NA (single class in split?)")
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

train_csv = os.path.join(outputdata_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
test_csv  = os.path.join(outputdata_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")

pd.concat([X_train.reset_index(drop=True),
           pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test.reset_index(drop=True),
           pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

df_scaled = scaler.transform(df[features])
df["PredictedProb"]  = stack.predict_proba(df_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)

print(f"[Graph] Generating visualization for Sensor {SELECTED_SENSOR_ID} with {len(df):,} edges...")
df_vis = df.sample(n=MAX_EDGES, random_state=RANDOM_STATE) if len(df) > MAX_EDGES else df.copy()

G = nx.from_pandas_edgelist(df_vis, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

inbound_counts, outbound_counts = {}, {}
prob_sum, edge_count = {}, {}

for _, row in df_vis.iterrows():
    src = row["SrcAddr"]
    dst = row["DstAddr"]
    prob = float(row["PredictedProb"])
    dir_raw = str(row["Dir_raw"]).strip()

    for node in (src, dst):
        inbound_counts.setdefault(node, 0)
        outbound_counts.setdefault(node, 0)
        prob_sum.setdefault(node, 0.0)
        edge_count.setdefault(node, 0)

    prob_sum[src] += prob
    prob_sum[dst] += prob
    edge_count[src] += 1
    edge_count[dst] += 1
   
    if dir_raw == "->":
        outbound_counts[src] += 1
        inbound_counts[dst] += 1
    elif dir_raw == "<-":
        outbound_counts[dst] += 1
        inbound_counts[src] += 1
    elif dir_raw == "<->":
        outbound_counts[src] += 1
        outbound_counts[dst] += 1
        inbound_counts[src]  += 1
        inbound_counts[dst]  += 1
    else:
        outbound_counts[src] += 1
        inbound_counts[dst]  += 1


node_roles = {}
cnc_candidates = []

for node in G.nodes():
    in_ct  = inbound_counts.get(node, 0)
    out_ct = outbound_counts.get(node, 0)
    deg    = edge_count.get(node, 0)
    avg_p  = prob_sum.get(node, 0.0) / max(deg, 1)

    total = in_ct + out_ct
    in_ratio = (in_ct / total) if total > 0 else 0.0
   
    if (
        (avg_p >= CNC_MIN_AVG_PROB) and
        (deg >= CNC_MIN_DEGREE) and
        (in_ratio >= CNC_MIN_IN_RATIO or in_ratio <= (1.0 - CNC_MIN_IN_RATIO))
    ):
        node_roles[node] = "C&C"
        cnc_candidates.append((node, avg_p, in_ratio, deg))
    else:
        node_roles[node] = "Normal"


cnc_candidates = sorted(
    cnc_candidates,
    key=lambda t: (t[1] * np.log1p(t[3])),
    reverse=True
)
print(f"\n[Detected] {len(cnc_candidates)} C&C nodes (C&C-only mode).")
for n, ap, ir, dg in cnc_candidates[:20]:
    print(f" - {n} | AvgProb={ap:.3f} InRatio={ir:.2f} Degree={dg}")

if cnc_candidates:
    cnc_df = pd.DataFrame(
        [{"Node": n, "AvgProb": ap, "InRatio": ir, "Degree": dg} for (n, ap, ir, dg) in cnc_candidates]
    )
    cnc_csv = os.path.join(output_dir, f"Detected_CNC_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
    cnc_df.to_csv(cnc_csv, index=False)
    print(f"[Export] C&C CSV -> {cnc_csv}")

pos = nx.kamada_kawai_layout(G)

edge_x, edge_y = [], []
for _, row in df_vis.iterrows():
    s, d = row["SrcAddr"], row["DstAddr"]
    if s not in pos or d not in pos:
        continue
    x0, y0 = pos[s]
    x1, y1 = pos[d]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.6, color="#AAAAAA"),
    mode="lines", hoverinfo="none"
)

node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    role  = node_roles.get(node, "Normal")
    avg_p = prob_sum.get(node, 0.0) / max(edge_count.get(node, 1), 1)
    in_ct = inbound_counts.get(node, 0)
    out_ct = outbound_counts.get(node, 0)

    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Role: {role}<br>AvgProb: {avg_p:.3f}<br>In/Out: {in_ct}/{out_ct}")

    if role == "C&C":
        node_color.append("#FF0000")
        node_size.append(28)
    else:
        node_color.append("#CCCCCC")
        node_size.append(8)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers",
    hovertext=node_text,
    hoverinfo="text",
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333")),
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Sensor {SELECTED_SENSOR_ID} – C&C-only Network View (sampled {len(df_vis):,} edges)",
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

html_output = os.path.join(graph_dir, f"CNCOnlyGraph_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")

print(f"\nDone. Trained model and C&C-only graph generated for Sensor {SELECTED_SENSOR_ID}.")
