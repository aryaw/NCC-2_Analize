import os
import gc
import joblib
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import traceback
import re  # 🔧 CHANGED: added regex support
from datetime import datetime
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
from libInternal import variableDump, getConnection, setFileLocation, setExportDataLocation

# === JOB LIMITS TO AVOID CPU OVERLOAD ===
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# === SELECT SENSOR ID ===
SELECTED_SENSOR_ID = "1"


def optimize_dataframe(df):
    """Downcast numeric columns to save memory."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# 🔧 CHANGED: Improved label mapping with regex
def fast_label_to_binary(df):
    """Convert 'Label' column to binary (1=bot/attack/malicious, 0=normal/benign) using regex matching."""
    labels_str = df["Label"].astype(str).str.lower().fillna("")

    # Flexible regex patterns
    bot_pattern = re.compile(
        r"\b(bot|botnet|cnc|c&c|malware|infected|attack|spam|ddos|trojan|worm|zombie|backdoor)\b",
        re.IGNORECASE,
    )
    normal_pattern = re.compile(
        r"\b(normal|benign|background|legit|clean|regular|safe|harmless)\b",
        re.IGNORECASE,
    )

    result = pd.Series(np.nan, index=df.index)

    # Apply regex checks
    result[df["Label"].astype(str).apply(lambda x: bool(bot_pattern.search(x)))] = 1
    result[df["Label"].astype(str).apply(lambda x: bool(normal_pattern.search(x)))] = 0

    # Numeric fallback (for numeric labels)
    numeric = pd.to_numeric(df["Label"], errors="coerce")
    result.loc[numeric.notna()] = (numeric.loc[numeric.notna()] >= 0.5).astype(int)

    # Drop NaN labels
    before = len(df)
    df["Label"] = result
    df = df.dropna(subset=["Label"])
    dropped = before - len(df)
    if dropped:
        print(f"[Label] Dropped {dropped:,} rows with undetermined Label")

    df["Label"] = df["Label"].astype(int)
    print("[Label] value counts:\n", df["Label"].value_counts())

    return df


# === PATH SETUP ===
fileTimeStamp, output_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir
os.makedirs(graph_dir, exist_ok=True)

# === LOAD DATA (Single Sensor) ===
try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), using direct DuckDB connection.")
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL AND SensorId = '{SELECTED_SENSOR_ID}'
"""
print(f"[Load] Running query for SensorId = {SELECTED_SENSOR_ID} ...")
df = con.sql(query).df()
print(f"[Load] rows: {len(df)}")

if df.empty:
    raise RuntimeError(f"No data found for SensorId = {SELECTED_SENSOR_ID}")

df = optimize_dataframe(df)
print(f"[Load] memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# === BINARY LABEL MAPPING ===
df = fast_label_to_binary(df)

print("\n[DEBUG] Label distribution after conversion for Sensor:", SELECTED_SENSOR_ID)
print(df["Label"].value_counts())
print("\n[DEBUG] Group by Label for Sensor:", SELECTED_SENSOR_ID)
print(df[["SensorId", "Label"]].groupby("Label").size())

# Safety check for binary class existence
if df["Label"].nunique() < 2:
    raise RuntimeError(f"Sensor {SELECTED_SENSOR_ID} contains only one class — cannot train classifier.")

# Drop irrelevant/missing features
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

# === ENCODE CATEGORICAL FEATURES ===
cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# === FEATURES ===
features = [col for col in ["Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
if not features:
    raise RuntimeError("No valid numeric features found in dataset!")

print(f"[Features] Using features ({len(features)}): {features}")

# === SPLIT TRAIN/TEST ===
X = df[features].fillna(df[features].mean())
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train = np.asarray(y_train).astype(int)
y_test = np.asarray(y_test).astype(int)

print("[Split] y_train distribution:", np.bincount(y_train))
print("[Split] y_test distribution:", np.bincount(y_test))

# === SCALING ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === DEFINE ENSEMBLE MODEL ===
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(max_depth=8, random_state=42))
]
meta = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=3,
    n_jobs=1,
    passthrough=False,
    verbose=1
)

# === TRAIN MODEL ===
print(f"\n[Train] Training stacking model for Sensor {SELECTED_SENSOR_ID} ...")
try:
    stack.fit(X_train_scaled, y_train)
    print("[Train] Done.")
except Exception:
    print("[Train] Exception during stack.fit():")
    traceback.print_exc()

# === EVALUATION ===
p_test = stack.predict_proba(X_test_scaled)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Evaluation for Sensor:", SELECTED_SENSOR_ID)
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

# === EXPORT TRAIN/TEST DATA ===
train_csv = os.path.join(output_dir, f"TrainData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
test_csv = os.path.join(output_dir, f"TestData_Sensor{SELECTED_SENSOR_ID}_{fileTimeStamp}.csv")
pd.concat([X_train, pd.Series(y_train, name="Label")], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, pd.Series(y_test, name="Label")], axis=1).to_csv(test_csv, index=False)
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

# === APPLY MODEL TO ALL DATA ===
df_scaled = scaler.transform(df[features])
df["PredictedProb"] = stack.predict_proba(df_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
df["PredictedRole"] = df["PredictedLabel"].apply(lambda x: "Botnet" if x == 1 else "Normal")

# === GRAPH VISUALIZATION ===
print(f"[Graph] Generating visualization for Sensor {SELECTED_SENSOR_ID} with {len(df)} edges...")

G = nx.from_pandas_edgelist(df, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

node_roles = {}
for addr in set(df["SrcAddr"]).union(df["DstAddr"]):
    subset = df[(df["SrcAddr"] == addr) | (df["DstAddr"] == addr)]
    avg_prob = subset["PredictedProb"].mean()
    if avg_prob > 0.85:
        node_roles[addr] = "C&C"
    elif avg_prob > 0.5:
        node_roles[addr] = "Bot"
    else:
        node_roles[addr] = "Normal"

pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)
edge_x, edge_y = [], []
for src, dst in G.edges():
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color="#AAA"), mode="lines", hoverinfo="none")
node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    role = node_roles.get(node, "Normal")
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Role: {role}")
    node_color.append("#007BFF" if role == "C&C" else "#FFB347" if role == "Bot" else "#CCCCCC")
    node_size.append(18 if role == "C&C" else 10)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    hovertext=node_text,
    hoverinfo="text",
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333")),
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Sensor {SELECTED_SENSOR_ID} - Botnet Network (Global Model)",
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

html_output = os.path.join(graph_dir, f"BotnetGraph_Sensor{SELECTED_SENSOR_ID}_GlobalModel_{fileTimeStamp}.html")
fig.write_html(html_output)
print(f"[Graph] Saved -> {html_output}")

print(f"\n✅ Done. Model trained and graph generated for Sensor {SELECTED_SENSOR_ID}.")
