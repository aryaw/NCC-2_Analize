import os
import gc
import joblib
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from libInternal import variableDump, getConnection, setFileLocation, setExportDataLocation

# job limit
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def optimize_dataframe(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def to_binary_label(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        return 1 if "bot" in x.lower() else 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if x in (0.0, 1.0):
            return int(x)
        return int(x >= 0.5)
    return 0

fileTimeStamp, output_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir
os.makedirs(graph_dir, exist_ok=True)

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), using direct DuckDB connection.")
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts, sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL
"""
df = con.sql(query).df()
print(f"[Load] rows: {len(df)}")

df = optimize_dataframe(df)
print(f"[Load] memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

df["Label"] = df["Label"].apply(to_binary_label)
before = len(df)
df = df.dropna(subset=["Label"])
dropped = before - len(df)
if dropped:
    print(f"[Label] Dropped {dropped} rows with undetermined Label")

df["Label"] = df["Label"].astype(int)
print("[Label] value counts:\n", df["Label"].value_counts())

if df["Label"].nunique() < 2:
    raise RuntimeError("Less than 2 classes present — cannot train classifier.")

df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])

cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

features = [col for col in ["Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
if not features:
    raise RuntimeError("No numeric features found in dataset!")

X = df[features].fillna(df[features].mean())
y = df["Label"]

print(f"[Features] Using features ({len(features)}): {features}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train = np.asarray(y_train).astype(int)
y_test = np.asarray(y_test).astype(int)

print("[Split] y_train unique:", np.unique(y_train), "count:", np.bincount(y_train))
print("[Split] y_test unique:", np.unique(y_test), "count:", np.bincount(y_test))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n# Feature Selection (chi2)")
selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X_train_scaled, y_train)

feature_scores = pd.DataFrame({
    "Feature": features,
    "Score": selector.scores_
}).sort_values(by="Score", ascending=False)
print(feature_scores)

top_features = feature_scores.head(5)["Feature"].tolist()
print("[FeatureSelection] Top features:", top_features)

selected_indices = [features.index(f) for f in top_features]
X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

train_export = pd.concat([
    pd.DataFrame(X_train[top_features], columns=top_features),
    pd.Series(y_train, name="Label").reset_index(drop=True)
], axis=1)
test_export = pd.concat([
    pd.DataFrame(X_test[top_features], columns=top_features),
    pd.Series(y_test, name="Label").reset_index(drop=True)
], axis=1)

train_csv = os.path.join(output_dir, f"TrainData_{fileTimeStamp}.csv")
test_csv = os.path.join(output_dir, f"TestData_{fileTimeStamp}.csv")
train_export.to_csv(train_csv, index=False, float_format="%.4f")
test_export.to_csv(test_csv, index=False, float_format="%.4f")
print(f"[Export] train -> {train_csv}")
print(f"[Export] test  -> {test_csv}")

base_learners = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=1)),
    ("hgb", HistGradientBoostingClassifier(max_depth=8, random_state=42))
]

meta = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)

unique, counts = np.unique(y_train, return_counts=True)
min_class = counts.min()
cv_splits = min(5, min_class) if min_class >= 2 else 2
cv_splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

# change stackclass
stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    # cv=cv_splitter,
    cv=3,
    n_jobs=1,
    passthrough=False,
    verbose=1
).set_params(memory="/tmp/joblib_cache")
print("\n[Train] Fitting stacking classifier...")
try:
    stack.fit(X_train_selected, y_train)
    print("[Train] Done.")
except Exception:
    print("[Train] Exception during stack.fit():")
    traceback.print_exc()

# use voting classifier
# print("\n[Train] Fitting soft voting classifier (lightweight version)...")
# voting_stack = VotingClassifier(
#     estimators=base_learners + [("extra", meta)],
#     voting="soft",
#     n_jobs=1
# )
# voting_stack.fit(X_train_selected, y_train)
# print("[Train] Done.")

p_test = stack.predict_proba(X_test_selected)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5

y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Stacking Classifier Evaluation")
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test, digits=4))

df_scaled = scaler.transform(df[features])
df_selected = df_scaled[:, selected_indices]

df["PredictedProb"] = stack.predict_proba(df_selected)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
df["PredictedRole"] = df["PredictedLabel"].apply(lambda x: "Botnet" if x == 1 else "Normal")

unique_sensors = sorted(df["SensorId"].unique().tolist() if "SensorId" in df.columns else [0])
MAX_EDGES = 1000

for sensor_id in unique_sensors[:3]:
    df_sensor = df[df["SensorId"] == sensor_id] if "SensorId" in df.columns else df
    if df_sensor.empty:
        continue
    if len(df_sensor) > MAX_EDGES:
        df_sensor = df_sensor.sample(MAX_EDGES, random_state=42)
    G = nx.from_pandas_edgelist(df_sensor, "SrcAddr", "DstAddr", create_using=nx.DiGraph())
    node_roles = {}
    for addr in set(df_sensor["SrcAddr"]).union(df_sensor["DstAddr"]):
        subset = df_sensor[(df_sensor["SrcAddr"] == addr) | (df_sensor["DstAddr"] == addr)]
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
            title=f"Sensor {sensor_id} - Stacking Botnet Graph",
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
    html_output = os.path.join(graph_dir, f"NCC2_Sensor_{sensor_id}_StackingGraph_{fileTimeStamp}.html")
    fig.write_html(html_output)
    print(f"[Graph] Saved sensor {sensor_id} -> {html_output}")

print("\n✅ Done.")
