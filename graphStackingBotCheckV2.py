import os
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import webbrowser
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from libInternal import variableDump, getConnection, setFileLocation, setExportDataLocation

# Initialization
fileTimeStamp, output_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir
os.makedirs(graph_dir, exist_ok=True)

# Load dataset from DuckDB
try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), using direct DuckDB connection.")
    con = duckdb.connect()

query = f"SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1)"
df = con.sql(query).df()
print(f"Loaded dataset: {len(df)} rows")

# Data preprocessing
df = df.dropna(subset=["SrcAddr", "DstAddr", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])
df["Label"] = df["Label"].apply(lambda x: 1 if "bot" in x.lower() else 0)

cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

features = [col for col in ["Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"] if col in df.columns]
X = df[features].fillna(df[features].mean())
y = df["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
print("\n# Feature Selection")
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train_scaled, y_train)
feature_scores = pd.DataFrame({
    "Feature": features,
    "Score": selector.scores_,
}).sort_values(by="Score", ascending=False)
print(feature_scores)

top_features = feature_scores.head(5)["Feature"].tolist()
print(f"Top selected features: {top_features}")

selected_indices = [features.index(f) for f in top_features]
X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

# Export processed datasets
train_export = pd.concat([pd.DataFrame(X_train[top_features], columns=top_features), y_train.reset_index(drop=True)], axis=1)
test_export = pd.concat([pd.DataFrame(X_test[top_features], columns=top_features), y_test.reset_index(drop=True)], axis=1)
train_csv = os.path.join(output_dir, f"TrainData_{fileTimeStamp}.csv")
test_csv = os.path.join(output_dir, f"TestData_{fileTimeStamp}.csv")
train_export.to_csv(train_csv, index=False)
test_export.to_csv(test_csv, index=False)
print(f"Exported training data: {train_csv}")
print(f"Exported testing data: {test_csv}")

# Build and train stacking model
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ("hgb", HistGradientBoostingClassifier(random_state=42))
]

meta = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta,
    cv=5,
    n_jobs=-1
)

stack.fit(X_train_selected, y_train)

# Evaluate performance
p_test = stack.predict_proba(X_test_selected)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, p_test)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5

y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Stacking Classifier Evaluation (ExtraTrees Meta-Learner)")
print("Best threshold:", round(best_threshold, 3))
print("Accuracy:", round((y_pred_test == y_test).mean() * 100, 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, p_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, digits=4))

# Predict roles for visualization
df["PredictedProb"] = stack.predict_proba(scaler.transform(df[top_features]))[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
df["PredictedRole"] = df["PredictedLabel"].apply(lambda x: "Botnet" if x == 1 else "Normal")

# Directed graph visualization
unique_sensors = sorted(df["sensorId"].unique().tolist() if "sensorId" in df.columns else [0])

for sensor_id in unique_sensors[:3]:
    df_sensor = df[df["sensorId"] == sensor_id] if "sensorId" in df.columns else df
    if df_sensor.empty:
        continue

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
            title=f"Sensor {sensor_id} - Stacking (ExtraTrees Meta) Botnet Graph",
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
    print(f"Saved graph for Sensor {sensor_id} → {html_output}")
