import os
import duckdb
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from libInternal import variableDump, getConnection, setFileLocation, setExportDataLocation

# === Initialization ===
fileTimeStamp, output_dir = setFileLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
graph_dir = output_dir
os.makedirs(graph_dir, exist_ok=True)

# === Connect to DuckDB ===
try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), falling back to direct DuckDB connect.")
    con = duckdb.connect()

# === Load dataset ===
query = f"SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1)"
df = con.sql(query).df()

# === Basic preprocessing ===
df = df.dropna(subset=["SrcAddr", "DstAddr", "Proto", "Dur", "TotBytes", "TotPkts", "Label"])
df["Label"] = df["Label"].apply(lambda x: 1 if "bot" in x.lower() else 0)

# Encode categorical features
cat_cols = ["Proto", "Dir", "State"]
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# Select numeric features for training
features = ["Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes"]
features = [f for f in features if f in df.columns]
X = df[features].fillna(df[features].mean())
y = df["Label"]

# === Handle missing values ===
# Fill missing numeric columns with zero
X = X.fillna(0)

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Define stacking model ===
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("hgb", HistGradientBoostingClassifier(random_state=42))
]
meta_learner = LogisticRegression(max_iter=200)
stack = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=3, n_jobs=-1)

# === Train & evaluate ===
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Predict on full dataset ===
df["PredictedLabel"] = stack.predict(scaler.transform(X))
df["PredictedRole"] = df["PredictedLabel"].apply(lambda x: "Botnet" if x == 1 else "Normal")

# === Generate directed graph visualization per Sensor ===
unique_sensors = sorted(df["sensorId"].unique().tolist() if "sensorId" in df.columns else [0])
for sensor_id in unique_sensors[:3]:
    df_sensor = df[df["sensorId"] == sensor_id] if "sensorId" in df.columns else df
    if df_sensor.empty:
        continue

    # Build directed graph
    G = nx.from_pandas_edgelist(df_sensor, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    # Compute per-node role from predictions
    node_roles = {}
    for addr in set(df_sensor["SrcAddr"]).union(df_sensor["DstAddr"]):
        subset = df_sensor[(df_sensor["SrcAddr"] == addr) | (df_sensor["DstAddr"] == addr)]
        if subset["PredictedLabel"].mean() > 0.8:
            node_roles[addr] = "C&C"
        elif subset["PredictedLabel"].mean() > 0.5:
            node_roles[addr] = "Bot"
        else:
            node_roles[addr] = "Normal"

    pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)

    # Build edge traces
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.6, color="#BBB"), mode="lines", hoverinfo="none"
    )

    # Build node traces
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
        hovertext=node_text, hoverinfo="text",
        marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#333")),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Sensor {sensor_id} - Stacking ML Botnet Graph",
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

    # === Save HTML ===
    html_output = os.path.join(graph_dir, f"NCC2_Sensor_{sensor_id}_StackingGraph_{fileTimeStamp}.html")
    fig.write_html(html_output)
    print(f"Saved graph for Sensor {sensor_id}: {html_output}")

# === Auto-open the first graph ===
if os.path.exists(html_output):
    webbrowser.open(f"file://{html_output}")

print(f"All graphs and model outputs saved to: {output_dir}")
