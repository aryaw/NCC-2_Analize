import os
import duckdb
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import webbrowser
from libInternal import variableDump, getConnection, setFileLocation

fileTimeStamp, output_dir = setFileLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

graph_dir = output_dir
os.makedirs(graph_dir, exist_ok=True)

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), falling back to direct DuckDB connect.")
    con = duckdb.connect()

query = f"""
SELECT 
    SensorId,
    SrcAddr,
    DstAddr,
    TRIM(Dir) AS Dir,
    LOWER(Label) AS Label
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE StartTime IS NOT NULL
  AND Label IS NOT NULL
  AND (
        LOWER(Label) LIKE '%bot%' 
        OR LOWER(Label) LIKE '%flow=bot%' 
        OR LOWER(Label) LIKE '%botnet%'
      )
  AND TRIM(Dir) = '->'
  AND SrcAddr IS NOT NULL
  AND DstAddr IS NOT NULL;
"""

df = con.sql(query).df()
variableDump("Botnet Outgoing Src-Dst by SensorId", df.head())

if df.empty:
    print("No matching data found for botnet outgoing flows. Check Label and Dir values in your dataset.")
    exit(0)

unique_sensors = sorted(df["SensorId"].unique().tolist())
if not unique_sensors:
    print("No SensorId values found in filtered dataset. Exiting.")
    exit(0)

print(f"Detected SensorIds: {unique_sensors}")

for sensor_id in unique_sensors:
    df_sensor = df[df["SensorId"] == sensor_id]
    if df_sensor.empty:
        print(f"No botnet rows found for SensorId {sensor_id}, skipping.")
        continue

    G = nx.from_pandas_edgelist(df_sensor, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    degree_df = pd.DataFrame({
        "Node": list(G.nodes()),
        "OutDegree": [G.out_degree(n) for n in G.nodes()],
        "InDegree": [G.in_degree(n) for n in G.nodes()]
    })
    degree_df["Score"] = degree_df["OutDegree"] - degree_df["InDegree"]
    cc_candidate = degree_df.sort_values("Score", ascending=False).iloc[0]
    cc_node = cc_candidate["Node"]
    cc_out = cc_candidate["OutDegree"]
    cc_in = cc_candidate["InDegree"]

    print(f"Sensor {sensor_id}: Detected potential C&C bot → {cc_node} (out={cc_out}, in={cc_in})")

    pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.7, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

    for node, attr in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        out_d = G.out_degree(node)
        in_d = G.in_degree(node)
        role = "C&C BOT" if node == cc_node else "Client Bot"

        hovertext = (
            f"<b>Node:</b> {node}<br>"
            f"<b>Out-degree:</b> {out_d}<br>"
            f"<b>In-degree:</b> {in_d}<br>"
            f"<b>Role:</b> {role}"
        )
        node_text.append(hovertext)

        node_color.append('#007BFF' if node == cc_node else '#FFB347')
        node_size.append(20 if node == cc_node else 10)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#333')
        ),
        text=None
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Sensor {sensor_id} - Botnet Graph (Detected C&C: {cc_node})",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    html_output = os.path.join(graph_dir, f"NCC2_Sensor_{sensor_id}_Graph_{fileTimeStamp}_getRolebot.html")

    combined_html = f"""
    <html>
    <head>
        <title>Sensor {sensor_id} - NCC2 Botnet Network Graph</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body class="bg-white" style="font-family:Arial, sans-serif;">
        <div class="container mt-5 mb-5">
            <div class="card shadow mb-4 border-0">
                <div class="card-body text-center">
                    <h2 class="mb-3 text-dark">Sensor {sensor_id} - Botnet Outgoing Connections</h2>
                    <p class="text-muted">
                        Filtered where Label LIKE '%bot%' and Dir = '→'.<br>
                        <b>C&C Bot:</b> <span class="text-primary">{cc_node}</span>
                    </p>
                </div>
            </div>

            <div class="card shadow-sm mb-4 border-0">
                <div class="card-body">
                    {fig.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </div>

            <div class="text-center mt-4 mb-3 text-secondary small">
                Generated from NCC2 Dataset | {fileTimeStamp}
            </div>
        </div>
    </body>
    </html>
    """
