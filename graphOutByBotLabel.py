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

    # Build directed graph
    G = nx.from_pandas_edgelist(df_sensor, "SrcAddr", "DstAddr", create_using=nx.DiGraph())

    # Compute layout
    pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)

    # Extract edge coordinates
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Extract node coordinates
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            color=[len(list(G.neighbors(n))) for n in G.nodes()],
            size=10,
            colorbar=dict(
                title=dict(text='Connections', side='right'),
                xanchor='left'
            ),
            line_width=1
        ),
        text=None
    )

    # Build Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Sensor {sensor_id} - Botnet Outgoing Connections",
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

    # Save individual HTML file directly in output_dir
    html_output = os.path.join(graph_dir, f"NCC2_Sensor_{sensor_id}_Graph_{fileTimeStamp}.html")

    combined_html = f"""
    <html>
    <head>
        <title>Sensor {sensor_id} - NCC2 Botnet Network Graph</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body class="bg-light" style="font-family:Arial, sans-serif;">
        <div class="container mt-5 mb-5">
            <div class="card shadow mb-4">
                <div class="card-body text-center">
                    <h2 class="mb-3">Sensor {sensor_id} - Botnet Outgoing Connections</h2>
                    <p class="text-muted">Filtered where Label LIKE '%bot%' and Dir = '→'.</p>
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

    with open(html_output, "w", encoding="utf-8") as f:
        f.write(combined_html)

    print(f"Saved graph for SensorId {sensor_id}: {html_output}")

first_html = os.path.join(graph_dir, f"NCC2_Sensor_{unique_sensors[0]}_Graph_{fileTimeStamp}.html")
if os.path.exists(first_html):
    webbrowser.open(f"file://{os.path.abspath(first_html)}")

print(f"All graph files saved directly in: {graph_dir}")
