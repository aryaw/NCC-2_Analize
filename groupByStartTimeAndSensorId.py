import os
import duckdb
import pandas as pd
import plotly.express as px
import webbrowser
from libInternal import variableDump, getConnection, setFileLocation

# === Initialization ===
fileTimeStamp, output_dir = setFileLocation()
file_chart_html = os.path.join(output_dir, f"NCC2_FlowsHistogram_Sensor_{fileTimeStamp}.html")

csv_path = "/home/arya/Documents/Pasca Stikom/BigData/Repo/DataLab/botNCC2/NCC2AllSensors_clean.csv"

# === Connect to DuckDB ===
try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), falling back to direct DuckDB connect.")
    con = duckdb.connect()

# === Query: extract StartTime and SensorId ===
query = f"""
SELECT 
    date_trunc('minute', StartTime) AS TimeGroup,
    SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE StartTime IS NOT NULL
ORDER BY TimeGroup;
"""

df = con.sql(query).df()
variableDump("Histogram grouped by SensorId", df.head())

# === Get unique SensorIds (limit to first 3 for visualization) ===
unique_sensors = df["SensorId"].unique()
unique_sensors = sorted(unique_sensors)[:3]  # only first 3 for clarity

print(f"Detected SensorIds (showing 3): {unique_sensors}")

# === Create histogram for each SensorId ===
charts_html = ""

for sensor_id in unique_sensors:
    df_sensor = df[df["SensorId"] == sensor_id]

    fig = px.histogram(
        df_sensor,
        x="TimeGroup",
        nbins=100,
        title=f"Sensor {sensor_id} - Network Flows Over Time (Histogram)",
        labels={"TimeGroup": "Time"},
        template="plotly_white",
        color_discrete_sequence=["#4C78A8"]
    )

    fig.update_layout(
        xaxis_title="Start Time (Grouped by Minute)",
        yaxis_title="Flow Count",
        bargap=0.1,
        title_x=0.5,
        hovermode="x unified",
        height=400
    )

    charts_html += f"""
    <div class="card mb-4 shadow-sm">
        <div class="card-body">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
    """

# === Build Bootstrap HTML layout ===
combined_html = f"""
<html>
<head>
    <title>NCC2 Histogram by SensorId</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Bootstrap -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">

    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-light" style="font-family:Arial, sans-serif;">
    <div class="container mt-5 mb-5">
        <div class="card shadow mb-4">
            <div class="card-body text-center">
                <h2 class="mb-3">NCC2 Dataset - Histogram by SensorId</h2>
                <p class="text-muted">Flow distribution across time intervals, grouped by minute, for each SensorId.</p>
            </div>
        </div>
        {charts_html}
        <div class="text-center mt-4 mb-3 text-secondary small">
            Generated from NCC2 Dataset | {fileTimeStamp}
        </div>
    </div>
</body>
</html>
"""

# === Save and open ===
with open(file_chart_html, "w", encoding="utf-8") as f:
    f.write(combined_html)

print(f"Histogram dashboard saved to: {file_chart_html}")
webbrowser.open(f"file://{os.path.abspath(file_chart_html)}")
