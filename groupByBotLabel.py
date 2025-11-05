import os
import duckdb
import pandas as pd
import plotly.express as px
import webbrowser
from libInternal import variableDump, getConnection, setFileLocation

fileTimeStamp, output_dir = setFileLocation()
file_chart_html = os.path.join(output_dir, f"NCC2_FlowsHistogram_Sensor_{fileTimeStamp}.html")

csv_path = "/DataLab/botNCC2/NCC2AllSensors_clean.csv"

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), falling back to direct DuckDB connect.")
    con = duckdb.connect()

query = f"""
SELECT 
    date_trunc('minute', StartTime) AS TimeGroup,
    SensorId,
    Label
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE StartTime IS NOT NULL
  AND (lower(Label) LIKE '%bot%')
ORDER BY TimeGroup;
"""

df = con.sql(query).df()
variableDump("Filtered histogram grouped by SensorId", df.head())

unique_sensors = sorted(df["SensorId"].unique())[:3]

print(f"Detected SensorIds (showing 3): {unique_sensors}")

charts_html = ""

for sensor_id in unique_sensors:
    df_sensor = df[df["SensorId"] == sensor_id]

    fig = px.histogram(
        df_sensor,
        x="TimeGroup",
        nbins=100,
        title=f"Sensor {sensor_id} - Botnet Flows Over Time (Histogram)",
        labels={"TimeGroup": "Time"},
        template="plotly_white",
        color_discrete_sequence=["#D62728"]  # red for bot activity
    )

    fig.update_layout(
        xaxis_title="Start Time (Grouped by Minute)",
        yaxis_title="Botnet Flow Count",
        bargap=0.1,
        title_x=0.5,
        hovermode="x unified",
        height=400
    )

    charts_html += f"""
    <div class="col-md-6">
        <div class="card mb-4 shadow-sm">
            <div class="card-body">
                {fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </div>
    </div>
    """

combined_html = f"""
<html>
<head>
    <title>NCC2 Botnet Histogram by SensorId</title>
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
                <h2 class="mb-3">NCC2 Dataset - Botnet Flow Histogram by SensorId</h2>
                <p class="text-muted">
                    Showing flows where Label contains 'bot' (case-insensitive), grouped by minute.
                </p>
            </div>
        </div>

        <div class="row">
            {charts_html}
        </div>

        <div class="text-center mt-4 mb-3 text-secondary small">
            Generated from NCC2 Dataset | {fileTimeStamp}
        </div>
    </div>
</body>
</html>
"""

with open(file_chart_html, "w", encoding="utf-8") as f:
    f.write(combined_html)

print(f"Botnet histogram dashboard saved to: {file_chart_html}")
webbrowser.open(f"file://{os.path.abspath(file_chart_html)}")
