import os
import duckdb
import pandas as pd
import webbrowser
from libInternal import variableDump, getConnection, setFileLocation

fileTimeStamp, output_dir = setFileLocation()
file_chart_svg = os.path.join(output_dir, f"NCC2Schema_{fileTimeStamp}.html")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception as e:
    print(f"Warning: getConnection() failed ({e}), falling back to direct DuckDB connect.")
    con = duckdb.connect()

schema_df = con.sql(f"""
DESCRIBE SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1)
""").df()

schema_df = schema_df[['column_name', 'column_type']].rename(
    columns={'column_name': 'Column', 'column_type': 'Type'}
)

variableDump("Schema DataFrame", schema_df.head())

schema_html_table = schema_df.to_html(
    classes='table table-striped table-bordered table-hover', 
    index=False, 
    table_id='schemaTable', 
    escape=False
)

combined_html = f"""
<html>
<head>
    <title>NCC2 Schema Table</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">

    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">

    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>

    <!-- Initialize DataTable -->
    <script>
    $(document).ready(function() {{
        $('#schemaTable').DataTable({{
            paging: true,
            searching: true,
            ordering: true,
            pageLength: 15,
            lengthChange: false,
            info: true
        }});
    }});
    </script>
</head>
<body class="bg-light" style="font-family: Arial, sans-serif;">
    <div class="container mt-5 mb-5">
        <div class="card shadow">
            <div class="card-body">
                <h2 class="text-center mb-4">NCC2 Dataset Schema</h2>
                <p class="text-muted text-center">
                    This table displays all detected columns and their data types from the NCC2 CSV dataset.
                </p>
                <hr>
                {schema_html_table}
            </div>
        </div>
        <div class="text-center mt-4 mb-3 text-secondary small">
            Generated from DuckDB | {fileTimeStamp}
        </div>
    </div>
</body>
</html>
"""

with open(file_chart_svg, "w", encoding="utf-8") as f:
    f.write(combined_html)

print(f"HTML schema report saved to: {file_chart_svg}")
webbrowser.open(f"file://{os.path.abspath(file_chart_svg)}")
