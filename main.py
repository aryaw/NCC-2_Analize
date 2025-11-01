import os
import pandas as pd
from flask import Flask, render_template_string, send_from_directory, jsonify

# === Configuration ===
ASSET_FOLDER = os.path.join(os.getcwd(), "assets/outputs")
PORT = 5000

app = Flask(__name__)

pd.set_option('display.float_format', '{:,.0f}'.format)

# === HTML Template ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Botnet Detection Dashboard</title>

    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <link rel="stylesheet"
          href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">

    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            display: flex;
            height: 100vh;
            background-color: #f9fafb;
        }
        .sidebar {
            width: 300px;
            background-color: #1e293b;
            color: #e2e8f0;
            overflow-y: auto;
            padding: 20px;
        }
        .sidebar h2 {
            color: #22c55e;
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #16a34a;
            padding-bottom: 10px;
        }
        .file-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .file-list a {
            color: #e2e8f0;
            display: block;
            padding: 8px;
            border-radius: 6px;
            text-decoration: none;
        }
        .file-list a:hover {
            background-color: #22c55e;
            color: #fff;
        }
        .file-list a.active {
            background-color: #16a34a;
            color: #fff;
            font-weight: bold;
        }
        .main {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .header {
            background: linear-gradient(to right, #16a34a, #4ade80);
            color: white;
            padding: 12px 20px;
            font-size: 18px;
            font-weight: bold;
        }
    </style>

    <script>
        async function refreshFileList() {
            try {
                const response = await fetch('/list_files');
                const { files } = await response.json();

                const list = document.querySelector('.file-list');
                const current = document.querySelector('.file-list a.active')?.textContent;
                list.innerHTML = '';

                files.forEach(file => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.textContent = file;
                    a.target = 'preview_frame';

                    if (file.endsWith('.csv')) a.href = '/csv/' + file;
                    else a.href = '/view/' + file;

                    if (file === current) a.classList.add('active');

                    a.onclick = function() {
                        document.querySelectorAll('.file-list a')
                            .forEach(link => link.classList.remove('active'));
                        this.classList.add('active');
                    };

                    li.appendChild(a);
                    list.appendChild(li);
                });
            } catch (err) { console.error(err); }
        }

        setInterval(refreshFileList, 10000);
        window.onload = refreshFileList;
    </script>
</head>
<body>

    <div class="sidebar">
        <h2>Botnet Reports</h2>
        <ul class="file-list">
            {% for file in files %}
            <li>
                <a href="{{ '/csv/' + file if file.endswith('.csv') else '/view/' + file }}"
                   target="preview_frame"
                   onclick="document.querySelectorAll('.file-list a')
                                .forEach(el => el.classList.remove('active'));
                            this.classList.add('active');">
                   {{ file }}
                </a>
            </li>
            {% endfor %}
        </ul>
    </div>

    <div class="main">
        <div class="header">Botnet Detection Data Viewer</div>
        <iframe name="preview_frame"></iframe>
    </div>

</body>
</html>
"""

# === Flask Routes ===

@app.route('/')
def index():
    os.makedirs(ASSET_FOLDER, exist_ok=True)
    files = sorted(
        [f for f in os.listdir(ASSET_FOLDER) if f.endswith((".html", ".csv"))],
        reverse=True
    )
    return render_template_string(HTML_TEMPLATE, files=files)


@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory(ASSET_FOLDER, filename)

@app.route('/list_files')
def list_files():
    files = sorted(
        [f for f in os.listdir(ASSET_FOLDER) if f.endswith((".html", ".csv"))],
        reverse=True
    )
    return jsonify({"files": files})


if __name__ == "__main__":
    os.makedirs(ASSET_FOLDER, exist_ok=True)
    print(f"\nBotnet Detection Viewer running: http://127.0.0.1:{PORT}/")
    print(f"Serving from: {ASSET_FOLDER}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
