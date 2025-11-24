import os
import pandas as pd
from flask import Flask, render_template_string, send_from_directory, jsonify
from markdown import markdown

OUTPUT_FOLDER = os.path.join(os.getcwd(), "assets/outputs")
PRESENT_FOLDER = os.path.join(os.getcwd(), "assets/present")
PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__)
pd.set_option('display.float_format', '{:,.0f}'.format)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Botnet Detection Dashboard</title>

  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">

  <style>
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background-color: #f9fafb;
    }

    .layout {
      display: flex;
      height: 100vh;
      transition: margin-left 0.3s ease;
    }

    /* === Desktop Sidebar === */
    .sidebar {
      width: 260px;
      background: linear-gradient(180deg, #16a34a, #22c55e);
      color: #f1f5f9;
      overflow-y: auto;
      flex-shrink: 0;
      transition: width 0.3s ease;
      position: relative;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .sidebar.collapsed {
      width: 70px;
    }

    .sidebar h2 {
      color: #fff;
      text-align: center;
      margin: 20px 10px;
      border-bottom: 2px solid #fff;
      padding-bottom: 10px;
      transition: opacity 0.3s ease;
      white-space: nowrap;
      overflow: hidden;
    }

    .sidebar.collapsed h2 {
      opacity: 0;
    }

    .file-list {
      flex-grow: 1;
    }

    .file-list a {
      color: #f1f5f9;
      display: block;
      padding: 10px 18px;
      border-radius: 6px;
      text-decoration: none;
      transition: padding 0.3s ease, background 0.2s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .file-list a:hover {
      background-color: #15803d;
      color: #fff;
    }

    .file-list a.active {
      background-color: #166534;
      color: #fff;
      font-weight: bold;
    }

    /* === Sidebar toggle (bottom, seamless) === */
    .toggle-btn {
      background: rgba(0, 0, 0, 0.1);
      border-top: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 12px;
      text-align: center;
      cursor: pointer;
      font-size: 18px;
      transition: background 0.3s ease, transform 0.3s ease;
    }

    .toggle-btn:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    .sidebar.collapsed .toggle-btn {
      transform: rotate(180deg);
    }

    /* === Header === */
    .header {
      background: linear-gradient(to right, #16a34a, #4ade80);
      color: white;
      padding: 12px 20px;
      font-size: 18px;
      font-weight: bold;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: relative;
      z-index: 10;
    }

    .header button {
      background: transparent;
      border: none;
      color: white;
      font-size: 26px;
      line-height: 1;
      padding: 4px 10px;
      border-radius: 6px;
      transition: background 0.2s, transform 0.1s;
    }
    .header button:hover {
      background: rgba(255, 255, 255, 0.15);
    }
    .header button:active {
      transform: scale(0.9);
    }

    iframe {
      width: 100%;
      height: calc(100vh - 55px);
      border: none;
    }

    /* === Mobile Sidebar === */
    .mobile-sidebar {
      position: fixed;
      top: 0;
      left: -280px;
      width: 280px;
      height: 100%;
      background: linear-gradient(180deg, #16a34a, #22c55e);
      color: #f1f5f9;
      overflow-y: auto;
      transition: left 0.3s ease;
      z-index: 20;
    }

    .mobile-sidebar.open {
      left: 0;
    }

    .mobile-sidebar h2 {
      text-align: center;
      color: #fff;
      margin: 20px;
      border-bottom: 2px solid #fff;
      padding-bottom: 10px;
    }

    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.35);
      opacity: 0;
      visibility: hidden;
      transition: opacity 0.3s ease;
      z-index: 15;
    }

    .overlay.show {
      opacity: 1;
      visibility: visible;
    }

    /* Hide desktop sidebar on small screens */
    @media (max-width: 768px) {
      .sidebar { display: none; }
      .toggle-btn { display: none; }
    }
  </style>

  <script>
    async function refreshFileList() {
      try {
        const response = await fetch('/list_files');
        const { files } = await response.json();
        const lists = document.querySelectorAll('.file-list');
        lists.forEach(list => {
          const current = list.querySelector('a.active')?.textContent;
          list.innerHTML = '';
          files.forEach(file => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.textContent = file;
            a.target = 'preview_frame';
            if (file.endsWith('.csv')) a.href = '/csv/' + file;
            else if (file.endsWith('.md')) a.href = '/md/' + file;
            else a.href = '/view/' + file;
            if (file === current) a.classList.add('active');

            a.onclick = function() {
              document.querySelectorAll('.file-list a')
                .forEach(link => link.classList.remove('active'));
              this.classList.add('active');
              closeMobileMenu();
            };

            li.appendChild(a);
            list.appendChild(li);
          });
        });
      } catch (err) { console.error(err); }
    }

    function toggleMobileMenu() {
      const sidebar = document.querySelector('.mobile-sidebar');
      const overlay = document.querySelector('.overlay');
      sidebar.classList.toggle('open');
      overlay.classList.toggle('show');
    }

    function closeMobileMenu() {
      document.querySelector('.mobile-sidebar').classList.remove('open');
      document.querySelector('.overlay').classList.remove('show');
    }

    function toggleDesktopSidebar() {
      const sidebar = document.querySelector('.sidebar');
      sidebar.classList.toggle('collapsed');
    }

    setInterval(refreshFileList, 10000);
    window.onload = refreshFileList;
  </script>
</head>
<body>
  <div class="header">
    <button class="d-md-none" type="button" onclick="toggleMobileMenu()">☰</button>
    <span>Botnet Detection Data Viewer</span>
  </div>

  <div class="overlay" onclick="closeMobileMenu()"></div>

  <!-- Mobile Sidebar -->
  <div class="mobile-sidebar d-md-none">
    <button class="btn-close position-absolute end-0 m-3" onclick="closeMobileMenu()"></button>
    <h2>Botnet Reports</h2>
    <ul class="file-list list-unstyled">
      {% for file in files %}
      <li>
        <a href="{{ '/csv/' + file if file.endswith('.csv')
                    else '/md/' + file if file.endswith('.md')
                    else '/view/' + file }}"
           target="preview_frame">{{ file }}</a>
      </li>
      {% endfor %}
    </ul>
  </div>

  <div class="layout">
    <!-- Desktop Sidebar -->
    <div class="sidebar d-none d-md-flex flex-column">
      <div>
        <h2>Botnet Reports</h2>
        <ul class="file-list list-unstyled">
          {% for file in files %}
          <li>
            <a href="{{ '/csv/' + file if file.endswith('.csv')
                        else '/md/' + file if file.endswith('.md')
                        else '/view/' + file }}"
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
      <div class="toggle-btn d-none d-md-block" onclick="toggleDesktopSidebar()">❮</div>
    </div>

    <!-- Main content -->
    <div class="main flex-grow-1">
      <iframe name="preview_frame"></iframe>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


@app.route('/')
def index():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PRESENT_FOLDER, exist_ok=True)

    files = []
    for folder in [OUTPUT_FOLDER, PRESENT_FOLDER]:
        files += [f for f in os.listdir(folder) if f.endswith((".html", ".csv", ".md"))]

    files = sorted(files, reverse=True)
    return render_template_string(HTML_TEMPLATE, files=files)

@app.route('/md/<path:filename>')
def view_markdown(filename):
    md_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(md_path):
        md_path = os.path.join(PRESENT_FOLDER, filename)

    if not os.path.exists(md_path):
        return f"<h3>File not found: {filename}</h3>", 404

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_content = markdown(
        md_content,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "codehilite",
            "nl2br",
            "sane_lists"
        ]
    )

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>{filename}</title>

        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-dark.min.css">

        <style>
            body {{
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 40px;
                display: flex;
                justify-content: center;
            }}
            .markdown-body {{
                box-sizing: border-box;
                min-width: 200px;
                max-width: 980px;
                width: 100%;
                background: #0d1117;
                border-radius: 12px;
                padding: 40px;
                color: #c9d1d9;
            }}
            h1, h2, h3, h4 {{
                color: #3fb950;
                margin-top: 1.5em;
                font-weight: bold;
            }}
            pre, code {{
                background-color: #161b22 !important;
                color: #e6edf3;
                border-radius: 6px;
                padding: 10px;
                display: block;
                overflow-x: auto;
                font-size: 0.9rem;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 1em;
            }}
            table, th, td {{
                border: 1px solid #30363d;
                padding: 8px;
            }}
            th {{
                background: #161b22;
                color: #58a6ff;
                font-weight: bold;
            }}
            a {{
                color: #58a6ff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
                color: #79c0ff;
            }}
            hr {{
                border-color: #21262d;
            }}
            blockquote {{
                border-left: 4px solid #30363d;
                padding-left: 15px;
                color: #8b949e;
                margin: 1em 0;
            }}
        </style>
    </head>
    <body>
        <article class="markdown-body">
            {html_content}
        </article>
    </body>
    </html>
    """
    return html_template

@app.route('/view/<path:filename>')
def view_file(filename):
    html_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(html_path):
        html_path = os.path.join(PRESENT_FOLDER, filename)

    if not os.path.exists(html_path):
        return f"<h3>File not found: {filename}</h3>", 404

    return send_from_directory(os.path.dirname(html_path), os.path.basename(html_path))


@app.route('/list_files')
def list_files():
    files = []
    for folder in [OUTPUT_FOLDER, PRESENT_FOLDER]:
        files += [f for f in os.listdir(folder) if f.endswith((".html", ".csv", ".md"))]
    files = sorted(files, reverse=True)
    return jsonify({"files": files})

@app.route('/healthz')
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PRESENT_FOLDER, exist_ok=True)

    print(f"\nBotnet Detection Viewer running: http://127.0.0.1:{PORT}/")
    print(f"Serving from: {OUTPUT_FOLDER} and {PRESENT_FOLDER}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
