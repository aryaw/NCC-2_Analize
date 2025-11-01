# 🕵️‍♂️ NCC2 Botnet Flow Analysis

This project provides a complete workflow for **processing, cleaning, and analyzing botnet network flow data** from the **NCC2 dataset** (derived from CTU-13).  
It includes scripts to convert raw NetFlow data, clean it, extract schemas, and visualize botnet activity.

---

## 📁 Project Structure

```
project-root/
│
├── assets/
│   ├── dataset/
│   │   └── NCC2AllSensors_clean.csv   # Cleaned CSV dataset (output of cleanUpCSV.py)
│   ├── outputs/                       # Generated graphs, schema tables, etc.
│
├── convertCSV.py                      # Convert raw NetFlow to structured CSV
├── cleanUpCSV.py                      # Remove duplicate headers & invalid rows
├── getSchema.py                       # Extract schema summary to DataTable HTML
├── graphBotnetC2.py                   # Detect & visualize C&C bots by SensorId
├── requirements.txt                   # Python dependencies
├── libInternal.py                     # Common helper utilities (file location, DB connection, etc.)
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/aryaw/NCC-2_Analize.git
cd NCC-2_Analize
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # On Linux / macOS
# OR
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧩 Workflow Overview

### 1. **Convert raw NetFlow to CSV**
Script: `convertCSV.py`

This script reads the raw botnet flow logs (e.g., `.binetflow`, `.txt`, or `.log`) and converts them into a structured **CSV format**.

```bash
python convertCSV.py
```

**Output:**
```
assets/dataset/NCC2AllSensors.csv
```

---

### 2. **Clean the CSV (remove duplicate headers & invalid rows)**
Script: `cleanUpCSV.py`

Removes repeated header rows (common in large merged datasets) and ensures all columns align correctly.

```bash
python cleanUpCSV.py
```

**Output:**
```
assets/dataset/NCC2AllSensors_clean.csv
```

---

### 3. **Extract and visualize data schema**
Script: `getSchema.py`

Loads the cleaned CSV into **DuckDB**, extracts all column names and inferred types, and displays them in an interactive **DataTables** HTML report.

```bash
python getSchema.py
```

**Output:**
```
assets/outputs/NCC2Schema_<timestamp>.html
```

---

## 🌐 Preview All Exported Reports in a Web Viewer

Script: `main.py`

You can preview all exported HTML files (from `getSchema.py`, `graphBotnetC2.py`, etc.) using a built-in Flask dashboard.

### Run the Viewer
```bash
python main.py
```

### Open in Browser
```
http://127.0.0.1:5000/
```

### Features
- Sidebar automatically lists all `.html` files in `assets/outputs/`  
- Click any file to preview it instantly in the right panel  
- Auto-refreshes every 8 seconds to detect new exports  
- Clean white/blue-black UI for clarity and consistency  

---

## 🧠 Example Dataset Columns (NCC2 / CTU-13 Format)

| Column | Description |
|--------|--------------|
| `StartTime` | Flow start timestamp |
| `Dur` | Flow duration (seconds) |
| `Proto` | Protocol (TCP/UDP/ICMP) |
| `SrcAddr` | Source IP address |
| `Sport` | Source port |
| `Dir` | Direction of traffic (`->` or `<-`) |
| `DstAddr` | Destination IP address |
| `Dport` | Destination port |
| `State` | TCP state (e.g., `S_RA`, `SYN_SENT`) |
| `TotPkts` | Total packet count |
| `TotBytes` | Total byte count |
| `Label` | Flow label (botnet or normal) |
| `activityLabel` | Activity phase or encoded label |
| `bonetName` | Botnet family name (e.g., Neris, Rbot, Virut) |
| `sensorId` | ID of the sensor capturing this flow |

---

## 🧱 Dependencies

Listed in `requirements.txt`:

---

## 🧰 Virtual Environment Management

To **deactivate** your virtual environment:
```bash
deactivate
```

To **update** your dependencies:
```bash
pip freeze > requirements.txt
```

---

## 🧾 License

This project is for **research and academic use only**.  
Dataset attribution: NCC2 / CTU-13 Botnet Dataset (CTU University, Czech Republic).  
Ensure compliance with NCC2 dataset licensing terms.
