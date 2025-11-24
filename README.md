
# NCC‑2 Botnet Flow Analysis  
## **Stacking Ensemble + Directed Graph-Based C&C Detection**  

---

# Overview

This repository provides a complete **research-grade + developer-ready pipeline** for analyzing the  
**NCC‑2 Simultaneous Botnet Dataset**, detecting:

- **Botnet activity**
- **Command-and-Control (C&C) nodes**
- **Network graph propagation behavior**
- **Cross-sensor coordinated attacks**

The system combines:

### **Machine Learning (Stacking Ensemble)**  
- Random Forest  
- Extra Trees  
- Histogram Gradient Boosting  
- Logistic Regression meta-learner  
- Auto probability thresholding  
- Per-sensor recalculated C&C weights  

### **Directed Graph Analytics**  
- Node-level inbound/outbound aggregation  
- Auto CNC probability weighting  
- CNC scoring (degree × uniqueness × directionality × ML probability)  
- 3D interactive network graph visualization  
- Node role classification  

---

# Dataset Description - NCC‑2 Dataset

### **NCC‑2 Dataset: Simultaneous Botnet Dataset**  
**Published:** 8 September 2022  
**DOI:** 10.17632/8dpt85jrhp.2  
**Format:** Bidirectional network flows (binetflow)  
**Features:** 18 core NetFlow-like attributes  
**Key Property:** *Simultaneous multi-botnet attacks*

The dataset integrates:

- **CTU‑13 botnet scenarios**  
- **NCC periodic botnet activity**  

and merges them to simulate **high‑intensity parallel botnet activity**, where:

- multiple botnets execute attack phases *at the same time*
- several sensors detect similar behaviors across the same timestamp window  
- C&C traffic overlaps, creating realistic multi-source attack waves

### **Why Simultaneous Attacks Matter**
Unlike CTU‑13 (sporadic) or NCC (periodic), NCC‑2 includes:

- High-volume burst attacks  
- Parallel coordinated botnet operations  
- More challenging detection difficulty  
- Real-world approximation of DDoS command infrastructure

---

# Citation

If you use this repository or analysis pipeline, please cite:

```
@dataset{ncc2_2022,
  title={NCC-2 Dataset: Simultaneous Botnet Dataset},
  author={Azmi, Mohd Nizam and others},
  year={2022},
  doi={10.17632/8dpt85jrhp.2},
  publisher={Mendeley Data},
  version={2}
}
```

And dataset sources:

```
[1] CTU-13 Dataset - Czech Technical University.
[2] NCC Dataset - National Cybersecurity Centre.
[3] NCC-2 Dataset construction methodology.
```

---

# 📁 Repository Structure

```
Repo/
├── assets/
│   ├── dataset/
│   │   └── NCC2AllSensors_clean.csv
│   ├── outputs/
│   ├── outputsData/
│   └── present/
│
├── graphStackingBotCheckV2_25.py     # Main analysis engine
├── cleanUpCSV.py                     # Clean raw flows
├── convertCSV.py                     # Convert binetflow to CSV
├── getSchema.py                      # Generate schema HTML
├── libInternal/                      # Helper utilities
│
├── main.py                           # Local viewer for generated HTML
├── requirements.txt
└── README.md
```

---

# System Architecture

## **1. Preprocessing**
- Load dataset with DuckDB
- Convert labels → binary bot vs normal (`fast_label_to_binary`)
- Add `LabelCNC` from ground truth (`detect_cnc_from_label`)
- Generate engineered network features:
  - byte ratios
  - packet/byte density
  - duration‑normalized features
  - traffic direction encodings
  - intensity scoring

---

# Machine Learning - Stacking Ensemble

### **Model Structure**
- **Base models:**
  - `RandomForestClassifier`
  - `ExtraTreesClassifier`
  - `HistGradientBoostingClassifier`
- **Meta learner:**
  - Logistic Regression (`predict_proba` stack method)

### **Features Used**
16 network-traffic engineered features:
- Direction (`->`, `<-`, `<->`)
- Duration
- Protocol (encoded)
- Total packets / bytes
- ToS differences
- Byte ratios
- Flow intensity
- Duration per packet
- Traffic balance
- Packet-byte ratio

### **Adaptive Threshold**
A global threshold is computed via:

```
argmax sqrt(TPR × (1 − FPR))
```

This gives a **balanced global detector**, preventing false positives.

---

# Directed Graph-Based C&C Detection

For each SensorId:

## **1. Aggregation per node**
For each IP node:

```
in_ct,  in_prob   = groupby(DstAddr)
out_ct, out_prob  = groupby(SrcAddr)
unique peers       = {distinct inbound/outbound nodes}
```

## **2. Auto CNC Probability Weighting**
Instead of fixed 0.7/0.3, weight is auto‑calculated:

```
dominance = mean(out_prob) / (mean(out_prob)+mean(in_prob))
w_out = 0.5 + 0.5 * dominance
w_in  = 1 - w_out
```

This adapts to sensors with heavy outbound botnet traffic.

## **3. CNC Probability**
```
cnc_prob = out_prob*w_out + in_prob*w_in
```

## **4. CNC Score**
Combines graph + ML signals:

```
cnc_score =
   cnc_prob *
   (1 + out_ratio*1.8 + in_ratio*0.8) *
   log1p(degree) *
   log1p(out_unique_dests + 1)
```

## **5. Strict Rule Auto Threshold**
```
auto_strict_thr = mean(cnc_prob) + 1.5*std
bounded between [0.40, 0.95]
```

## **6. Percentile Rule**
Nodes with CNC Score ≥ 95th percentile are candidates.

## **7. Role Classification**
Labels node as:
- **C&C**
- **Normal**

## **8. Export to CSV per sensor**
- Full stats  
- CNC probability  
- CNC score  
- Role  
- Reason tags  

---

# 3D Graph Visualization

For each sensor, a 3D graph is generated:

- Red nodes → C&C
- Blue nodes → sampled normal nodes
- Directed edges drawn without overloading the graph
- Limits auto-adjust to available RAM
- Interactive Plotly HTML exported

---

# Running the Pipeline

### **Run full C&C detection + ML + graph generation**
```bash
python graphStackingBotCheckV2_25.py
```

### **Run local HTML viewer**
```bash
python main.py
```

---

# Requirements

Install:

```bash
pip install -r requirements.txt
```

This includes:

- DuckDB
- Scikit-learn
- NetworkX
- Plotly
- Pandas, NumPy
- psutil

---

# License
This project is for **academic and cybersecurity research only**.  
You must follow licensing requirements of the NCC‑2 dataset.