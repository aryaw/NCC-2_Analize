# NCC‑2 Botnet Flow Analysis

## **Stacking Ensemble + Directed Graph-Based C&C Detection**

---

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

---

## Abstract

This work presents a fully integrated, research‑grade pipeline for analyzing the NCC‑2 Simultaneous Botnet Dataset through a dual‑layer methodology that combines **stacking ensemble machine learning** with **directed, topology‑aware graph analytics**. The pipeline is designed to bridge the gap between flow‑level behavioral modeling and node‑level relational inference, enabling the detection of Command‑and‑Control (C&C) infrastructures embedded within large‑scale, multi‑sensor network environments.

The system incorporates rigorous data preprocessing (including regex‑based flow reconstruction), comprehensive feature engineering, adaptive model evaluation, and sensor‑specific probability weighting to enhance detection robustness. In addition, a graph‑driven scoring mechanism captures structural anomalies such as degree imbalance, peer uniqueness, and weighted communication patterns. The resulting framework supports both automated C&C candidate identification and interactive 3D network visualization, providing an analysis workflow suitable for academic research, operational threat hunting, and reproducible scientific studies.

## 1. Introduction

Detecting C&C infrastructure in network traffic is challenging because machine‑learning signals often focus on the **flow level**, whereas C&C behavior is inherently **node‑level and relational**. This pipeline bridges that gap by combining per‑flow ML predictions with graph‑derived structural features (degree, uniqueness, edge weights) to produce node-level C&C detection.

Primary objectives:

* Build a robust botnet detector capable of identifying simultaneous, high‑volume attacks.
* Provide per‑sensor C&C node candidates backed by ML probabilities and graph‑based evidence.

---

## 2. Data Architecture and Conversion

### 2.1 Data Source

The core dataset is the cleaned CSV file `NCC2AllSensors_clean.csv`, containing bidirectional NetFlow-like records.

### 2.2 Regex-Based Conversion

Original NCC‑2 files may come from binetflow or structured text. A regex-based parsing phase extracts relevant information:

1. Pattern extraction for IPs, ports, states, and metrics.
2. Cleaning malformed or non-numeric values.
3. Filtering valid IPv4 addresses using `^[0-9.]+$`.

DuckDB’s `read_csv_auto` is used for large-scale ingestion while maintaining correctness through regex validation.

---

## 3. Preprocessing and Labeling

1. Removal of rows with missing critical fields.
2. Injection of ground-truth C&C labels via `detect_cnc_from_label`.
3. Memory reduction using type downcasting.
4. Binary label conversion (normal = 0, botnet = 1).

---

## 4. Feature Engineering

Focusing on both **flow-level and graph-aware features**.

### 4.1 Core Flow Features

Includes direction, protocol, duration, packet counts, byte counts, and state values.

### 4.2 Ratio & Intensity Features

Designed to normalize traffic patterns:

* ByteRatio
* DurationRate
* FlowIntensity
* PktByteRatio
* SrcByteRatio
* TrafficBalance
* DurationPerPkt
* Intensity

### 4.3 Graph-Aware Features

* EdgeWeight: number of flows between a source and destination.
* SrcTotalWeight, DstTotalWeight: total degree weights for each node.

---

## 5. Sampling, Scaling, and Train/Test Split

* Sampling if dataset exceeds limits.
* Stratified 70/30 split.
* StandardScaler normalization.

---

## 6. Model Architecture and Training

### 6.1 Stacking Model

Base learners:

* RandomForest
* ExtraTrees
* HistGradientBoosting

Meta learner:

* Logistic Regression

### 6.2 Motivation

Tree-based models capture nonlinear flow behaviors, while the meta-learner integrates probabilistic outputs for stability.

### 6.3 Fallback

If stacking fails, a high-capacity RandomForest is used.

---

## 7. Model Thresholding

### 7.1 Adaptive Global Threshold

Chosen by maximizing:

sqrt( TPR × (1 − FPR) )

### 7.2 Purpose

Balances sensitivity and specificity under imbalanced conditions.

---

## 8. Per-Sensor Aggregation & C&C Candidate Selection

Node-level metrics include:

* inbound/outbound counts and probabilities
* unique source/destination counts
* total graph weights
* degree, ratios

### 8.1 Adaptive Weighting

Weights for inbound/outbound probabilities adjusted per sensor.

### 8.2 C&C Probability

cnc_prob = out_prob * w_out + in_prob * w_in

### 8.3 C&C Scoring

Combines cnc_prob with structural metrics such as degree and peer uniqueness.

---

## 9. Candidate Rules

Strict rule:

* cnc_prob > mean + 1.5×std
* out_ct > 120
* out_ratio > 0.7

Percentile rule:

* cnc_score ≥ 95th percentile
* cnc_prob ≥ 0.40
* out_ct ≥ 20

Final candidate set = union of strict, percentile, and ground-truth nodes.

---

## 10. 3‑D Graph Construction

* Directed weighted graph built per sensor.
* Node set expanded with 1‑hop neighbors.
* Memory‑adaptive node/edge limits.

### Rendering

* spring_layout (3‑D)
* Red: top C&C nodes
* Yellow: other C&C candidates
* Blue: normal nodes
* Interactive Plotly HTML output

---

## 11. Evaluation

Metrics:

* Precision
* Recall
* F1-score
* ROC-AUC

---

## 12. Reproducibility

* Fixed random seeds
* Thread limiting
* Full output export to `outputs/` and `outputsData/`

---

## 13. Limitations & Future Work

Limitations:

* Hard dependency on ground-truth C&C labels
* No temporal modeling
* Vulnerable to low-and-slow encrypted attacks

Future work:

* Temporal graph neural networks
* Transformer/LSTM sequence modeling per node

---
