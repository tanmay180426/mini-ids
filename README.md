# Mini-IDS: Intelligent Network Security Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64-lightgrey.svg)](https://www.microsoft.com/windows)
[![Tests](https://img.shields.io/badge/Tests-125%20Passed-brightgreen.svg)](tests/)
[![Security](https://img.shields.io/badge/NIDS-Passive%20Hybrid%20Detection-orange.svg)](docs/architecture.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intelligent, hybrid Network Intrusion Detection System (NIDS) and Exploratory Security Analytics Platform built with Python, Scapy, scikit-learn, SQLite, and Streamlit.

Designed for Computer Science & Engineering final-year evaluations, technical reviews, and cybersecurity operations research, Mini-IDS integrates three interconnected disciplines:
1. **Cybersecurity & Network Forensics:** Stateful heuristic rule detection, raw packet dissection, bidirectional flow reconstruction, and timeline evidence correlation.
2. **Artificial Intelligence & Machine Learning:** Supervised classification (Random Forest, Logistic Regression, HistGradientBoosting) and unsupervised anomaly detection (Isolation Forest) with model explainability and defensive triage playbooks.
3. **Data Science & Traffic Analytics:** Statistical dataset profiling, correlation analysis, discrete interval traffic binning, and interactive exploratory data analysis (EDA).

---

## 1. System Architecture & Capabilities

```
+---------------------------------------------------------------------------------------+
|                                    TRAFFIC INGESTION                                  |
|                                                                                       |
|  +---------------------------+                     +-------------------------------+  |
|  |  Offline PCAP/PCAPNG File |                     |  Live Network Interface       |  |
|  |  (capture/packet_capture) |                     |  (capture/live_capture.py)    |  |
|  +-------------+-------------+                     +---------------+---------------+  |
+----------------|---------------------------------------------------|------------------+
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             PACKET NORMALIZATION LAYER                                |
|                                                                                       |
|   NormalizedPacket: timestamp, src_ip, dst_ip, src_port, dst_port, protocol,          |
|                     tcp_flags, length                                                 |
+------------------------------------------+--------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+--------------------------------------+      +-----------------------------------------+
|     STREAMING HEURISTIC ENGINE       |      |     BIDIRECTIONAL FLOW AGGREGATOR       |
|                                      |      |     (ml/feature_extraction.py)          |
|  - PortScanDetector (Windowed)       |      |                                         |
|  - SynRateDetector (Windowed)        |      |  - Canonical 5-Tuple Key Tracking       |
|  - TrafficAnomalyDetector (Vol.)     |      |  - 28 Tabular Statistical Features      |
+-------------------+------------------+      +--------------------+--------------------+
                    |                                              | Completed Flows
                    | Heuristic Alerts                             v
                    |                         +-----------------------------------------+
                    |                         |        MACHINE LEARNING PIPELINE        |
                    |                         |                                         |
                    |                         |  - Supervised Random Forest Classifier  |
                    |                         |  - Supervised Logistic Regression       |
                    |                         |  - Unsupervised Isolation Forest        |
                    |                         +--------------------+--------------------+
                    |                                              |
                    +----------------------+-----------------------+
                                           | Correlated Alerts & Flows
                                           v
+---------------------------------------------------------------------------------------+
|                             PERSISTENT SQLITE STORAGE                                 |
|                                                                                       |
|   Tables: packets | alerts | flows | models (WAL journal mode)                        |
+------------------------------------------+--------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                       STREAMLIT 9-PAGE SECURITY DASHBOARD                             |
|                                                                                       |
|  [1] Overview            [4] Traffic Intelligence     [7] Data Explorer               |
|  [2] Live Monitoring     [5] AI/ML Detection          [8] Detection Rules             |
|  [3] Alert Investigation [6] Model Laboratory         [9] System & Settings           |
+---------------------------------------------------------------------------------------+
```

---

## 2. Key Features

- **Hybrid Detection Engine (`detection/hybrid_detection.py`):** Correlates real-time packet heuristics with flow-level supervised predictions and unsupervised Isolation Forest outlier scores.
- **Leakage-Free ML Pipeline (`ml/`):** Full end-to-end scikit-learn pipeline featuring imputer and scaler transformers fitted strictly on training partitions, stratified and temporal splitting, and verifiable metadata manifests.
- **Model Registry & Validation (`ml/model_registry.py`):** Versioned model artifact bundles with dry-run schema validation and hot-swappable active model activation.
- **Model Explainability & Defensive Triage (`ml/explainability.py`):** Global feature importances, local flow threat attributions, and actionable incident response recommendations.
- **Restrained Professional UI (`analytics/theme.py`):** Custom warm-neutral aesthetic (`#F6F5F0` background, `#202C3B` slate sidebar, `#C76D45` terracotta accent) eliminating generic neon AI templates.
- **Offline & Live Capture:** Stream PCAP files or sniff live Windows interfaces via Npcap without dropping packets (buffered asynchronous background worker).
- **Sanitized Demonstration Mode (`storage/demo_data.py`):** Built-in deterministic synthetic telemetry utilizing RFC 5737 documentation IP prefixes (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) for cloud deployments and demonstrations without accessing live network adapters.

---

## 3. Real Benchmark Model Evaluation Results

All reported metrics are measured from genuine evaluation runs on held-out test records (`data/datasets/cicids2017/processed/test.csv`). None of these numbers are fabricated or simulated.

| Model Version | Algorithm | Task Type | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | ROC-AUC | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `random_forest_binary_1792500911` | Random Forest | Binary | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 0.0681 ms/sample |
| `logistic_regression_binary_1792500911` | Logistic Regression | Binary | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.0012 ms/sample** |
| `random_forest_multiclass_1792500921` | Random Forest | Multiclass | **99.59%** | **99.62%** | **99.59%** | **99.59%** | **1.0000** | 0.0636 ms/sample |
| `isolation_forest_binary_1792500912` | Isolation Forest | Anomaly | *Unsupervised* | *N/A* | *N/A* | *N/A* | *N/A* | 0.0205 ms/sample |

### Multiclass Confusion Matrix (Random Forest)
```
Classes: ['BENIGN', 'BruteForce', 'DoS/DDoS', 'PortScan']
[[144    1    0    0]    <- BENIGN
 [  0   12    0    0]    <- BruteForce
 [  0    0   36    0]    <- DoS/DDoS
 [  0    0    0   48]]   <- PortScan
```

---

## 4. Installation & Windows Setup

### 4.1 Prerequisites
1. **Python 3.10+ (Tested through Python 3.14):** Verify with `python --version`.
2. **Npcap for Windows (Required only for live interface capture):**
   - Download and install [Npcap](https://npcap.com/#download).
   - Ensure the option **"Install Npcap in WinPcap API-compatible Mode"** is checked.
   - *Note: Npcap is NOT required for offline PCAP analysis, model training, or demonstration mode.*

### 4.2 Clone & Virtual Environment Setup
Open PowerShell as Administrator (or standard user if live capture is not required):

```powershell
# 1. Enable execution of PowerShell scripts if needed
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 2. Clone repository and navigate to workspace
git clone https://github.com/tanmay180426/mini-ids.git
cd mini-ids

# 3. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Quick Start Guide

### 5.1 Launch the Streamlit Security Dashboard
```powershell
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser. The dashboard automatically detects available data or enables **Demonstration Mode** if no database exists yet.

### 5.2 Offline PCAP Analysis
To analyze a recorded network capture and persist alerts, packets, and flows:
```powershell
# Standard heuristic analysis
python main.py --pcap data/sample.pcap

# Hybrid analysis (Heuristics + Supervised ML + Isolation Forest Anomaly Detection)
python main.py --pcap data/sample.pcap --hybrid
```

### 5.3 Live Windows Packet Capture
```powershell
# List available network adapters discovered by Npcap
python main.py --list-interfaces

# Sniff packets live on interface (e.g. Wi-Fi) with count and timeout limits
python main.py --interface "Wi-Fi" --count 500 --timeout 30 --hybrid
```

---

## 6. Machine Learning Workflow Commands

Mini-IDS includes automated CLI scripts under `scripts/` to manage the entire ML lifecycle:

```powershell
# 1. Inspect existing datasets and local status
python scripts/inspect_datasets.py

# 2. Download benchmark datasets (or generate deterministic sample fixture)
python scripts/download_datasets.py --sample

# 3. Clean, preprocess, and partition datasets (stratified split)
python scripts/preprocess_datasets.py --dataset cicids2017 --split stratified

# 4. Train supervised classifiers and unsupervised anomaly models
python scripts/train_models.py --dataset cicids2017

# 5. Evaluate models on test partitions and generate JSON/Markdown reports
python scripts/evaluate_models.py

# 6. Validate integrity and schema compatibility of saved model bundles
python scripts/validate_artifacts.py
```

---

## 7. Running the Automated Test Suite

Mini-IDS includes 125 automated unit, boundary, integration, and headless Streamlit UI tests:

```powershell
# Run the full test suite
python -m pytest

# Run with concise summary
python -m pytest -q

# Run specific module test suites
python -m pytest tests/test_ml.py tests/test_hybrid.py tests/test_analytics.py
```

---

## 8. 9-Page Security Analytics Interface

The Streamlit dashboard provides 9 distinct, focused analytical workspaces:

1. **Overview:** Executive security posture, KPI scorecards, model readiness cards, alert distribution charts, and quick capture controls.
2. **Live Monitoring:** Interface selection, packet/duration limits, active sniffer telemetry, and real-time event streaming.
3. **Alert Investigation:** Deep forensic workbench with detection method filters, correlated raw packet inspection, and contextual defensive playbooks.
4. **Traffic Intelligence:** Protocol breakdowns, top destination ports, most active hosts, and discrete interval traffic throughput.
5. **AI/ML Detection:** Live flow inference console, comparison between heuristic rules vs ML predictions, and anomaly scoring diagnostics.
6. **Model Laboratory:** Registered model versions, evaluation metrics comparison table, feature importance ranking, and one-click model activation.
7. **Data Explorer:** Tabular dataset profiling, missing/infinite value health checks, class distribution charts, and Pearson correlation matrices.
8. **Detection Rules:** Configuration and threshold documentation for heuristic rules and ML confidence boundaries.
9. **System & Settings:** Database diagnostics, table row counts, schema status, log viewer, and demonstration mode toggle.

---

## 9. Limitations & Ethical Considerations

> [!CAUTION]
> **Defensive and Educational Scope Only:**
> Mini-IDS is designed strictly for authorized security monitoring, educational demonstrations, and defense research. It must not be deployed to monitor networks or hosts without explicit authorization.

- **Passive Detection Only:** Mini-IDS does not block traffic, alter firewall tables, or terminate TCP connections. Alerts represent correlated statistical telemetry rather than absolute confirmation of malicious intent.
- **Model Generalization:** Machine learning models trained on benchmark datasets (e.g. CICIDS2017) may experience performance degradation on divergent network environments due to distribution shift. Continuous retraining and heuristic baseline hybrid detection are recommended.
- **High-Throughput Limitations:** Python and Scapy are optimized for educational clarity and modularity. In gigabit enterprise production environments, dedicated kernel-bypass solutions (e.g. DPDK, PF_RING) and hardware-accelerated engines (e.g. Suricata, Zeek) are standard.

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
