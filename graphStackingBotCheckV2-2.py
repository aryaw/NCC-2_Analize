import os
import gc
import re
import duckdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits=1):
        yield

from libInternal import (
    getConnection,
    setFileLocation,
    setExportDataLocation,
    optimize_dataframe,
    fast_label_to_binary,
)

RANDOM_STATE = 42
MAX_ROWS_FOR_STACKING = 1_000_000
SAFE_THREADS = "1"
os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
    "NUMEXPR_NUM_THREADS": SAFE_THREADS,
    "MKL_THREADING_LAYER": "GNU",   # harmless if MKL absent; helps avoid OMP clashes
    "JOBLIB_TEMP_FOLDER": "/tmp",
})

fileTimeStamp, output_dir = setFileLocation()
fileDataTimeStamp, outputdata_dir = setExportDataLocation()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")

# -------------------- load ----------------------
try:
    con = getConnection()
    print("Using connection from getConnection()")
except Exception:
    con = duckdb.connect()

query = f"""
SELECT SrcAddr, DstAddr, Proto, Dir, State, Dur, TotBytes, TotPkts,
       sTos, dTos, SrcBytes, Label, SensorId
FROM read_csv_auto('{csv_path}', sample_size=-1)
WHERE Label IS NOT NULL
"""

print("[Load] Reading dataset...")
df = con.sql(query).df()
if df.empty:
    raise RuntimeError("No labeled data found in dataset.")

df = optimize_dataframe(df)

# --- robust binary labels (uses your libInternal fast_label_to_binary) ---
df = fast_label_to_binary(df)
print(f"[Info] Loaded {len(df):,} flows across {df['SensorId'].nunique()} sensors")

# ---------------- preprocessing & features ----------------
df = df.dropna(subset=["SrcAddr", "DstAddr", "Dir", "Proto", "Dur", "TotBytes", "TotPkts", "Label"]).copy()

# Dir mapping
dir_map_num = {"->": 1, "<-": -1, "<->": 0}
df["Dir_raw"] = df["Dir"].astype(str).fillna("->")
df["Dir"] = df["Dir_raw"].map(dir_map_num).fillna(0).astype(int)

# categorical encodings
for c in ["Proto", "State"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# feature engineering
df["ByteRatio"]      = df["TotBytes"] / (df["TotPkts"] + 1)
df["DurationRate"]   = df["TotPkts"]  / (df["Dur"] + 0.1)
df["FlowIntensity"]  = df["SrcBytes"] / (df["TotBytes"] + 1)
df["PktByteRatio"]   = df["TotPkts"]  / (df["TotBytes"] + 1)
df["SrcByteRatio"]   = df["SrcBytes"] / (df["TotBytes"] + 1)
df["TrafficBalance"] = (df["sTos"] - df["dTos"]).abs()
df["DurationPerPkt"] = df["Dur"] / (df["TotPkts"] + 1)
df["Intensity"]      = df["TotBytes"] / (df["Dur"] + 1)

features = [
    "Dir", "Dur", "Proto", "TotBytes", "TotPkts", "sTos", "dTos", "SrcBytes",
    "ByteRatio", "DurationRate", "FlowIntensity", "PktByteRatio",
    "SrcByteRatio", "TrafficBalance", "DurationPerPkt", "Intensity"
]

# ---------------- split & (stable) scale -----------------
X_full = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_full = df["Label"].astype(int)

# sample guard for stacking memory
if len(df) > MAX_ROWS_FOR_STACKING:
    print(f"[Sample] Dataset too large ({len(df):,}), using {MAX_ROWS_FOR_STACKING:,} for model training...")
    df_sample = df.sample(n=MAX_ROWS_FOR_STACKING, random_state=RANDOM_STATE)
    X = df_sample[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_sample["Label"].astype(int)
else:
    X, y = X_full, y_full

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

# Keep float64 (more stable for HGB on some BLAS builds)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # float64
X_test_scaled  = scaler.transform(X_test)        # float64

# ------------------ training: stacking (passthrough=False) --------------
trained_model = None
print("\n[Train] Starting model training (Stacking, passthrough=False)...")

with threadpool_limits(limits=1):  # critical for stability on older CPUs
    try:
        base_learners = [
            ("rf", RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1, verbose=0
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=100, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=1, verbose=0
            )),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE, early_stopping=False
            )),
        ]

        meta = LogisticRegression(
            solver="lbfgs", max_iter=1000
        )

        stack = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            stack_method="predict_proba",
            passthrough=False,
            cv=2,
            n_jobs=1,
            verbose=1
        )

        stack.fit(X_train_scaled, y_train)
        trained_model = stack
        print("[Train] Stacking model trained successfully.")

    except Exception as e:
        print(f"[WARN] Stacking failed: {type(e).__name__}: {e}")
        print("[Fallback] Using RandomForest instead...")
        trained_model = RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        ).fit(X_train_scaled, y_train)
        print("[Fallback] RandomForest model trained.")

# ------------------ evaluation ------------------
p_test = trained_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thr_roc = roc_curve(y_test, p_test)
best_threshold = thr_roc[np.argmax(np.sqrt(tpr * (1 - fpr)))]
y_pred_test = (p_test >= best_threshold).astype(int)

print("\n# Global Model Evaluation")
print("Best threshold:", round(float(best_threshold), 4))
print("Accuracy:", round(float((y_pred_test == y_test).mean() * 100), 2), "%")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:",    recall_score(y_test, y_pred_test))
print("F1:",        f1_score(y_test, y_pred_test))
print("ROC-AUC:",   roc_auc_score(y_test, p_test))

# free split arrays
del X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, p_test
gc.collect()

# ------------------ full inference ------------------
X_all_scaled = scaler.transform(X_full)     # float64
df["PredictedProb"] = trained_model.predict_proba(X_all_scaled)[:, 1]
df["PredictedLabel"] = (df["PredictedProb"] >= best_threshold).astype(int)
del X_all_scaled, X_full, y_full
gc.collect()

# ------------------ per-sensor C&C detection ------------------
detected_summary = []
for sid in sorted(df["SensorId"].unique()):
    print(f"\n=== [Sensor {sid}] Auto C&C Detection ===")
    df_s = df[df["SensorId"] == sid].copy()

    # per-node stats (in/out)
    agg_in = df_s.groupby("DstAddr")["PredictedProb"].agg(["count", "mean"]) \
                 .rename(columns={"count": "in_ct", "mean": "in_prob"})
    agg_out = df_s.groupby("SrcAddr")["PredictedProb"].agg(["count", "mean"]) \
                  .rename(columns={"count": "out_ct", "mean": "out_prob"})
    stats = agg_in.join(agg_out, how="outer").fillna(0)

    stats["in_ratio"]  = stats["in_ct"]  / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["out_ratio"] = stats["out_ct"] / (stats["in_ct"] + stats["out_ct"] + 1e-9)
    stats["avg_prob"]  = (stats["in_prob"] + stats["out_prob"]) / 2
    stats["degree"]    = stats["in_ct"] + stats["out_ct"]

    # heuristic: high prob, high degree, directional skew
    node_roles = {}
    for n, r in stats.iterrows():
        if (r["avg_prob"] > 0.70) and (r["degree"] > 100) and ((r["in_ratio"] > 0.70) or (r["out_ratio"] > 0.70)):
            node_roles[n] = "C&C"
        else:
            node_roles[n] = "Normal"

    cc_nodes = [n for n, role in node_roles.items() if role == "C&C"]
    print(f"[Detected] {len(cc_nodes)} potential C&C nodes")

    if cc_nodes:
        cnc_df = stats.loc[cc_nodes].copy()
        cnc_df["SensorId"] = sid
        cnc_df["cnc_score"] = cnc_df["avg_prob"] * (1 + cnc_df["in_ratio"] + cnc_df["out_ratio"]) * np.log1p(cnc_df["degree"])
        cnc_df = cnc_df.sort_values("cnc_score", ascending=False)
        print(cnc_df.head(3)[["avg_prob", "degree", "in_ratio", "out_ratio", "cnc_score"]])
        detected_summary.append(cnc_df)
    else:
        print("[Info] No C&C nodes detected in this sensor.")

# ------------------ export ------------------
if detected_summary:
    summary_df = pd.concat(detected_summary, ignore_index=True)
    summary_csv = os.path.join(output_dir, f"CNC_AutoDetected_SafeAll_{fileTimeStamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[Export] Saved -> {summary_csv}")
else:
    print("\n[Summary] No C&C detected in any sensor.")

print("\nDone. Memory-safe stacking (passthrough=False) + per-sensor C&C detection complete.")
