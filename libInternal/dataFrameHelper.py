import os
import re
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from typing import Tuple

def optimize_dataframe(df):
    """downcast numeric columns to save memory."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def fast_label_to_binary(df):
    """Convert 'Label' column to binary (1=bot/attack/malicious, 0=normal/benign) using regex matching."""
    labels_str = df["Label"].astype(str).str.lower().fillna("")

    bot_pattern = re.compile(
        r"\b(bot|botnet|cnc|c&c|malware|infected|attack|spam|ddos|trojan|worm|zombie|backdoor)\b",
        re.IGNORECASE,
    )
    normal_pattern = re.compile(
        r"\b(normal|benign|background|legit|clean|regular|safe|harmless)\b",
        re.IGNORECASE,
    )

    # set as result = [key, value]
    result = pd.Series(np.nan, index=df.index)
        
    def classify_label(label):
        text = str(label)
        if bot_pattern.search(text):
            return 1
        else:
            return 0
    result = df["Label"].apply(classify_label)

    # convert a column to numeric, numeric fallback to NaN, only takes numeric data
    numeric = pd.to_numeric(df["Label"], errors="coerce")
    # check if value ≥ 0.5 = 1
    result.loc[numeric.notna()] = (numeric.loc[numeric.notna()] >= 0.5).astype(int)

    # drop NaN
    before = len(df)
    df["Label"] = result
    df = df.dropna(subset=["Label"])
    dropped = before - len(df)
    if dropped:
        print(f"[Label] Dropped {dropped:,} rows with undetermined Label")
    df["Label"] = df["Label"].astype(int)
    print("[Label] value counts:\n", df["Label"].value_counts())
    return df

def detect_cnc_from_label(df):
    """
    Detect potential C&C
    - From-Botnet
    - To-Botnet
    - Botnet
    """
    labels = df["Label"].astype(str).str.lower()

    df["LabelFromBotnet"] = labels.str.contains(r"from[-_]?botnet", regex=True)
    df["LabelToBotnet"]   = labels.str.contains(r"to[-_]?botnet", regex=True)
    df["LabelBotnet"]     = labels.str.contains(r"botnet", regex=True)

    df["LabelCNC"] = 0
    df.loc[df["LabelToBotnet"], "LabelCNC"] = 1

    print("[LabelCNC] Summary:")
    print(df["LabelCNC"].value_counts())
    return df
