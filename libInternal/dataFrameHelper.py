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

    result = pd.Series(np.nan, index=df.index)

    # regex checks
    # result[df["Label"].astype(str).apply(lambda x: bool(bot_pattern.search(x)))] = 1
    # result[df["Label"].astype(str).apply(lambda x: bool(normal_pattern.search(x)))] = 0

    # change regex with if/else logic
    def classify_label(label):
        text = str(label)
        if bot_pattern.search(text):
            return 1
        elif normal_pattern.search(text):
            return 0
        else:
            return np.nan
    result = df["Label"].apply(classify_label)

    # numeric fallback
    numeric = pd.to_numeric(df["Label"], errors="coerce")
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

def generate_plotly_evaluation_report(
    y_true,
    y_pred,
    y_prob,
    sensor_id="Unknown",
    best_threshold=None,
    output_dir=".",
    file_timestamp=None
):

    # === Compute metrics ===
    accuracy = round((y_pred == y_true).mean() * 100, 2)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [accuracy / 100, precision, recall, f1, roc_auc]
    })

    # bar chart
    metrics_fig = go.Figure(
        data=[
            go.Bar(
                x=metrics_df["Metric"],
                y=metrics_df["Score"],
                text=[f"{v:.3f}" for v in metrics_df["Score"]],
                textposition="auto",
                marker_color="#2e86de"
            )
        ],
        layout=go.Layout(
            title=f"Model Evaluation Metrics — Sensor {sensor_id}<br><sup>Best threshold: {best_threshold}</sup>",
            title_x=0.5,
            xaxis=dict(title="Metrics"),
            yaxis=dict(title="Score", range=[0, 1]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=13),
            height=500
        )
    )

    # report table
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4).reset_index().rename(columns={"index": "Class"})

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(report_df.columns),
                    fill_color="paleturquoise",
                    align="center"
                ),
                cells=dict(
                    values=[report_df[c] for c in report_df.columns],
                    fill_color="lavender",
                    align="center"
                )
            )
        ],
        layout=go.Layout(
            title=f"🧾 Classification Report — Sensor {sensor_id}",
            title_x=0.5,
            height=600
        )
    )

    # merge report
    combined = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Evaluation Metrics", "Classification Report"),
        row_heights=[0.4, 0.6],
        specs=[[{"type": "xy"}], [{"type": "table"}]]
    )
    combined.add_trace(metrics_fig.data[0], row=1, col=1)
    combined.add_trace(table_fig.data[0], row=2, col=1)
    combined.update_layout(
        title_text=f"Full Model Evaluation — Sensor {sensor_id}",
        title_x=0.5,
        height=900,
        showlegend=False
    )

    # save
    if not file_timestamp:
        import datetime
        file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)

    metrics_html = os.path.join(output_dir, f"Metrics_Sensor{sensor_id}_{file_timestamp}.html")
    table_html = os.path.join(output_dir, f"ClassReport_Sensor{sensor_id}_{file_timestamp}.html")
    combined_html = os.path.join(output_dir, f"FullReport_Sensor{sensor_id}_{file_timestamp}.html")

    metrics_fig.write_html(metrics_html)
    table_fig.write_html(table_html)
    combined.write_html(combined_html)

    print(f"[Report] Metrics saved -> {metrics_html}")
    print(f"[Report] Table saved   -> {table_html}")
    print(f"[Report] Combined saved -> {combined_html}")

    return {
        "metrics": metrics_html,
        "table": table_html,
        "combined": combined_html
    }

def compute_activity_groups(sensor_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    sdf = sensor_df.sort_values(["SrcAddr", "DstAddr", "StartTime"]).copy()
    sdf["PrevTime"] = sdf.groupby(["SrcAddr", "DstAddr"])["StartTime"].shift(1)
    sdf["TimeGap"]  = (sdf["StartTime"] - sdf["PrevTime"]).dt.total_seconds().fillna(0)

    pos = sdf.loc[sdf["TimeGap"] > 0, "TimeGap"]
    if len(pos) > 0:
        median_gap = pos.median()
        iqr = pos.quantile(0.75) - pos.quantile(0.25)
        G = median_gap + 2 * iqr
        if not np.isfinite(G) or G <= 0:
            G = 30.0
    else:
        G = 30.0

    sdf["ActivityGroup"] = sdf.groupby(["SrcAddr", "DstAddr"])["TimeGap"].apply(lambda x: (x > G).cumsum())
    return sdf, float(G)



