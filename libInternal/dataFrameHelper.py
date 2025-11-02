import re
import pandas as pd

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

    # Flexible regex patterns
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