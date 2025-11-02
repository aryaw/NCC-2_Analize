import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
csv_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
input_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors.csv")
output_path = os.path.join(PROJECT_ROOT, "assets", "dataset", "NCC2AllSensors_clean.csv")
header_key = "StartTime"

def remove_duplicate_headers(in_path, out_path, header_key="StartTime", chunksize=500_000):
    print(f"Checking and cleaning file: {in_path}")
    print("This may take a few minutes for large files...")

    if not os.path.exists(in_path):
        print("Input file not found!")
        return

    total_rows = 0
    cleaned_rows = 0
    header_written = False

    # Read in chunks
    reader = pd.read_csv(in_path, chunksize=chunksize)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for chunk_i, chunk in enumerate(reader):
            # If this chunk contains the header line repeated, remove it
            if header_key in chunk.columns:
                # Drop rows where first column == header name (e.g., 'StartTime')
                first_col = chunk.columns[0]
                chunk = chunk[chunk[first_col] != header_key]

            # Write to output CSV
            chunk.to_csv(f_out, index=False, header=not header_written)
            header_written = True

            total_rows += len(chunk)
            cleaned_rows += len(chunk)
            if chunk_i % 10 == 0:
                print(f"  Processed chunks: {chunk_i} ({cleaned_rows:,} rows written)")

    print(f"Cleaning complete. Total rows written: {cleaned_rows:,}")
    print(f"Cleaned file saved at: {out_path}")

    # === Verification ===
    print("\nVerifying header count...")
    header_count = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{header_key},"):
                header_count += 1
    print(f"Header occurrences in cleaned file: {header_count}")
    if header_count == 1:
        print("Verification OK — only one header remains.")
    else:
        print(f"Warning: {header_count} headers still found. Check manually.")


# === RUN ===
if __name__ == "__main__":
    remove_duplicate_headers(input_path, output_path, header_key=header_key)
