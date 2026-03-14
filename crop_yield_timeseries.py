import pandas as pd
import os

INPUT_FILE = "merge.csv"
OUTPUT_FILE = "crop_yield_timeseries.csv"

# ✅ Correct column names from your CSV
DISTRICT_COL = "District"
CROP_COL = "Crop"
YEAR_COL = "Year"
YIELD_COL = "Production"

# ---- Safety check ----
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# ---- Check columns ----
missing = [c for c in [DISTRICT_COL, CROP_COL, YEAR_COL, YIELD_COL] if c not in df.columns]
if missing:
    raise KeyError(f"❌ These columns are missing in CSV: {missing}")

# ---- Create time-series dataframe ----
ts_df = df[[DISTRICT_COL, CROP_COL, YEAR_COL, YIELD_COL]].copy()
ts_df = ts_df.sort_values([DISTRICT_COL, CROP_COL, YEAR_COL])

# Rename to standard names for LSTM
ts_df.columns = ["District", "Crop", "Year", "Yield"]

# Save
ts_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Time-series file created: {OUTPUT_FILE}")
