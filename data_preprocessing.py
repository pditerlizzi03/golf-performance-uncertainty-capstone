import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# PATH TO DATA
DATA_PATH = Path(__file__).resolve().parent / "PGA_raw.csv"

df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# DROP UNINFORMATIVE COLUMNS
cols_to_drop = [
    "Unnamed: 2",
    "Unnamed: 3",
    "Unnamed: 4",
    "no_cut"
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print("\nShape after dropping columns:", df.shape)
print("\nRemaining columns:")
print(df.columns.tolist())

# CLEAN COLUMN NAMES
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())

# TARGET VARIABLE
if "strokes" not in df.columns or "hole_par" not in df.columns:
    raise ValueError("Required columns 'strokes' or 'hole_par' not found.")

df["rel_strokes"] = df["strokes"] - df["hole_par"]

print("\nRelative strokes summary:")
print(df["rel_strokes"].describe())

# BASIC DATA CHECKS

# Missing values
missing = df.isna().mean().sort_values(ascending=False)
print("\nMissing value fraction by column:")
print(missing[missing > 0])

# Duplicates
dup_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {dup_count}")

# Unique entities
print("\nUnique counts:")
for col in ["player_id", "tournament_id", "season"]:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()}")

# FILTER PLAYERS WITH SUFFICIENT DATA
MIN_TOURNAMENTS = 20

player_counts = (
    df.groupby("player_id")
      .size()
      .rename("n_tournaments")
)

eligible_players = player_counts[player_counts >= MIN_TOURNAMENTS].index

df = df[df["player_id"].isin(eligible_players)].copy()

print("\nShape after filtering players:", df.shape)
print(f"Players retained: {df['player_id'].nunique()}")

# SORT FOR TIME-DEPENDENT FEATURES
df = df.sort_values(
    by=["player_id", "season", "tournament_id"]
).reset_index(drop=True)

print("\nData sorted for rolling features.")

# SAVE CLEAN DATASET
OUTPUT_PATH = "pga_clean_base.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nClean dataset saved to: {OUTPUT_PATH}")





