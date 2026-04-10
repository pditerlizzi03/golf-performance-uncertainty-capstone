import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

# PATHS
INPUT_PATH = "pga_clean_base.csv"    
OUTPUT_PATH = "pga_features.csv"


# PARAMETERS
ROLL_WINDOWS = [5, 10, 20]  # tournaments
MIN_PERIODS = 3             # minimum past tournaments required to compute rolling stats
DOWNSIDE_Q = 0.75           # defines "bad performance" threshold per player (75th percentile)

# LOAD
df = pd.read_csv(INPUT_PATH)
print("Loaded:", df.shape)

required_cols = ["player_id", "season", "tournament_id", "rel_strokes"]
missing_req = [c for c in required_cols if c not in df.columns]
if missing_req:
    raise ValueError(f"Missing required columns: {missing_req}")

# Sort for time-ordered computations
df = df.sort_values(by=["player_id", "season", "tournament_id"]).reset_index(drop=True)

# 1) EXPANDING (PAST-ONLY) PLAYER STATS (Recommended for modeling)

# Expanding mean/std using ONLY past tournaments (shifted).
df["exp_mean_rel"] = (
    df.groupby("player_id")["rel_strokes"]
      .apply(lambda s: s.shift(1).expanding(min_periods=MIN_PERIODS).mean())
      .reset_index(level=0, drop=True)
)

df["exp_sd_rel"] = (
    df.groupby("player_id")["rel_strokes"]
      .apply(lambda s: s.shift(1).expanding(min_periods=MIN_PERIODS).std())
      .reset_index(level=0, drop=True)
)

# 2) ROLLING FORM & CONSISTENCY (PAST-ONLY, LAGGED)
for w in ROLL_WINDOWS:
    df[f"roll_mean_rel_{w}"] = (
        df.groupby("player_id")["rel_strokes"]
          .apply(lambda s: s.shift(1).rolling(window=w, min_periods=MIN_PERIODS).mean())
          .reset_index(level=0, drop=True)
    )

    df[f"roll_sd_rel_{w}"] = (
        df.groupby("player_id")["rel_strokes"]
          .apply(lambda s: s.shift(1).rolling(window=w, min_periods=MIN_PERIODS).std())
          .reset_index(level=0, drop=True)
    )

# 3) DOWNSIDE-RISK METRICS (FULLY PAST-ONLY)

# Compute a past-only expanding quantile per player:
# bad_threshold_rel_it = Q_{DOWNSIDE_Q}(rel_strokes for player i in tournaments < t)
# Implementation uses a group-by loop for clarity and correctness.

bad_threshold = np.full(len(df), np.nan, dtype=float)

for player_id, idx in df.groupby("player_id").groups.items():
    # idx is an index array (already in time order due to sorting)
    s = df.loc[idx, "rel_strokes"].astype(float).values

    # For each position k, compute quantile over s[:k] (past only)
    # Need at least MIN_PERIODS past observations.
    for k in range(len(s)):
        if k >= MIN_PERIODS:
            bad_threshold[idx[k]] = np.quantile(s[:k], DOWNSIDE_Q)
        else:
            bad_threshold[idx[k]] = np.nan

df["bad_threshold_rel"] = bad_threshold

# Indicator: whether current tournament is "bad" relative to past threshold
# If threshold is NaN (insufficient history), mark as NaN rather than 0/1.
df["is_bad_perf"] = np.where(
    df["bad_threshold_rel"].isna(),
    np.nan,
    (df["rel_strokes"] >= df["bad_threshold_rel"]).astype(int)
)

# Rolling bad-performance rate using only past events
for w in ROLL_WINDOWS:
    df[f"roll_bad_rate_{w}"] = (
        df.groupby("player_id")["is_bad_perf"]
          .apply(lambda s: s.shift(1).rolling(window=w, min_periods=MIN_PERIODS).mean())
          .reset_index(level=0, drop=True)
    )

# 4) SANITY CHECKS
feature_prefixes = [
    "exp_mean_rel", "exp_sd_rel",
    "roll_mean_rel_", "roll_sd_rel_",
    "bad_threshold_rel", "roll_bad_rate_"
]

created_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]

print("\nFeature columns created:")
print(created_cols)

print("\nMissingness in key engineered features (top 15):")
miss = df[created_cols].isna().mean().sort_values(ascending=False).head(15)
print(miss)

print("\nExample rows (one player, first 12 tournaments):")
example_player = df["player_id"].iloc[0]
print(
    df[df["player_id"] == example_player].head(12)[
        ["player_id", "season", "tournament_id", "rel_strokes",
         "exp_mean_rel", "exp_sd_rel",
         "roll_mean_rel_5", "roll_sd_rel_5",
         "bad_threshold_rel", "is_bad_perf", "roll_bad_rate_5"]
    ]
)

# 5) SAVE
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved feature dataset to: {OUTPUT_PATH}")

