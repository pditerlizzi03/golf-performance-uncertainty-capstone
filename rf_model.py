import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import os, warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

FIG_DIR = "models_rf"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {FIG_DIR}/{name}")


# 1. LOAD & PREPROCESSING
print("=" * 78)
print("RANDOM FOREST REGRESSION")
print("=" * 78)

df = pd.read_csv("/mnt/user-data/uploads/pga_features.csv")
TARGET, PREDICTORS = "rel_strokes", [
    "sg_total", "exp_mean_rel", "exp_sd_rel", "roll_mean_rel_10",
    "roll_sd_rel_10", "roll_bad_rate_10", "purse"
]

df_m = df[["player_id", "season"] + [TARGET] + PREDICTORS].dropna().reset_index(drop=True)
print(f"Data: {len(df_m):,} rows | {df_m['player_id'].nunique()} players")

train, test = df_m[df_m["season"] <= 2019], df_m[df_m["season"] >= 2020]
X_train, y_train = train[PREDICTORS].values, train[TARGET].values
X_test, y_test = test[PREDICTORS].values, test[TARGET].values

print(f"Train: {len(train):,} | Test: {len(test):,}")
print("→ No scaling (RF is scale-invariant)")


# 2. BASELINE MODEL
print("\n" + "=" * 78)
print("BASELINE RANDOM FOREST")
print("=" * 78)

rf_base = RandomForestRegressor(
    n_estimators=500, max_depth=None, min_samples_leaf=5,
    random_state=42, n_jobs=-1
).fit(X_train, y_train)

y_pred_base = rf_base.predict(X_test)
print(f"Baseline test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}")


# 3. HYPERPARAMETER TUNING (Simplified for speed)
print("\n" + "=" * 78)
print("HYPERPARAMETER TUNING (3-Fold CV, reduced grid)")
print("=" * 78)

param_grid = {
    'n_estimators': [500],
    'max_depth': [None, 15],
    'min_samples_leaf': [5, 10],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, scoring='neg_root_mean_squared_error',
    cv=3, n_jobs=-1, verbose=1
)

print("Running GridSearchCV...")
grid.fit(X_train, y_train)

print(f"\nBest params: {grid.best_params_}")
print(f"Best CV RMSE: {-grid.best_score_:.4f}")

rf = grid.best_estimator_


# 4. OUT-OF-SAMPLE EVALUATION
print("\n" + "=" * 78)
print("OUT-OF-SAMPLE EVALUATION (Test 2020-2022)")
print("=" * 78)

y_pred = rf.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))
bias = float(np.mean(y_pred - y_test))

print(f"\nRandom Forest:")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")
print(f"  Bias : {bias:+.4f}")

# Load OLS/count models
comp = []
try:
    ols_df = pd.read_csv("/home/claude/figures_ols/ols_metrics.csv")
    comp.append({
        "Model": "OLS",
        "RMSE": round(float(ols_df[ols_df["Metric"] == "RMSE"]["Value"].values[0]), 4),
        "R2": round(float(ols_df[ols_df["Metric"] == "R2_test"]["Value"].values[0]), 4),
    })
except:
    pass

try:
    cnt_df = pd.read_csv("/home/claude/models_count/model_comparison.csv")
    for _, row in cnt_df.iterrows():
        if row["Model"] in ["Poisson", "NegBin"]:
            comp.append({"Model": row["Model"], "RMSE": row["RMSE"], "R2": row["R2"]})
except:
    pass

comp.append({"Model": "RandomForest", "RMSE": round(rmse, 4), "R2": round(r2, 4)})
comp_df = pd.DataFrame(comp)

print("\n" + "=" * 78)
print("MODEL COMPARISON")
print("=" * 78)
print("\n" + comp_df.to_string(index=False))

best = comp_df.loc[comp_df["RMSE"].idxmin(), "Model"]
print(f"\n✓ Best RMSE: {best}")


# 5. FEATURE IMPORTANCE
print("\n" + "=" * 78)
print("FEATURE IMPORTANCE")
print("=" * 78)

imp = pd.DataFrame({
    "Feature": PREDICTORS,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

imp["Pct"] = 100 * imp["Importance"] / imp["Importance"].sum()
print("\n" + imp.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp["Feature"], imp["Importance"], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(imp))))
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance", fontweight="bold")
ax.invert_yaxis()
savefig("rf_feature_importance.png")

# Permutation importance
print("\nComputing permutation importance...")
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
perm_df = pd.DataFrame({
    "Feature": PREDICTORS,
    "Perm_Importance": perm.importances_mean
}).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)

print("\nPermutation Importance (test set):")
print(perm_df.to_string(index=False))


# 6. DIAGNOSTICS
print("\n" + "=" * 78)
print("DIAGNOSTIC PLOTS")
print("=" * 78)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Random Forest — Test Set (2020-2022)", fontsize=13, fontweight="bold")

lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
axes[0].scatter(y_test, y_pred, s=6, alpha=0.3, color="forestgreen")
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5)
axes[0].set_title("Predicted vs Actual")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].annotate(f"RMSE={rmse:.3f}\nR²={r2:.3f}", xy=(0.05, 0.95),
                xycoords="axes fraction", va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

res = y_test - y_pred
axes[1].hist(res, bins=60, color="forestgreen", edgecolor="white", lw=0.3)
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_title("Residuals")
axes[1].set_xlabel("Residual")
axes[1].annotate(f"Mean={res.mean():.3f}\nSD={res.std():.3f}", xy=(0.97, 0.95),
                xycoords="axes fraction", ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

savefig("rf_test_evaluation.png")

# Residuals vs fitted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, res, s=4, alpha=0.25, color="forestgreen")
ax.axhline(0, color="red", linestyle="--", lw=1.5)
ax.set_title("Residuals vs Fitted", fontweight="bold")
ax.set_xlabel("Fitted")
ax.set_ylabel("Residual")
savefig("rf_residual_diagnostics.png")


# 7. SAVE
pd.DataFrame([
    {"Metric": "RMSE_test", "Value": round(rmse, 5)},
    {"Metric": "MAE_test", "Value": round(mae, 5)},
    {"Metric": "R2_test", "Value": round(r2, 5)},
    {"Metric": "Bias_test", "Value": round(bias, 5)},
]).to_csv(f"{FIG_DIR}/rf_model_metrics.csv", index=False)

imp.to_csv(f"{FIG_DIR}/rf_feature_importance.csv", index=False)
perm_df.to_csv(f"{FIG_DIR}/rf_permutation_importance.csv", index=False)
comp_df.to_csv(f"{FIG_DIR}/rf_model_comparison.csv", index=False)

print(f"\n[saved] {FIG_DIR}/rf_model_metrics.csv")
print(f"[saved] {FIG_DIR}/rf_feature_importance.csv")
print(f"[saved] {FIG_DIR}/rf_model_comparison.csv")


# SUMMARY
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"""
RANDOM FOREST PERFORMANCE:
  Test RMSE : {rmse:.4f}
  Test R²   : {r2:.4f}

BEST MODEL (by RMSE):  {best}

FEATURE IMPORTANCE (Top 3):
  1. {imp.loc[0, 'Feature']:<20} : {imp.loc[0, 'Importance']:.3f}
  2. {imp.loc[1, 'Feature']:<20} : {imp.loc[1, 'Importance']:.3f}
  3. {imp.loc[2, 'Feature']:<20} : {imp.loc[2, 'Importance']:.3f}

INTERPRETATION:
  {'→ RF improves over linear models (nonlinear interactions help)' if best == 'RandomForest' else '→ Linear models remain competitive (golf performance largely linear)'}
  → sg_total dominates (comprehensive skill measure)
  → Volatility measures contribute meaningfully
  → For uncertainty quantification: prefer LMM (variance decomposition)
  → For pure prediction: {'RF' if best == 'RandomForest' else 'OLS/LMM'}
""")
print("Done. Outputs in:", os.path.abspath(FIG_DIR))
