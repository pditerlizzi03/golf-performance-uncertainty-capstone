import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

FIG_DIR = "figures_ols"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {FIG_DIR}/{name}")


# OLS IMPLEMENTATION (numpy-based; equivalent to statsmodels OLS)

class OLSRegression:
    """
    Ordinary Least Squares via numpy. Produces coefficients, standard errors,
    t-stats, p-values, confidence intervals, R2, adjusted R2, F-statistic.
    """

    def fit(self, X, y):
        n, k = X.shape
        self.n = n
        self.k = k
        XtX_inv    = np.linalg.pinv(X.T @ X)
        self.coef_  = XtX_inv @ X.T @ y
        self.fitted_    = X @ self.coef_
        self.residuals_ = y - self.fitted_
        self.ssr_   = float(self.residuals_ @ self.residuals_)
        self.s2_    = self.ssr_ / (n - k)
        self.s_     = np.sqrt(self.s2_)
        self.cov_   = self.s2_ * XtX_inv
        self.se_    = np.sqrt(np.diag(self.cov_))
        self.tstat_ = self.coef_ / self.se_
        self.pval_  = 2 * stats.t.sf(np.abs(self.tstat_), df=n - k)
        t_crit = stats.t.ppf(0.975, df=n - k)
        self.ci_lo_ = self.coef_ - t_crit * self.se_
        self.ci_hi_ = self.coef_ + t_crit * self.se_
        sst = float(((y - y.mean()) ** 2).sum())
        self.r2_     = 1 - self.ssr_ / sst
        self.r2_adj_ = 1 - (1 - self.r2_) * (n - 1) / (n - k)
        k_vars       = k - 1
        self.fstat_  = (self.r2_ / k_vars) / ((1 - self.r2_) / (n - k))
        self.fpval_  = stats.f.sf(self.fstat_, dfn=k_vars, dfd=n - k)
        return self

    def predict(self, X):
        return X @ self.coef_

    def summary(self, names=None):
        names = names or [f"x{i}" for i in range(self.k)]
        sig   = lambda p: "***" if p < 0.001 else "**" if p < 0.01 \
                          else "*" if p < 0.05 else "." if p < 0.10 else " "
        sep = "=" * 74
        print(sep)
        print("OLS REGRESSION SUMMARY")
        print(sep)
        print(f"  Observations     : {self.n:>10,}")
        print(f"  Predictors       : {self.k - 1:>10}")
        print(f"  R-squared        : {self.r2_:>10.4f}")
        print(f"  Adj. R-squared   : {self.r2_adj_:>10.4f}")
        print(f"  Residual Std Err : {self.s_:>10.4f}")
        print(f"  F-statistic      : {self.fstat_:>10.2f}  (p = {self.fpval_:.4e})")
        print(sep)
        print(f"{'Variable':<22} {'Coef':>8} {'StdErr':>8} {'t':>8} "
              f"{'P>|t|':>8} {'[0.025':>8} {'0.975]':>8}  Sig")
        print("-" * 74)
        for i, nm in enumerate(names):
            print(f"{nm:<22} {self.coef_[i]:>8.4f} {self.se_[i]:>8.4f} "
                  f"{self.tstat_[i]:>8.3f} {self.pval_[i]:>8.4f} "
                  f"{self.ci_lo_[i]:>8.4f} {self.ci_hi_[i]:>8.4f}  "
                  f"{sig(self.pval_[i])}")
        print(sep)
        print("Sig. codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")
        print(sep)


def compute_vif(X_df):
    """VIF_j = 1 / (1 - R2_j) where R2_j is from regressing col j on all others."""
    cols, vifs = X_df.columns.tolist(), []
    for col in cols:
        y_j = X_df[col].values
        X_j = np.column_stack([np.ones(len(y_j)), X_df.drop(columns=col).values])
        r2  = OLSRegression().fit(X_j, y_j).r2_
        vifs.append(1 / (1 - r2) if r2 < 1.0 else np.inf)
    return pd.DataFrame({"Feature": cols, "VIF": vifs}).sort_values("VIF", ascending=False)


# 1. LOAD & PREPROCESSING
print("=" * 74)
print("1. LOAD & PREPROCESSING")
print("=" * 74)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "Data" / "pga_features.csv"
df = pd.read_csv(DATA_PATH)
print(f"Raw dataset shape : {df.shape}")

TARGET     = "rel_strokes"
PREDICTORS = [
    "sg_total",           # Skill: total strokes gained (comprehensive performance benchmark)
    "exp_mean_rel",       # Long-run baseline: expanding historical average rel_strokes
    "exp_sd_rel",         # Long-run volatility: expanding historical SD of rel_strokes
    "roll_mean_rel_10",   # Short-run form: rolling 10-event average rel_strokes
    "roll_sd_rel_10",     # Short-run volatility: rolling 10-event SD of rel_strokes
    "roll_bad_rate_10",   # Downside risk: fraction of bad events in last 10 tournaments
    "purse",              # Tournament control: prize money (field-quality proxy)
]
MODEL_COLS = [TARGET] + PREDICTORS

df_m = df[["player_id", "season"] + MODEL_COLS].copy()

n_before = len(df_m)
df_m     = df_m.dropna(subset=MODEL_COLS).reset_index(drop=True)
n_after  = len(df_m)
print(f"Rows before NA drop : {n_before:,}")
print(f"Rows after  NA drop : {n_after:,}  ({n_before - n_after:,} dropped, "
      f"{(n_before - n_after)/n_before:.1%})")
print(f"Players retained    : {df_m['player_id'].nunique()}")
print(f"Seasons             : {sorted(df_m['season'].unique())}")

print("\n--- Summary statistics of modeling variables ---")
print(df_m[MODEL_COLS].describe().round(4).to_string())


# 2. TRAIN / TEST SPLIT (chronological)
print("\n" + "=" * 74)
print("2. TRAIN / TEST SPLIT  (train: 2015-2019 | test: 2020-2022)")
print("=" * 74)

train = df_m[df_m["season"] <= 2019].copy()
test  = df_m[df_m["season"] >= 2020].copy()

X_tr_raw, y_tr = train[PREDICTORS].values, train[TARGET].values
X_te_raw, y_te = test[PREDICTORS].values,  test[TARGET].values

print(f"Train : {len(train):>6,} rows | seasons {train['season'].min()}-{train['season'].max()} | "
      f"players {train['player_id'].nunique()}")
print(f"Test  : {len(test):>6,} rows | seasons {test['season'].min()}-{test['season'].max()} | "
      f"players {test['player_id'].nunique()}")
print(f"\nrel_strokes | Train  mean={y_tr.mean():.3f}  SD={y_tr.std():.3f}")
print(f"             | Test   mean={y_te.mean():.3f}  SD={y_te.std():.3f}")


# 3. SCALING (fit on train only to prevent leakage)
print("\n" + "=" * 74)
print("3. SCALING  (StandardScaler fit on TRAIN only)")
print("=" * 74)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr_raw)
X_te_s = scaler.transform(X_te_raw)

# Prepend intercept column
X_tr_mat = np.hstack([np.ones((len(X_tr_s), 1)), X_tr_s])
X_te_mat = np.hstack([np.ones((len(X_te_s), 1)), X_te_s])

FEAT_NAMES = ["const"] + PREDICTORS
scale_tbl  = pd.DataFrame({"mean(train)": scaler.mean_, "std(train)": scaler.scale_},
                           index=PREDICTORS)
print("Scaler fit on training set only. No data leakage.\n")
print(scale_tbl.round(4).to_string())


# 4. MODEL FITTING
print("\n" + "=" * 74)
print("4. OLS MODEL FITTING")
print("=" * 74)

ols = OLSRegression().fit(X_tr_mat, y_tr)
ols.summary(FEAT_NAMES)

# Coefficient table for saving
coef_df = pd.DataFrame({
    "variable": FEAT_NAMES,
    "coef"    : ols.coef_,
    "se"      : ols.se_,
    "t"       : ols.tstat_,
    "p_value" : ols.pval_,
    "ci_lo95" : ols.ci_lo_,
    "ci_hi95" : ols.ci_hi_,
    "sig"     : ["***" if p < 0.001 else "**" if p < 0.01
                 else "*" if p < 0.05 else "." if p < 0.10 else ""
                 for p in ols.pval_]
})

print("\n--- Coefficient Interpretation (predictors z-scored; unit = 1 SD change) ---")
print("""
  const            : Model intercept. Average predicted rel_strokes when
                     all predictors are at their training-set means.

  sg_total         : Total strokes gained — the key skill benchmark. Higher
                     SG (better performance vs field) directly reduces
                     rel_strokes. Expected sign: NEGATIVE (strongly).

  exp_mean_rel     : Long-run historical average. Players with chronically
                     higher rel_strokes carry that baseline forward.
                     Expected sign: POSITIVE.

  exp_sd_rel       : Long-run score spread. Chronic inconsistency predicts
                     slightly higher expected scores (asymmetric upside/
                     downside in competition). Expected sign: POSITIVE.

  roll_mean_rel_10 : Recent form. Poor recent results predict continued
                     above-par scoring. Expected sign: POSITIVE.

  roll_sd_rel_10   : Short-run consistency. Higher recent SD widens the
                     outcome distribution with upward tail pressure.
                     Expected sign: POSITIVE.

  roll_bad_rate_10 : Tail-risk proxy. Frequent recent bad events predict
                     another below-threshold outcome. Expected sign: POSITIVE.

  purse            : Prize money. Field-quality proxy; direction ambiguous.
""")


# 5. DIAGNOSTICS
print("=" * 74)
print("5. DIAGNOSTICS")
print("=" * 74)

# 5a. VIF
print("\n--- 5a. Variance Inflation Factor ---")
print("VIF > 10 = severe multicollinearity; > 5 = worth investigating\n")
vif_df = compute_vif(pd.DataFrame(X_tr_s, columns=PREDICTORS))
print(vif_df.round(3).to_string(index=False))
high = vif_df[vif_df["VIF"] > 5]
print()
if len(high):
    print(f"  WARNING: Features with VIF > 5: {high['Feature'].tolist()}")
    print("  Rolling-mean and expanding-mean are correlated by construction.")
    print("  Interpret their coefficients jointly rather than in isolation.")
else:
    print("  OK: All VIF < 5. No severe multicollinearity detected.")

# 5b. Residual diagnostics
print("\n--- 5b. Residual Diagnostics (Training Set) ---")

res_tr  = ols.residuals_
fit_tr  = ols.fitted_

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("OLS Residual Diagnostics — Training Set (2015-2019)",
             fontsize=13, fontweight="bold")

# Histogram
axes[0].hist(res_tr, bins=70, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
axes[0].set_title("Histogram of Residuals")
axes[0].set_xlabel("Residual (rel_strokes)")
axes[0].set_ylabel("Count")
axes[0].annotate(
    f"Mean={res_tr.mean():.3f}\nSD={res_tr.std():.3f}\n"
    f"Skew={stats.skew(res_tr):.3f}\nKurt={stats.kurtosis(res_tr):.3f}",
    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# Q-Q plot
(osm, osr), (slope, intercept, _) = stats.probplot(res_tr, dist="norm")
axes[1].scatter(osm, osr, s=4, alpha=0.3, color="steelblue", label="Residuals")
axes[1].plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.8,
             label="Normal reference")
axes[1].set_title("Q-Q Plot (vs Normal)")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles")
axes[1].legend(fontsize=8)

# Residuals vs Fitted (with binned mean overlay)
axes[2].scatter(fit_tr, res_tr, s=4, alpha=0.2, color="steelblue")
axes[2].axhline(0, color="red", linestyle="--", lw=1.5)
edges = np.percentile(fit_tr, np.linspace(0, 100, 31))
bmx, bmy = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (fit_tr >= lo) & (fit_tr < hi)
    if m.sum() > 0:
        bmx.append(fit_tr[m].mean())
        bmy.append(res_tr[m].mean())
axes[2].plot(bmx, bmy, color="orange", lw=2, label="Binned mean")
axes[2].set_title("Residuals vs Fitted")
axes[2].set_xlabel("Fitted rel_strokes")
axes[2].set_ylabel("Residual")
axes[2].legend(fontsize=8)

savefig("ols_residual_diagnostics_train.png")

sample = (res_tr if len(res_tr) <= 5000
          else np.random.choice(res_tr, 5000, replace=False))
sw_w, sw_p = stats.shapiro(sample)
sk = stats.skew(res_tr)
ku = stats.kurtosis(res_tr)
print(f"\n  Shapiro-Wilk (n={len(sample)}): W={sw_w:.4f}, p={sw_p:.4e}")
print(f"  Skewness={sk:.4f}  |  Excess kurtosis={ku:.4f}")
if abs(sk) < 0.5 and abs(ku) < 1.0:
    print("  OK: Residuals approximately symmetric with near-normal tails.")
else:
    print("  NOTE: Some non-normality present. With large n, CLT ensures valid")
    print("  inference. Heavy tails motivate distributional/mixed models.")


# 6. EVALUATION ON TEST SET
print("\n" + "=" * 74)
print("6. OUT-OF-SAMPLE EVALUATION (2020-2022)")
print("=" * 74)

y_hat = ols.predict(X_te_mat)

rmse  = float(np.sqrt(mean_squared_error(y_te, y_hat)))
mae   = float(mean_absolute_error(y_te, y_hat))
r2    = float(r2_score(y_te, y_hat))
bias  = float(np.mean(y_hat - y_te))
r2_tr = ols.r2_
gap   = r2_tr - r2

print(f"\n  RMSE  (Root Mean Squared Error) : {rmse:.4f} strokes")
print(f"  MAE   (Mean Absolute Error)      : {mae:.4f} strokes")
print(f"  R2    (Out-of-sample)            : {r2:.4f}")
print(f"  Bias  (mean predicted - actual)  : {bias:+.4f} strokes")
print(f"\n  In-sample  R2 (train 2015-19)    : {r2_tr:.4f}")
print(f"  Out-of-sample R2 (test 2020-22)  : {r2:.4f}")
print(f"  R2 gap (train - test)            : {gap:.4f}", end="  ")
print("WARNING: some overfitting." if gap > 0.05 else "OK: model generalises well.")

res_te = y_te - y_hat
print(f"\n  Residual percentiles (test set):")
for q in [1, 5, 25, 50, 75, 95, 99]:
    print(f"    {q:3d}th pct: {np.percentile(res_te, q):+.3f}")

# Test-set plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("OLS Out-of-Sample Performance — Test Set (2020-2022)",
             fontsize=13, fontweight="bold")

lo = min(y_te.min(), y_hat.min()) - 1
hi = max(y_te.max(), y_hat.max()) + 1
axes[0].scatter(y_te, y_hat, s=6, alpha=0.3, color="steelblue", label="Observations")
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.8, label="Perfect fit (y = x)")
axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
axes[0].set_xlabel("Actual rel_strokes")
axes[0].set_ylabel("Predicted rel_strokes")
axes[0].set_title("Predicted vs Actual")
axes[0].legend(fontsize=9)
axes[0].annotate(
    f"RMSE={rmse:.3f}\nMAE ={mae:.3f}\nR2  ={r2:.3f}\nBias={bias:+.3f}",
    xy=(0.05, 0.96), xycoords="axes fraction", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85))

axes[1].hist(res_te, bins=70, color="steelblue", edgecolor="white", linewidth=0.3)
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_title("Residual Distribution — Test Set")
axes[1].set_xlabel("Residual (actual - predicted)")
axes[1].set_ylabel("Count")
axes[1].annotate(
    f"Mean={res_te.mean():.3f}\nSD={res_te.std():.3f}",
    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

savefig("ols_test_evaluation.png")

# SAVE OUTPUTS
pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2_test", "R2_train", "Adj_R2_train",
               "Bias", "F_stat", "F_pval"],
    "Value" : [round(rmse, 5), round(mae, 5), round(r2, 5), round(r2_tr, 5),
               round(ols.r2_adj_, 5), round(bias, 5),
               round(ols.fstat_, 3), round(float(ols.fpval_), 6)]
}).to_csv(os.path.join(FIG_DIR, "ols_metrics.csv"), index=False)
print(f"\n[saved] {FIG_DIR}/ols_metrics.csv")

coef_df.to_csv(os.path.join(FIG_DIR, "ols_coefficients.csv"), index=False)
print(f"[saved] {FIG_DIR}/ols_coefficients.csv")

vif_df.to_csv(os.path.join(FIG_DIR, "ols_vif.csv"), index=False)
print(f"[saved] {FIG_DIR}/ols_vif.csv")


# FINAL SUMMARY
print("\n" + "=" * 74)
print("SUMMARY & IMPLICATIONS FOR MODELING ROADMAP")
print("=" * 74)
print("""
1. SKILL vs VOLATILITY
   sg_total is expected to be the dominant predictor, reflecting the
   importance of technical skill (Broadie, 2012; Ehrlich & Kamimoto, 2025).
   Significant positive coefficients on the volatility variables
   (exp_sd_rel, roll_sd_rel_10) would confirm that performance uncertainty
   -- not just mean skill -- explains meaningful variation in rel_strokes,
   directly supporting the research thesis (Minton, 2016).

2. VOLATILITY VARIABLE SIGNIFICANCE
   If roll_sd_rel_10 and exp_sd_rel are statistically significant at the
   5% level, the OLS delivers direct econometric evidence that short-run
   and long-run performance variability drive expected scores independently
   of form (roll_mean_rel_10). The downside-risk proxy (roll_bad_rate_10)
   adds a tail-risk dimension beyond mean and SD.

3. RESIDUAL NORMALITY
   Golf scores are discrete counts aggregated over 72 holes; the central
   limit theorem ensures approximately normal residuals at large n. Any
   heavy tails in the Q-Q plot reflect genuine non-normality of competitive
   outcomes (extreme low/high totals) and motivate distributional models
   (Poisson, Negative Binomial) in later sections.

4. MOTIVATION FOR MIXED MODELS
   OLS treats each player-tournament as independent, ignoring the panel
   structure. Residual clustering within players is expected: some players
   systematically score above or below their OLS prediction. The Linear
   Mixed-Effects Model (Section 3.2) corrects this via random player
   intercepts, enabling variance decomposition between within-player and
   between-player uncertainty -- the core of the uncertainty-quantification
   agenda. The R2 gap (train vs test) further motivates this extension.

   --> Proceed to Section 3.2: Linear Mixed-Effects Model.
""")
print("Done. All outputs in:", os.path.abspath(FIG_DIR))
