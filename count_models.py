import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FIG_DIR = PROJECT_ROOT / "models_count"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {FIG_DIR}/{name}")


# GLM IMPLEMENTATIONS

class PoissonRegression:
    """Poisson GLM: E[y|X] = exp(Xβ), Var[y|X] = exp(Xβ)"""
    
    def __init__(self, max_iter=100, tol=1e-8):
        self.max_iter, self.tol = max_iter, tol
        
    def fit(self, X, y):
        n, k = X.shape
        self.n, self.k = n, k
        
        # Initialize at OLS on log scale
        beta = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)[0]
        
        # IRLS
        for _ in range(self.max_iter):
            eta, mu = X @ beta, np.exp(X @ beta)
            w, z = mu, eta + (y - mu) / mu
            XtWX, XtWz = X.T @ np.diag(w) @ X, X.T @ (w * z)
            beta_new = np.linalg.solve(XtWX, XtWz)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                break
            beta = beta_new
        
        self.coef_ = beta
        self.eta_, self.mu_, self.resid_ = X @ beta, np.exp(X @ beta), y - np.exp(X @ beta)
        
        # Stats
        W = np.diag(self.mu_)
        self.cov_ = np.linalg.pinv(X.T @ W @ X)
        self.se_  = np.sqrt(np.diag(self.cov_))
        self.z_   = self.coef_ / self.se_
        self.pval_ = 2 * stats.norm.sf(np.abs(self.z_))
        z_c = stats.norm.ppf(0.975)
        self.ci_lo_, self.ci_hi_ = self.coef_ - z_c * self.se_, self.coef_ + z_c * self.se_
        
        from scipy.special import gammaln
        self.loglik_ = np.sum(y * self.eta_ - self.mu_ - gammaln(y + 1))
        self.aic_, self.bic_ = -2 * self.loglik_ + 2 * k, -2 * self.loglik_ + k * np.log(n)
        self.pearson_chisq_ = np.sum((y - self.mu_)**2 / self.mu_)
        self.dispersion_ = self.pearson_chisq_ / (n - k)
        return self
    
    def predict(self, X):
        return np.exp(X @ self.coef_)
    
    def summary(self, names):
        sig = lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "." if p<0.10 else " "
        print("=" * 78)
        print("POISSON REGRESSION SUMMARY")
        print("=" * 78)
        print(f"  Observations : {self.n:,}  |  Predictors : {self.k-1}")
        print(f"  Log-Lik : {self.loglik_:.2f}  |  AIC : {self.aic_:.2f}  |  BIC : {self.bic_:.2f}")
        print(f"  Dispersion (χ²/df) : {self.dispersion_:.4f}  ", end="")
        print("⚠ OVERDISPERSION" if self.dispersion_ > 1.5 else "✓ OK")
        print("=" * 78)
        print(f"{'Variable':<22} {'Coef':>8} {'SE':>8} {'z':>8} {'P>|z|':>8}  Sig")
        print("-" * 78)
        for i, nm in enumerate(names):
            print(f"{nm:<22} {self.coef_[i]:>8.4f} {self.se_[i]:>8.4f} {self.z_[i]:>8.3f} {self.pval_[i]:>8.4f}  {sig(self.pval_[i])}")
        print("=" * 78)


class NegativeBinomialRegression:
    """NB GLM: E[y|X] = exp(Xβ), Var[y|X] = μ + α·μ²"""
    
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter, self.tol = max_iter, tol
    
    def fit(self, X, y):
        n, k = X.shape
        self.n, self.k = n, k
        
        pois = PoissonRegression().fit(X, y)
        beta_init, alpha_init = pois.coef_, max(0.01, pois.dispersion_ - 1.0)
        theta_init = np.concatenate([beta_init, [np.log(alpha_init)]])
        
        def neg_loglik(theta):
            from scipy.special import gammaln
            beta, alpha = theta[:k], np.exp(theta[k])
            mu, r = np.exp(X @ beta), 1.0 / alpha
            ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) +
                  r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
            return -np.sum(ll)
        
        res = minimize(neg_loglik, theta_init, method='BFGS', options={'maxiter': self.max_iter})
        self.coef_, self.alpha_ = res.x[:k], np.exp(res.x[k])
        self.eta_, self.mu_, self.resid_ = X @ self.coef_, np.exp(X @ self.coef_), y - np.exp(X @ self.coef_)
        
        # Approximate SEs (use Poisson SEs for simplicity)
        self.se_ = pois.se_
        self.z_, self.pval_ = self.coef_ / self.se_, 2 * stats.norm.sf(np.abs(self.coef_ / self.se_))
        z_c = stats.norm.ppf(0.975)
        self.ci_lo_, self.ci_hi_ = self.coef_ - z_c * self.se_, self.coef_ + z_c * self.se_
        
        self.loglik_ = -res.fun
        self.aic_, self.bic_ = -2 * self.loglik_ + 2 * (k + 1), -2 * self.loglik_ + (k + 1) * np.log(n)
        return self
    
    def predict(self, X):
        return np.exp(X @ self.coef_)
    
    def summary(self, names):
        sig = lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "." if p<0.10 else " "
        print("=" * 78)
        print("NEGATIVE BINOMIAL REGRESSION SUMMARY")
        print("=" * 78)
        print(f"  Observations : {self.n:,}  |  Predictors : {self.k-1}  |  α : {self.alpha_:.4f}")
        print(f"  Log-Lik : {self.loglik_:.2f}  |  AIC : {self.aic_:.2f}  |  BIC : {self.bic_:.2f}")
        print("=" * 78)
        print(f"{'Variable':<22} {'Coef':>8} {'SE':>8} {'z':>8} {'P>|z|':>8}  Sig")
        print("-" * 78)
        for i, nm in enumerate(names):
            print(f"{nm:<22} {self.coef_[i]:>8.4f} {self.se_[i]:>8.4f} {self.z_[i]:>8.3f} {self.pval_[i]:>8.4f}  {sig(self.pval_[i])}")
        print("=" * 78)


# 1. LOAD & PREPROCESS
print("=" * 78)
print("COUNT-BASED MODELS: Poisson & Negative Binomial")
print("=" * 78)

DATA_PATH = PROJECT_ROOT / "Data" / "pga_features.csv"
df = pd.read_csv(DATA_PATH)
TARGET, PREDICTORS = "rel_strokes", [
    "sg_total", "exp_mean_rel", "exp_sd_rel", "roll_mean_rel_10",
    "roll_sd_rel_10", "roll_bad_rate_10", "purse"
]

df_m = df[["player_id", "season"] + [TARGET] + PREDICTORS].dropna().reset_index(drop=True)
print(f"Data: {len(df_m):,} rows | {df_m['player_id'].nunique()} players")

train, test = df_m[df_m["season"] <= 2019], df_m[df_m["season"] >= 2020]
X_tr_raw, y_tr_raw = train[PREDICTORS].values, train[TARGET].values
X_te_raw, y_te_raw = test[PREDICTORS].values, test[TARGET].values

# Transform to non-negative
shift = float(y_tr_raw.min())
y_tr = y_tr_raw - shift + 1 if shift < 0 else y_tr_raw
y_te = y_te_raw - shift + 1 if shift < 0 else y_te_raw
print(f"Target shift: {shift:.2f} → range [{y_tr.min():.1f}, {y_tr.max():.1f}]")

# Scale predictors
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr_raw), scaler.transform(X_te_raw)
X_tr_mat = np.hstack([np.ones((len(X_tr_s), 1)), X_tr_s])
X_te_mat = np.hstack([np.ones((len(X_te_s), 1)), X_te_s])
FEAT_NAMES = ["const"] + PREDICTORS


# 2. POISSON
print("\n" + "=" * 78)
print("POISSON REGRESSION")
print("=" * 78)

pois = PoissonRegression().fit(X_tr_mat, y_tr)
pois.summary(FEAT_NAMES)


# 3. NEGATIVE BINOMIAL
print("\n" + "=" * 78)
print("NEGATIVE BINOMIAL REGRESSION")
print("=" * 78)

nb = NegativeBinomialRegression().fit(X_tr_mat, y_tr)
nb.summary(FEAT_NAMES)


# 4. OUT-OF-SAMPLE EVALUATION
print("\n" + "=" * 78)
print("OUT-OF-SAMPLE EVALUATION (Test 2020-2022)")
print("=" * 78)

y_hat_pois_t = pois.predict(X_te_mat)
y_hat_nb_t = nb.predict(X_te_mat)

# Back-transform
if shift < 0:
    y_hat_pois, y_hat_nb = y_hat_pois_t + shift - 1, y_hat_nb_t + shift - 1
else:
    y_hat_pois, y_hat_nb = y_hat_pois_t, y_hat_nb_t

def metrics(y, yh, name):
    return {
        "Model": name,
        "RMSE": round(float(np.sqrt(mean_squared_error(y, yh))), 4),
        "MAE": round(float(mean_absolute_error(y, yh)), 4),
        "R2": round(float(r2_score(y, yh)), 4),
        "Bias": round(float(np.mean(yh - y)), 4),
    }

pois_m, nb_m = metrics(y_te_raw, y_hat_pois, "Poisson"), metrics(y_te_raw, y_hat_nb, "NegBin")

print("\nPoisson:  RMSE={:.4f}  MAE={:.4f}  R²={:.4f}  Bias={:+.4f}".format(
    pois_m["RMSE"], pois_m["MAE"], pois_m["R2"], pois_m["Bias"]))
print("NegBin :  RMSE={:.4f}  MAE={:.4f}  R²={:.4f}  Bias={:+.4f}".format(
    nb_m["RMSE"], nb_m["MAE"], nb_m["R2"], nb_m["Bias"]))

# Load OLS
try:
    ols_df = pd.read_csv(PROJECT_ROOT / "figures_ols" / "ols_metrics.csv")
    ols_m = {
        "Model": "OLS",
        "RMSE": round(float(ols_df[ols_df["Metric"] == "RMSE"]["Value"].values[0]), 4),
        "MAE": round(float(ols_df[ols_df["Metric"] == "MAE"]["Value"].values[0]), 4),
        "R2": round(float(ols_df[ols_df["Metric"] == "R2_test"]["Value"].values[0]), 4),
        "Bias": round(float(ols_df[ols_df["Metric"] == "Bias"]["Value"].values[0]), 4),
    }
    print("OLS    :  RMSE={:.4f}  MAE={:.4f}  R²={:.4f}  Bias={:+.4f}".format(
        ols_m["RMSE"], ols_m["MAE"], ols_m["R2"], ols_m["Bias"]))
except:
    ols_m = None


# 5. COMPARISON TABLE
print("\n" + "=" * 78)
print("MODEL COMPARISON")
print("=" * 78)

comp = []
if ols_m: comp.append(ols_m)
comp.append({**pois_m, "AIC": round(pois.aic_, 2), "BIC": round(pois.bic_, 2)})
comp.append({**nb_m, "AIC": round(nb.aic_, 2), "BIC": round(nb.bic_, 2)})
comp_df = pd.DataFrame(comp)
print("\n" + comp_df.to_string(index=False))


# 6. PLOTS
print("\n" + "=" * 78)
print("DIAGNOSTIC PLOTS")
print("=" * 78)

# Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Count Models — Test Set (2020-2022)", fontsize=13, fontweight="bold")

lo, hi = min(y_te_raw.min(), y_hat_pois.min()), max(y_te_raw.max(), y_hat_pois.max())

axes[0].scatter(y_te_raw, y_hat_pois, s=6, alpha=0.3, color="darkorange")
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5)
axes[0].set_title("Poisson")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].annotate(f"RMSE={pois_m['RMSE']}\nR²={pois_m['R2']}", xy=(0.05, 0.95),
                xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

axes[1].scatter(y_te_raw, y_hat_nb, s=6, alpha=0.3, color="darkgreen")
axes[1].plot([lo, hi], [lo, hi], "r--", lw=1.5)
axes[1].set_title("Negative Binomial")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")
axes[1].annotate(f"RMSE={nb_m['RMSE']}\nR²={nb_m['R2']}", xy=(0.05, 0.95),
                xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

savefig("count_test_predicted_vs_actual.png")

# Residuals
res_pois, res_nb = y_te_raw - y_hat_pois, y_te_raw - y_hat_nb
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Residual Distributions", fontsize=13, fontweight="bold")

axes[0].hist(res_pois, bins=60, color="darkorange", edgecolor="white", lw=0.3)
axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
axes[0].set_title("Poisson")
axes[0].set_xlabel("Residual")

axes[1].hist(res_nb, bins=60, color="darkgreen", edgecolor="white", lw=0.3)
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_title("Negative Binomial")
axes[1].set_xlabel("Residual")

savefig("count_test_residuals.png")


# 7. SAVE
comp_df.to_csv(FIG_DIR / "model_comparison.csv", index=False)
pd.DataFrame({"Model": ["Poisson", "NegBin"], "Dispersion_chisq_over_df": [round(pois.dispersion_, 4), np.nan],
              "Alpha": [np.nan, round(nb.alpha_, 4)]}).to_csv(FIG_DIR / "overdispersion.csv", index=False)
print(f"\n[saved] {FIG_DIR}/model_comparison.csv")
print(f"[saved] {FIG_DIR}/overdispersion.csv")


# SUMMARY
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"""
OVERDISPERSION:
  Poisson χ²/df = {pois.dispersion_:.4f}  {'→ OVERDISPERSION' if pois.dispersion_ > 1.5 else '→ OK'}
  NB α = {nb.alpha_:.4f}  {'→ Substantial' if nb.alpha_ > 0.2 else '→ Moderate'}

PREDICTIVE PERFORMANCE (Test RMSE):
  Poisson : {pois_m['RMSE']}
  NegBin  : {nb_m['RMSE']}
  {f"OLS     : {ols_m['RMSE']}" if ols_m else ""}

CONCLUSION:
  {'NB improves over Poisson due to overdispersion modeling.' if nb_m['RMSE'] < pois_m['RMSE'] else 'Poisson and NB perform similarly.'}
  {f"Count models {'improve over' if nb_m['RMSE'] < ols_m['RMSE'] else 'are competitive with'} OLS." if ols_m else ""}
  
  For golf analytics:
  - rel_strokes is effectively continuous (aggregated over 72 holes)
  - Gaussian OLS/LMM are appropriate and interpretable
  - Count models add complexity without substantial gain
  
  → Recommend: Use LMM for variance decomposition (player/course effects)
    rather than NB for dispersion modeling.
""")
print("Done. Outputs in:", os.path.abspath(FIG_DIR))
