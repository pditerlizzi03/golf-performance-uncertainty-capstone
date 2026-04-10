import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as sps
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

FIG_DIR = "figures_lmm"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {FIG_DIR}/{name}")

SEP = "=" * 74
sep = "-" * 74


# LMM ENGINE — EM ALGORITHM

class LinearMixedModel:
    """
    Linear Mixed-Effects Model with multiple crossed random intercepts.

    Model:  y = X*beta + Z1*u1 + Z2*u2 + ... + epsilon
    where:
        X      = (n x p)   fixed-effects design matrix (with intercept col)
        Zk     = (n x qk)  random-effects design matrix for grouping k
        uk     ~ N(0, sigma2_k * I_{qk})   group random intercepts
        epsilon ~ N(0, sigma2_e * I_n)     residual noise

    Estimation via Expectation-Maximization (EM):
    - E-step: compute posterior mean and variance of random effects
    - M-step: update variance components sigma2_k, sigma2_e, and beta
    - Converges to REML-equivalent variance component estimates

    After convergence, BLUPs (Best Linear Unbiased Predictors) are
    extracted as the posterior means of the random effects from the E-step.
    """

    def __init__(self, max_iter=200, tol=1e-6, verbose=True):
        self.max_iter = max_iter
        self.tol      = tol
        self.verbose  = verbose

    def fit(self, y, X, group_arrays):
        """
        Parameters
        ----------
        y            : np.ndarray (n,)   response
        X            : np.ndarray (n,p)  fixed-effects matrix (incl. const)
        group_arrays : list of np.ndarray (n,)  group labels per random effect
                       e.g. [player_ids, course_ids]
        """
        n, p = X.shape
        n_re = len(group_arrays)                    # number of random-effect groupings

        # Map group labels -> integer indices; build Z matrices
        self.group_maps_  = []   # label -> int index
        self.group_labels_= []   # int index -> original label
        Zs = []                  # list of (n x q_k) incidence matrices
        qs = []                  # group sizes
        for grp in group_arrays:
            labels = np.array(grp)
            unique = np.unique(labels)
            lbl2id = {lbl: i for i, lbl in enumerate(unique)}
            idx    = np.array([lbl2id[l] for l in labels])
            q      = len(unique)
            Z      = np.zeros((n, q), dtype=np.float64)
            Z[np.arange(n), idx] = 1.0
            self.group_maps_.append(lbl2id)
            self.group_labels_.append(unique)
            Zs.append(Z)
            qs.append(q)

        # Initialise variance components
        sigma2_e  = float(np.var(y))
        sigma2_re = [float(np.var(y)) / (n_re + 1)] * n_re   # one per grouping

        # EM iterations
        lls = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):

            # ----------------------------------------------------------------
            # Build marginal covariance V = sum_k sigma2_k * Z_k Z_k' + sigma2_e I
            # For large n, work with the Woodbury identity to avoid n x n matrix.
            # V^{-1} y  is computed blockwise using the matrix inversion lemma.
            # ----------------------------------------------------------------
            # Henderson Mixed Model Equations (MME) instead —
            # equivalent to the E/M steps but numerically more stable.
            #
            # MME:  [ X'X/se  X'Z/se ] [beta] = [ X'y/se ]
            #       [ Z'X/se  D^{-1}+Z'Z/se ] [u]    [ Z'y/se ]
            # where D = block-diag(sigma2_k * I_qk)
            #
            # This is O((p + sum_k q_k)^3), not O(n^3) — feasible for large n.
            # ----------------------------------------------------------------

            q_total = sum(qs)
            Z_full  = np.hstack(Zs)          # (n x q_total)

            # Build D^{-1} (block diagonal of 1/sigma2_k * I)
            D_inv_diag = np.concatenate([
                np.full(qs[k], 1.0 / sigma2_re[k]) for k in range(n_re)
            ])
            D_inv = np.diag(D_inv_diag)

            # MME coefficient matrix C and RHS r (scaled by 1/sigma2_e)
            inv_se = 1.0 / sigma2_e
            XtX  = X.T @ X
            XtZ  = X.T @ Z_full
            ZtX  = Z_full.T @ X
            ZtZ  = Z_full.T @ Z_full

            C_top    = np.hstack([XtX,  XtZ ])
            C_bottom = np.hstack([ZtX,  ZtZ + sigma2_e * D_inv])
            C        = np.vstack([C_top, C_bottom]) * inv_se

            r = np.concatenate([X.T @ y, Z_full.T @ y]) * inv_se

            # Solve MME
            try:
                sol = np.linalg.solve(C, r)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(C, r, rcond=None)[0]

            beta_new = sol[:p]
            u_new    = sol[p:]                 # all random effects stacked

            # Split u_new back into per-grouping blups
            u_list = []
            offset = 0
            for k in range(n_re):
                u_list.append(u_new[offset:offset + qs[k]])
                offset += qs[k]

        
            # M-step: update variance components
            
            # Residual: y - X beta - Z u
            y_hat_full  = X @ beta_new + Z_full @ u_new
            e           = y - y_hat_full

            # sigma2_e update: E[e'e] / n
            # For EM, we also need tr(C^{-1} * [lower-right block of C * sigma2_e])
            # Approximate: use the point estimate for speed & stability
            sigma2_e_new = float(e @ e) / n

            # sigma2_k update: E[u_k'u_k] / q_k
            sigma2_re_new = []
            try:
                C_inv = np.linalg.inv(C)         # (p+q_total) x (p+q_total)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C)

            offset = p                           # offset into C_inv for random effects
            for k in range(n_re):
                uk   = u_list[k]
                qk   = qs[k]
                # tr(C_inv[offset:offset+qk, offset:offset+qk]) * sigma2_e
                # gives the uncertainty correction in EM
                tr_k = np.trace(C_inv[offset:offset+qk, offset:offset+qk]) * sigma2_e
                s2k  = (float(uk @ uk) + tr_k) / qk
                sigma2_re_new.append(max(s2k, 1e-8))   # clamp to avoid collapse
                offset += qk

            
            # Log-likelihood (for convergence monitoring and AIC/BIC)
            
            # Marginal log-likelihood of y under current parameters
            # log L = -0.5 * (n * log(2pi) + log|V| + y'V^{-1}y)
            # Use Henderson's formula: log|V| = log|D| + log|C/sigma_e| + q_total*log(sigma_e)
            # Approximation via residual:
            ll = -0.5 * n * np.log(2 * np.pi * sigma2_e_new) \
                 - 0.5 * float(e @ e) / sigma2_e_new
            for k in range(n_re):
                ll -= 0.5 * qs[k] * np.log(sigma2_re_new[k]) \
                    + 0.5 * float(u_list[k] @ u_list[k]) / sigma2_re_new[k]
            lls.append(ll)

            
            # Convergence check
            
            delta = abs(ll - prev_ll)
            if self.verbose and (iteration % 25 == 0 or delta < self.tol):
                parts = " | ".join(
                    f"sigma2_re[{k}]={sigma2_re_new[k]:.4f}" for k in range(n_re))
                print(f"  iter {iteration+1:>4d} | LL={ll:>12.4f} | delta={delta:.2e} | "
                      f"sigma2_e={sigma2_e_new:.4f} | {parts}")

            if delta < self.tol and iteration > 5:
                print(f"  Converged at iteration {iteration + 1}  (delta={delta:.2e})")
                break

            # Update
            sigma2_e     = sigma2_e_new
            sigma2_re    = sigma2_re_new
            prev_ll      = ll
            self.beta_    = beta_new

        else:
            print(f"  WARNING: EM did not converge in {self.max_iter} iterations.")

        
        # Store fitted quantities
        
        self.beta_       = beta_new
        self.u_list_     = u_list
        self.sigma2_e_   = sigma2_e_new
        self.sigma2_re_  = sigma2_re_new
        self.n_re_       = n_re
        self.qs_         = qs
        self.Zs_         = Zs
        self.Z_full_     = Z_full
        self.lls_        = lls
        self.final_ll_   = lls[-1]
        self.n_          = n
        self.p_          = p
        self.n_params_   = p + n_re + 1   # beta + sigma2_re_k + sigma2_e

        # Fitted values and residuals
        self.fitted_    = X @ self.beta_ + Z_full @ u_new
        self.residuals_ = y - self.fitted_

        
        # Fixed-effect inference via sandwich approximation
        # (GLS covariance of beta given estimated variance components)
        
        # Var(beta_hat) = (X' V^{-1} X)^{-1}
        # Using Henderson's C_inv top-left block * sigma2_e
        self.cov_beta_  = C_inv[:p, :p] * sigma2_e_new
        self.se_beta_   = np.sqrt(np.diag(self.cov_beta_))
        df_resid        = n - p
        self.tstat_     = self.beta_ / self.se_beta_
        self.pval_      = 2 * sps.t.sf(np.abs(self.tstat_), df=df_resid)
        t_crit          = sps.t.ppf(0.975, df=df_resid)
        self.ci_lo_     = self.beta_ - t_crit * self.se_beta_
        self.ci_hi_     = self.beta_ + t_crit * self.se_beta_

        
        # Information criteria
        
        k_aic = self.n_params_
        self.aic_ = -2 * self.final_ll_ + 2 * k_aic
        self.bic_ = -2 * self.final_ll_ + np.log(n) * k_aic

        
        # R² metrics
        
        # Marginal R²: variance explained by fixed effects alone
        var_fixed   = float(np.var(X @ self.beta_))
        var_total   = float(np.var(y))
        self.r2_marginal_   = var_fixed / var_total

        # Conditional R²: variance explained by fixed + random effects
        var_rand    = sum(self.sigma2_re_)   # sum of random-effect variances
        self.r2_conditional_ = (var_fixed + var_rand) / (var_fixed + var_rand + self.sigma2_e_)

        return self

    def predict(self, X, group_arrays_new=None):
        """
        Predict using fixed effects + random intercepts where available.
        For unseen groups, random effect = 0 (population-level prediction).
        """
        y_hat = X @ self.beta_

        if group_arrays_new is None:
            return y_hat

        for k, (grp_new, lbl2id, u_k) in enumerate(
                zip(group_arrays_new, self.group_maps_, self.u_list_)):
            for i, lbl in enumerate(grp_new):
                if lbl in lbl2id:
                    y_hat[i] += u_k[lbl2id[lbl]]
                # else: unseen group -> BLUP = 0 (population mean)
        return y_hat

    def summary(self, feat_names=None):
        """Print a formatted model summary."""
        names = feat_names or [f"x{i}" for i in range(self.p_)]
        sig   = lambda p: "***" if p<0.001 else "**" if p<0.01 \
                          else "*" if p<0.05 else "." if p<0.10 else " "
        print(SEP)
        print("LINEAR MIXED-EFFECTS MODEL SUMMARY")
        print(SEP)
        print(f"  Observations         : {self.n_:>10,}")
        print(f"  Fixed-effect params  : {self.p_:>10}")
        print(f"  Random-effect groups : {self.n_re_:>10}")
        print(f"  Final log-likelihood : {self.final_ll_:>10.4f}")
        print(f"  AIC                  : {self.aic_:>10.4f}")
        print(f"  BIC                  : {self.bic_:>10.4f}")
        print(f"  Marginal  R²         : {self.r2_marginal_:>10.4f}")
        print(f"  Conditional R²       : {self.r2_conditional_:>10.4f}")
        print()
        print("Random Effects:")
        for k in range(self.n_re_):
            sd = np.sqrt(self.sigma2_re_[k])
            print(f"  Group {k+1} variance: {self.sigma2_re_[k]:.4f}  "
                  f"SD: {sd:.4f}   (n_groups={self.qs_[k]})")
        print(f"  Residual  variance: {self.sigma2_e_:.4f}  "
              f"SD: {np.sqrt(self.sigma2_e_):.4f}")
        print()
        print("Fixed Effects:")
        print(f"  {'Variable':<22} {'Coef':>8} {'StdErr':>8} {'t':>8} "
              f"{'P>|t|':>8} {'[0.025':>8} {'0.975]':>8}  Sig")
        print("  " + "-" * 70)
        for i, nm in enumerate(names):
            print(f"  {nm:<22} {self.beta_[i]:>8.4f} {self.se_beta_[i]:>8.4f} "
                  f"{self.tstat_[i]:>8.3f} {self.pval_[i]:>8.4f} "
                  f"{self.ci_lo_[i]:>8.4f} {self.ci_hi_[i]:>8.4f}  "
                  f"{sig(self.pval_[i])}")
        print(SEP)
        print("Sig. codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")
        print(SEP)


# 1. LOAD & PREPROCESSING
print(SEP)
print("1. LOAD & PREPROCESSING")
print(SEP)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "Data" / "pga_features.csv"
df = pd.read_csv(DATA_PATH)
print(f"Raw dataset shape : {df.shape}")

TARGET     = "rel_strokes"
PREDICTORS = [
    "sg_total",           # Skill: comprehensive strokes-gained benchmark
    "exp_mean_rel",       # Long-run baseline: expanding historical average
    "exp_sd_rel",         # Long-run volatility: expanding historical SD
    "roll_mean_rel_10",   # Short-run form: rolling 10-event average
    "roll_sd_rel_10",     # Short-run volatility: rolling 10-event SD
    "roll_bad_rate_10",   # Downside risk: fraction bad events in last 10
    "purse",              # Tournament control: prize money
]
GROUPING_COLS = ["player_id", "course"]   # random-effect groupings
MODEL_COLS    = [TARGET] + PREDICTORS

df_m = df[["player_id", "season", "course"] + MODEL_COLS].copy()

n_before = len(df_m)
df_m     = df_m.dropna(subset=MODEL_COLS).reset_index(drop=True)
n_after  = len(df_m)
print(f"Rows before NA drop : {n_before:,}")
print(f"Rows after  NA drop : {n_after:,}  ({n_before - n_after:,} dropped, "
      f"{(n_before-n_after)/n_before:.1%})")
print(f"Players retained    : {df_m['player_id'].nunique()}")
print(f"Courses  retained   : {df_m['course'].nunique()}")
print(f"Seasons             : {sorted(df_m['season'].unique())}")


# 2. TRAIN / TEST SPLIT (chronological — no leakage)
print("\n" + SEP)
print("2. TRAIN / TEST SPLIT  (train: 2015-2019 | test: 2020-2022)")
print(SEP)

train = df_m[df_m["season"] <= 2019].copy().reset_index(drop=True)
test  = df_m[df_m["season"] >= 2020].copy().reset_index(drop=True)

print(f"Train : {len(train):>6,} rows | seasons {train['season'].min()}-"
      f"{train['season'].max()} | "
      f"players {train['player_id'].nunique()} | "
      f"courses {train['course'].nunique()}")
print(f"Test  : {len(test):>6,} rows | seasons {test['season'].min()}-"
      f"{test['season'].max()}  | "
      f"players {test['player_id'].nunique()} | "
      f"courses {test['course'].nunique()}")

# Players / courses in test but NOT in train (will get BLUP = 0)
new_players = set(test["player_id"]) - set(train["player_id"])
new_courses  = set(test["course"])   - set(train["course"])
print(f"\nNew players in test (BLUP=0): {len(new_players)}")
print(f"New courses  in test (BLUP=0): {len(new_courses)}")


# 3. SCALING (fit on train only)
print("\n" + SEP)
print("3. SCALING  (StandardScaler fit on TRAIN only)")
print(SEP)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(train[PREDICTORS].values)
X_te_s = scaler.transform(test[PREDICTORS].values)

# Add intercept column
ones_tr = np.ones((len(X_tr_s), 1))
ones_te = np.ones((len(X_te_s), 1))
X_tr = np.hstack([ones_tr, X_tr_s])
X_te = np.hstack([ones_te, X_te_s])

y_tr = train[TARGET].values
y_te = test[TARGET].values

FEAT_NAMES = ["const"] + PREDICTORS

print("Scaler fit on training set only — no data leakage.\n")
tbl = pd.DataFrame({"mean(train)": scaler.mean_, "std(train)": scaler.scale_},
                   index=PREDICTORS)
print(tbl.round(4).to_string())


# 4. MODEL FITTING
print("\n" + SEP)
print("4a. MODEL 1: RANDOM INTERCEPT FOR PLAYER ONLY")
print("    rel_strokes ~ fixed_effects + (1 | player_id)")
print(SEP)

t0 = time.time()
lmm1 = LinearMixedModel(max_iter=200, tol=1e-6, verbose=True)
lmm1.fit(y_tr, X_tr, [train["player_id"].values])
print(f"\nFit time: {time.time()-t0:.1f}s")
lmm1.summary(FEAT_NAMES)

print("\n" + SEP)
print("4b. MODEL 2: RANDOM INTERCEPTS FOR PLAYER + COURSE")
print("    rel_strokes ~ fixed_effects + (1 | player_id) + (1 | course)")
print(SEP)

t0 = time.time()
lmm2 = LinearMixedModel(max_iter=200, tol=1e-6, verbose=True)
lmm2.fit(y_tr, X_tr, [train["player_id"].values, train["course"].values])
print(f"\nFit time: {time.time()-t0:.1f}s")
lmm2.summary(FEAT_NAMES)

# Use Model 2 (player + course) as the primary model going forward
lmm = lmm2
print("\nModel 2 (player + course) selected as primary model.")


# 5. DIAGNOSTICS (training set)
print("\n" + SEP)
print("5. DIAGNOSTICS — Training Set")
print(SEP)

res_tr = lmm.residuals_
fit_tr = lmm.fitted_

# Basic residual stats
print(f"\nResidual statistics (training set):")
print(f"  Mean      : {res_tr.mean():.5f}")
print(f"  SD        : {res_tr.std():.4f}   (cf. OLS: 4.1017)")
print(f"  Skewness  : {sps.skew(res_tr):.4f}")
print(f"  Kurt(exc) : {sps.kurtosis(res_tr):.4f}")

# Shapiro-Wilk on subsample
samp = np.random.choice(res_tr, 5000, replace=False)
sw_w, sw_p = sps.shapiro(samp)
print(f"\n  Shapiro-Wilk (n=5000): W={sw_w:.4f}, p={sw_p:.4e}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LMM Residual Diagnostics — Training Set (2015-2019)",
             fontsize=13, fontweight="bold")

# Histogram
axes[0].hist(res_tr, bins=70, color="#2E86AB", edgecolor="white", lw=0.3)
axes[0].axvline(0, color="red", ls="--", lw=1.5)
axes[0].set_title("Histogram of Residuals")
axes[0].set_xlabel("Residual (rel_strokes)")
axes[0].set_ylabel("Count")
axes[0].annotate(
    f"Mean={res_tr.mean():.3f}\nSD={res_tr.std():.3f}\n"
    f"Skew={sps.skew(res_tr):.3f}\nKurt={sps.kurtosis(res_tr):.3f}",
    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# Q-Q plot
(osm, osr), (slope, intercept, _) = sps.probplot(res_tr, dist="norm")
axes[1].scatter(osm, osr, s=4, alpha=0.3, color="#2E86AB", label="LMM residuals")
axes[1].plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.8,
             label="Normal reference")
axes[1].set_title("Q-Q Plot (vs Normal)")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles")
axes[1].legend(fontsize=8)

# Residuals vs fitted (with binned mean)
axes[2].scatter(fit_tr, res_tr, s=4, alpha=0.2, color="#2E86AB")
axes[2].axhline(0, color="red", ls="--", lw=1.5)
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

savefig("lmm_residual_diagnostics_train.png")


# 6. VARIANCE DECOMPOSITION
print("\n" + SEP)
print("6. VARIANCE DECOMPOSITION")
print(SEP)

var_player  = lmm.sigma2_re_[0]
var_course  = lmm.sigma2_re_[1]
var_resid   = lmm.sigma2_e_
var_total   = var_player + var_course + var_resid

print(f"\n{'Source':<25} {'Variance':>10} {'SD':>10} {'% of Total':>12}")
print(sep)
print(f"{'Player (random intercept)':<25} {var_player:>10.4f} "
      f"{np.sqrt(var_player):>10.4f} {100*var_player/var_total:>11.2f}%")
print(f"{'Course (random intercept)':<25} {var_course:>10.4f} "
      f"{np.sqrt(var_course):>10.4f} {100*var_course/var_total:>11.2f}%")
print(f"{'Residual (within-player)':<25} {var_resid:>10.4f} "
      f"{np.sqrt(var_resid):>10.4f} {100*var_resid/var_total:>11.2f}%")
print(sep)
print(f"{'Total':<25} {var_total:>10.4f} "
      f"{np.sqrt(var_total):>10.4f} {'100.00%':>12}")

# ICC: player-only, course-only, combined
icc_player  = var_player / var_total
icc_course  = var_course / var_total
icc_combined= (var_player + var_course) / var_total

print(f"\n  ICC (player only)          : {icc_player:.4f}")
print(f"  ICC (course only)          : {icc_course:.4f}")
print(f"  ICC (player + course)      : {icc_combined:.4f}")
print(f"\n  Marginal  R² (fixed FX)    : {lmm.r2_marginal_:.4f}")
print(f"  Conditional R² (fixed+rand): {lmm.r2_conditional_:.4f}")
print(f"  Explained by random effects: "
      f"{lmm.r2_conditional_ - lmm.r2_marginal_:.4f}")

print(f"""
  INTERPRETATION:
  - {100*icc_player:.1f}% of total variance in rel_strokes is attributable to
    stable between-player differences in baseline scoring level.
    This is structural player skill — slow-moving, career-level variation.
  - {100*icc_course:.1f}% is attributable to between-course difficulty.
    Even after controlling for skill/form, course layout systematically
    shifts average scores up or down.
  - {100*(var_resid/var_total):.1f}% is residual noise — within-player, within-course
    variation that cannot be explained by any included predictor.
    This represents the irreducible uncertainty in golf performance:
    the random bounce, the unpredictable weather, the hot/cold day.
  - The gap between marginal R² ({lmm.r2_marginal_:.3f}) and conditional R²
    ({lmm.r2_conditional_:.3f}) quantifies the contribution of random effects.
""")

# Variance decomposition bar chart
fig, ax = plt.subplots(figsize=(8, 5))
labels   = ["Player\n(between)", "Course\n(between)", "Residual\n(within)"]
values   = [100*var_player/var_total, 100*var_course/var_total,
            100*var_resid/var_total]
colors   = ["#2E86AB", "#A23B72", "#F18F01"]
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("Variance Decomposition of rel_strokes\n"
             "(Linear Mixed-Effects Model)", fontweight="bold", fontsize=12)
ax.set_ylabel("% of Total Variance")
ax.set_ylim(0, max(values) * 1.25)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5, f"{val:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
savefig("lmm_variance_decomposition.png")


# =============================================================================
# RANDOM EFFECTS EXTRACTION (BLUPs)
# =============================================================================
print(SEP)
print("RANDOM EFFECTS EXTRACTION (BLUPs)")
print(SEP)

player_blups = pd.DataFrame({
    "player_id": lmm.group_labels_[0],
    "blup"      : lmm.u_list_[0]
}).sort_values("blup")

# Merge player name from original data (if available)
name_map = df[["player_id", "player"]].drop_duplicates().set_index("player_id")["player"]
player_blups["player_name"] = player_blups["player_id"].map(name_map)

print(f"\nTotal players with BLUP estimates: {len(player_blups)}")
print(f"BLUP mean  : {player_blups['blup'].mean():.5f}  (should be ~0 by construction)")
print(f"BLUP SD    : {player_blups['blup'].std():.4f}")
print(f"BLUP range : {player_blups['blup'].min():.4f}  to  {player_blups['blup'].max():.4f}")

print("\n--- Top 10 LOWEST BLUPs (best players — outperform their fixed-effects profile) ---")
print(player_blups.head(10)[["player_name", "player_id", "blup"]].to_string(index=False))

print("\n--- Top 10 HIGHEST BLUPs (worst-performing relative to prediction) ---")
print(player_blups.tail(10)[["player_name", "player_id", "blup"]].to_string(index=False))

# BLUP histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Random Effects (BLUPs) — Estimated Player & Course Intercepts",
             fontsize=12, fontweight="bold")

axes[0].hist(player_blups["blup"], bins=40, color="#2E86AB",
             edgecolor="white", lw=0.4)
axes[0].axvline(0, color="red", ls="--", lw=1.5)
axes[0].set_title("Player Random Intercepts (BLUPs)")
axes[0].set_xlabel("BLUP (strokes deviation from population mean)")
axes[0].set_ylabel("Count")
axes[0].annotate(
    f"n={len(player_blups)}\nMean={player_blups['blup'].mean():.3f}\n"
    f"SD={player_blups['blup'].std():.3f}",
    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

course_blups = pd.DataFrame({
    "course": lmm.group_labels_[1],
    "blup"  : lmm.u_list_[1]
}).sort_values("blup")

axes[1].barh(
    range(len(course_blups)),
    course_blups["blup"].values,
    color=["#2E86AB" if v < 0 else "#E84855" for v in course_blups["blup"]],
    edgecolor="white", lw=0.3
)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title("Course Random Intercepts (BLUPs)")
axes[1].set_xlabel("BLUP (strokes: negative = easier course)")
axes[1].set_yticks([])
axes[1].annotate(
    f"n={len(course_blups)} courses\nSD={course_blups['blup'].std():.3f}",
    xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

savefig("lmm_blup_distributions.png")

# Save BLUPs
player_blups.to_csv(os.path.join(FIG_DIR, "lmm_player_blups.csv"), index=False)
course_blups.to_csv(os.path.join(FIG_DIR, "lmm_course_blups.csv"), index=False)
print(f"\n[saved] {FIG_DIR}/lmm_player_blups.csv")
print(f"[saved] {FIG_DIR}/lmm_course_blups.csv")


# 7. OUT-OF-SAMPLE EVALUATION (TEST SET)
print("\n" + SEP)
print("7. OUT-OF-SAMPLE EVALUATION (2020-2022)")
print(SEP)

# Model 2: player + course random effects
y_hat2 = lmm2.predict(
    X_te,
    [test["player_id"].values, test["course"].values]
)

rmse2  = float(np.sqrt(mean_squared_error(y_te, y_hat2)))
mae2   = float(mean_absolute_error(y_te, y_hat2))
r2_2   = float(r2_score(y_te, y_hat2))
bias2  = float(np.mean(y_hat2 - y_te))

# Model 1: player random effect only
y_hat1 = lmm1.predict(X_te, [test["player_id"].values])
rmse1  = float(np.sqrt(mean_squared_error(y_te, y_hat1)))
mae1   = float(mean_absolute_error(y_te, y_hat1))
r2_1   = float(r2_score(y_te, y_hat1))
bias1  = float(np.mean(y_hat1 - y_te))

# OLS baseline metrics (from ols_baseline.py)
ols_metrics = {"RMSE": 4.9283, "MAE": 3.7661, "R2": 0.5621, "Bias": 1.1410}

print("\n" + "-" * 60)
print(f"{'Metric':<12} {'OLS (baseline)':>16} {'LMM-1 (player)':>16} {'LMM-2 (p+c)':>13}")
print("-" * 60)
print(f"{'RMSE':<12} {ols_metrics['RMSE']:>16.4f} {rmse1:>16.4f} {rmse2:>13.4f}")
print(f"{'MAE':<12} {ols_metrics['MAE']:>16.4f} {mae1:>16.4f} {mae2:>13.4f}")
print(f"{'R2 (test)':<12} {ols_metrics['R2']:>16.4f} {r2_1:>16.4f} {r2_2:>13.4f}")
print(f"{'Bias':<12} {ols_metrics['Bias']:>+16.4f} {bias1:>+16.4f} {bias2:>+13.4f}")
print("-" * 60)

# Improvements vs OLS
rmse_imp = ols_metrics["RMSE"] - rmse2
r2_imp   = r2_2 - ols_metrics["R2"]
print(f"\n  LMM-2 vs OLS:  RMSE improvement = {rmse_imp:+.4f} strokes")
print(f"                 R2   improvement = {r2_imp:+.4f}")

# Test-set plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("LMM-2 Out-of-Sample Performance — Test Set (2020-2022)",
             fontsize=13, fontweight="bold")

lo = min(y_te.min(), y_hat2.min()) - 1
hi = max(y_te.max(), y_hat2.max()) + 1
axes[0].scatter(y_te, y_hat2, s=6, alpha=0.3, color="#2E86AB", label="Observations")
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.8, label="Perfect fit (y = x)")
axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
axes[0].set_xlabel("Actual rel_strokes")
axes[0].set_ylabel("Predicted rel_strokes")
axes[0].set_title("Predicted vs Actual (LMM-2)")
axes[0].legend(fontsize=9)
axes[0].annotate(
    f"RMSE={rmse2:.3f}\nMAE ={mae2:.3f}\nR2  ={r2_2:.3f}\nBias={bias2:+.3f}",
    xy=(0.05, 0.96), xycoords="axes fraction", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85))

res_te2 = y_te - y_hat2
axes[1].hist(res_te2, bins=70, color="#2E86AB", edgecolor="white", lw=0.3)
axes[1].axvline(0, color="red", ls="--", lw=1.5)
axes[1].set_title("Residual Distribution — Test Set (LMM-2)")
axes[1].set_xlabel("Residual (actual - predicted)")
axes[1].set_ylabel("Count")
axes[1].annotate(
    f"Mean={res_te2.mean():.3f}\nSD={res_te2.std():.3f}",
    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

savefig("lmm_test_evaluation.png")


# 8. MODEL COMPARISON: OLS vs LMM-1 vs LMM-2
print("\n" + SEP)
print("8. MODEL COMPARISON — OLS vs LMM-1 vs LMM-2")
print(SEP)

# OLS log-likelihood approximation: -n/2 * log(2*pi*sigma2_e) - SSR/(2*sigma2_e)
# Using OLS residuals from the script run above (we re-fit a quick OLS for LL)
class SimpleOLS:
    def fit(self, X, y):
        self.beta_ = np.linalg.lstsq(X, y, rcond=None)[0]
        res        = y - X @ self.beta_
        self.s2_   = float(res @ res) / len(y)
        self.ll_   = -0.5 * len(y) * np.log(2*np.pi*self.s2_) - float(res@res)/(2*self.s2_)
        self.r2_   = 1 - float(res@res) / float(((y-y.mean())**2).sum())
        n, k       = X.shape
        self.aic_  = -2 * self.ll_ + 2 * k
        self.bic_  = -2 * self.ll_ + np.log(n) * k
        return self

ols_ref = SimpleOLS().fit(X_tr, y_tr)

print(f"\n{'Metric':<25} {'OLS':>12} {'LMM-1 (player)':>16} {'LMM-2 (p+c)':>13}")
print(sep)
print(f"{'Log-Likelihood':<25} {ols_ref.ll_:>12.2f} {lmm1.final_ll_:>16.2f} {lmm2.final_ll_:>13.2f}")
print(f"{'AIC':<25} {ols_ref.aic_:>12.2f} {lmm1.aic_:>16.2f} {lmm2.aic_:>13.2f}")
print(f"{'BIC':<25} {ols_ref.bic_:>12.2f} {lmm1.bic_:>16.2f} {lmm2.bic_:>13.2f}")
print(f"{'R2 (train)':<25} {ols_ref.r2_:>12.4f} {lmm1.r2_conditional_:>16.4f} {lmm2.r2_conditional_:>13.4f}")
print(f"{'R2 (test)':<25} {ols_metrics['R2']:>12.4f} {r2_1:>16.4f} {r2_2:>13.4f}")
print(sep)

# LRT: OLS vs LMM-1 (1 extra parameter: sigma2_player)
lrt_stat_1 = 2 * (lmm1.final_ll_ - ols_ref.ll_)
lrt_p_1    = float(sps.chi2.sf(lrt_stat_1, df=1))
print(f"\n  LRT: OLS vs LMM-1  chi2(1) = {lrt_stat_1:.2f},  p = {lrt_p_1:.4e}")

# LRT: LMM-1 vs LMM-2 (1 extra parameter: sigma2_course)
lrt_stat_2 = 2 * (lmm2.final_ll_ - lmm1.final_ll_)
lrt_p_2    = float(sps.chi2.sf(lrt_stat_2, df=1))
print(f"  LRT: LMM-1 vs LMM-2  chi2(1) = {lrt_stat_2:.2f},  p = {lrt_p_2:.4e}")

if lrt_p_1 < 0.05:
    print(f"\n  ✓ Player random effects significantly improve fit over OLS (p={lrt_p_1:.3e}).")
else:
    print(f"\n  Player random effects do not significantly improve fit (p={lrt_p_1:.3e}).")

if lrt_p_2 < 0.05:
    print(f"  ✓ Course random effects significantly improve fit over LMM-1 (p={lrt_p_2:.3e}).")
else:
    print(f"  Course random effects do not significantly improve fit (p={lrt_p_2:.3e}).")

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison: OLS vs LMM-1 vs LMM-2",
             fontsize=13, fontweight="bold")

model_names = ["OLS", "LMM-1\n(player)", "LMM-2\n(p+course)"]
aic_vals    = [ols_ref.aic_, lmm1.aic_, lmm2.aic_]
bic_vals    = [ols_ref.bic_, lmm1.bic_, lmm2.bic_]
r2t_vals    = [ols_metrics["R2"], r2_1, r2_2]
colors_cmp  = ["#F18F01", "#2E86AB", "#A23B72"]

for ax, vals, title, fmt in zip(
    axes,
    [aic_vals, bic_vals, r2t_vals],
    ["AIC (lower = better)", "BIC (lower = better)", "Out-of-sample R² (higher = better)"],
    [",.0f", ",.0f", ".4f"]
):
    bars = ax.bar(model_names, vals, color=colors_cmp, edgecolor="white", lw=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(title.split("(")[0].strip())
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(vals)-min(vals))*0.01,
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=9)

savefig("lmm_model_comparison.png")


# SAVE ALL RESULTS

# Coefficient table (LMM-2)
coef_df = pd.DataFrame({
    "variable": FEAT_NAMES,
    "coef"    : lmm2.beta_,
    "se"      : lmm2.se_beta_,
    "t"       : lmm2.tstat_,
    "p_value" : lmm2.pval_,
    "ci_lo95" : lmm2.ci_lo_,
    "ci_hi95" : lmm2.ci_hi_,
    "sig"     : ["***" if p<0.001 else "**" if p<0.01
                 else "*" if p<0.05 else "." if p<0.10 else ""
                 for p in lmm2.pval_]
})
coef_df.to_csv(os.path.join(FIG_DIR, "lmm2_coefficients.csv"), index=False)

# Metrics table
metrics_df = pd.DataFrame({
    "Model"  : ["OLS", "LMM-1", "LMM-2"],
    "RMSE"   : [ols_metrics["RMSE"], round(rmse1,5), round(rmse2,5)],
    "MAE"    : [ols_metrics["MAE"],  round(mae1,5),  round(mae2,5)],
    "R2_test": [ols_metrics["R2"],   round(r2_1,5),  round(r2_2,5)],
    "Bias"   : [ols_metrics["Bias"], round(bias1,5), round(bias2,5)],
    "AIC"    : [round(ols_ref.aic_,2), round(lmm1.aic_,2), round(lmm2.aic_,2)],
    "BIC"    : [round(ols_ref.bic_,2), round(lmm1.bic_,2), round(lmm2.bic_,2)],
    "LL"     : [round(ols_ref.ll_,2), round(lmm1.final_ll_,2), round(lmm2.final_ll_,2)],
})
metrics_df.to_csv(os.path.join(FIG_DIR, "model_comparison_metrics.csv"), index=False)

# Variance decomposition
var_df = pd.DataFrame({
    "Source"   : ["Player", "Course", "Residual", "Total"],
    "Variance" : [var_player, var_course, var_resid, var_total],
    "SD"       : [np.sqrt(v) for v in [var_player, var_course, var_resid, var_total]],
    "Pct"      : [100*v/var_total for v in [var_player, var_course, var_resid, var_total]],
})
var_df.to_csv(os.path.join(FIG_DIR, "lmm_variance_decomposition.csv"), index=False)

print(f"\n[saved] {FIG_DIR}/lmm2_coefficients.csv")
print(f"[saved] {FIG_DIR}/model_comparison_metrics.csv")
print(f"[saved] {FIG_DIR}/lmm_variance_decomposition.csv")
print(f"[saved] figures in {FIG_DIR}/")


# FINAL SUMMARY
print("\n" + SEP)
print("FINAL SUMMARY & CAPSTONE IMPLICATIONS")
print(SEP)
print(f"""
1. VARIANCE DECOMPOSITION FINDINGS
   {100*icc_player:.1f}% of total variance in rel_strokes is stable between-player
   skill heterogeneity — the portion OLS conflates with residual noise.
   {100*icc_course:.1f}% is between-course difficulty. Together, {100*icc_combined:.1f}% of variance
   is structural (grouping effects), validating the use of a mixed model
   over pooled OLS for this hierarchical panel dataset.

2. FIXED EFFECTS COMPARISON
   The LMM fixed-effect coefficients for skill (sg_total) and volatility
   predictors are expected to closely mirror OLS estimates, confirming
   robustness. Any differences reflect the re-weighting of observations
   under GLS relative to OLS — players with many tournaments receive
   relatively less weight per observation in the LMM.

3. PREDICTIVE PERFORMANCE
   The LMM-2 (player + course random effects) is expected to outperform
   OLS on out-of-sample R² and RMSE, because:
   (a) For players seen in training, BLUPs shift predictions toward each
       player's historical baseline, capturing systematic over/under-prediction.
   (b) Course adjustments absorb venue-specific difficulty effects that
       fixed effects cannot fully capture via purse alone.

4. UNCERTAINTY QUANTIFICATION
   The conditional R² vs marginal R² gap quantifies how much of predictive
   accuracy comes from modelling structural heterogeneity rather than just
   fixed-effect predictors. This is the LMM's core contribution to the
   uncertainty-quantification research agenda: it separates predictable
   structural variance from irreducible residual noise.

5. NEXT STEPS
   - Count-based models (Poisson / Negative Binomial) to handle the
     discrete stroke-count generating process (Section 3.3).
   - Random Forest benchmark for non-linear effect comparisons (Section 3.4).
   - Final comparison table across all four model families.
""")
print("Done. All outputs saved to:", os.path.abspath(FIG_DIR))
