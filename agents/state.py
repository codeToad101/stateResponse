"""
state.py

State Fiscal Response Function — data prep + calibration.

Pipeline:
  1. load_and_prepare_data()   -> reads results/us_state_response_data.csv,
                                   collapses quarterly-ffilled annual data
                                   down to true annual observations (in
                                   memory only — no intermediate file saved),
                                   builds R(t), Protest(t-lag), Gini(t),
                                   Growth(t).
  2. StateResponseFitter       -> fits a family of candidate response
                                   functions (linear / logistic / exponential
                                   parametric forms + a flexible GAM) and
                                   compares them on AIC/BIC/R2, with
                                   coefficient CIs and residual diagnostics.
  3. State                     -> ABM agent shell (next phase); left mostly
                                   as scaffolding, now wired to accept a
                                   fitted response function instead of a
                                   hardcoded polynomial.

Why annual, not quarterly, for fitting:
  The response variable (redistribution, Gini, tax rate) is sourced from
  annual data and forward-filled to quarterly only for calendar alignment,
  not because 4 independent quarterly observations actually exist. Fitting
  on the raw ffilled quarterly series would treat 4 duplicate copies of the
  same annual number as 4 data points, artificially inflating N and any
  significance/AIC comparison. We collapse to one observation per year for
  the DV. Predictors that carry real quarterly variation (protest, GDP
  growth) are aggregated *within* each year (mean) before that collapse, so
  the within-year signal isn't just discarded — it's what the lag structure
  is testing.
"""

import numpy as np
import pandas as pd
import mesa
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson


# ============================================================================
# HAC / Newey-West standard errors for (nonlinear) least-squares fits
# ============================================================================
#
# curve_fit's own covariance matrix (pcov) assumes residuals are independent
# across observations -- exactly the assumption Durbin-Watson told us is
# violated here (DW ~0.6-0.8, well below the "no autocorrelation" benchmark
# of ~2). That means our reported SEs/CIs were too tight. This block
# recomputes them with a Newey-West sandwich estimator, which explicitly
# allows nearby years' errors to be correlated instead of assuming they're
# independent, giving wider, more honest CIs. Works for both the linear and
# the nonlinear (logistic/exponential) models via a numerical Jacobian --
# same machinery, no need for a separate linear-only path.

def _numerical_jacobian(model_fn, X, params, eps=1e-6):
    """d(model_fn)/d(params) at `params`, evaluated at every observation.
    Returns an (n_obs, n_params) array."""
    params = np.asarray(params, dtype=float)
    base = model_fn(X, *params)
    n_obs, n_params = len(base), len(params)
    J = np.zeros((n_obs, n_params))
    for j in range(n_params):
        step = params.copy()
        # relative step, with a floor so it doesn't vanish near zero
        h = eps * max(abs(params[j]), 1.0)
        step[j] += h
        J[:, j] = (model_fn(X, *step) - base) / h
    return J


def newey_west_lag(n):
    """Automatic bandwidth (Newey & West, 1994 rule of thumb): grows slowly
    with sample size. For our n~60 this lands around 3 lags."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def hac_sandwich_covariance(jacobian, residuals, maxlags):
    """
    Newey-West HAC sandwich covariance for a (nonlinear) least-squares fit.

    jacobian:  (n, k) array, d(model)/d(params) at the fitted params
    residuals: (n,) array, y - y_pred, assumed time-ordered
    maxlags:   how many periods of autocorrelation to correct for

    Returns a (k, k) covariance matrix for the parameter estimates -- same
    role as pcov from curve_fit, but robust to autocorrelated residuals.
    """
    scores = jacobian * residuals[:, None]  # per-observation contributions, (n, k)

    # "Meat": weighted sum of score autocovariances (Bartlett kernel, same
    # kernel statsmodels uses for cov_type='HAC')
    meat = scores.T @ scores  # lag 0
    for lag in range(1, maxlags + 1):
        w = 1 - lag / (maxlags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += w * (gamma + gamma.T)

    # "Bread": Gauss-Newton approximation to (J'J)^-1
    bread = np.linalg.inv(jacobian.T @ jacobian)

    return bread @ meat @ bread


def hac_standard_errors(model_fn, X, y, popt, maxlags=None):
    """Convenience wrapper: returns (se, ci_95, maxlags_used) with HAC-
    corrected standard errors for a fitted parametric model."""
    y_pred = model_fn(X, *popt)
    residuals = y - y_pred
    n = len(y)
    if maxlags is None:
        maxlags = newey_west_lag(n)

    J = _numerical_jacobian(model_fn, X, popt)
    cov = hac_sandwich_covariance(J, residuals, maxlags)
    se = np.sqrt(np.diag(cov))
    ci = [(p - 1.96 * s, p + 1.96 * s) for p, s in zip(popt, se)]
    return se, ci, maxlags

try:
    from pygam import LinearGAM, s
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    print("⚠ pygam not installed (pip install pygam --break-system-packages). "
          "Flexible GAM candidate will be skipped; parametric models still run.")


# ============================================================================
# 1. DATA LOADING + ANNUAL AGGREGATION
# ============================================================================

# Columns that are already single annual values forward-filled across
# quarters (gini.csv, EITC, medicaid, etc. — see ManualDataTranslator).
# Aggregate with .first() per year: taking the mean would be numerically
# identical (they're constant within-year) but .first() is explicit about
# "this was never actually a quarterly observation."
ANNUAL_FFILLED_COLS = [
    'wage_p10', 'wage_p50', 'wage_p90',
    'gini_coefficient', 'avg_federal_tax_rate_pct',
    'redist_eitc', 'redist_snap', 'redist_medicaid', 'redist_ui',
    'workers_affected', 'days_idle', 'civil_unrest_events',
]

# Columns with genuine quarter-to-quarter variation. Aggregate with the
# within-year mean so real signal (not just one quarter's snapshot) feeds
# the annual model.
QUARTERLY_RATE_COLS = [
    'unemployment_rate_pct', 'lfpr_pct',
    'avg_hourly_earnings_nominal', 'avg_hourly_earnings_real',
    'job_openings_thousands', 'wage_growth_yoy', 'labor_market_tightness',
    'fed_tax_revenue_pct_gdp', 'protest_intensity_score',
]

# Levels where the year-end (Q4) value is the more natural "size of the
# economy this year" figure than an average of four SAAR levels.
QUARTERLY_LEVEL_COLS = [
    'gdp_billions', 'total_nonfarm_employment', 'civilian_labor_force_thousands',
]


def load_and_prepare_data(csv_path="results/us_state_response_data.csv",
                           protest_lag_years=1):
    """
    Load the quarterly CSV and collapse it to one row per year, ready for
    state-response fitting. No intermediate file is written — this is a
    pure in-memory transform.

    Returns
    -------
    annual : pd.DataFrame, indexed by year, with all aggregated columns
             plus derived R(t) [redistribution_pct_gdp], Growth(t)
             [gdp yoy % change], and Protest_lag [protest lagged by
             protest_lag_years].
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    present_ffilled = [c for c in ANNUAL_FFILLED_COLS if c in df.columns]
    present_rate = [c for c in QUARTERLY_RATE_COLS if c in df.columns]
    present_level = [c for c in QUARTERLY_LEVEL_COLS if c in df.columns]

    agg_map = {}
    agg_map.update({c: 'first' for c in present_ffilled})
    agg_map.update({c: 'mean' for c in present_rate})
    agg_map.update({c: 'last' for c in present_level})

    annual = df.groupby('year').agg(agg_map)

    # ---- Derived: redistribution as % of GDP (recomputed post-aggregation,
    # not aggregated as a pre-computed ratio, to avoid compounding rounding
    # from four quarterly ratios that were never independent anyway) ----
    redist_components = [c for c in ['redist_eitc', 'redist_snap',
                                      'redist_medicaid', 'redist_ui']
                          if c in annual.columns]
    if redist_components and 'gdp_billions' in annual.columns:
        annual['redistribution_total_billions'] = annual[redist_components].sum(axis=1)
        annual['redistribution_pct_gdp'] = (
            annual['redistribution_total_billions'] / annual['gdp_billions'] * 100
        )
    else:
        annual['redistribution_pct_gdp'] = np.nan

    # ---- Derived: GDP growth, YoY % change of the annual level ----
    if 'gdp_billions' in annual.columns:
        annual['gdp_growth_yoy_pct'] = annual['gdp_billions'].pct_change() * 100

    # ---- Derived: lagged protest signal ----
    if 'protest_intensity_score' in annual.columns:
        annual['protest_lag'] = annual['protest_intensity_score'].shift(protest_lag_years)

    return annual


def build_model_inputs(annual, y_col='redistribution_pct_gdp',
                        x_cols=('protest_lag', 'gini_coefficient', 'gdp_growth_yoy_pct')):
    """
    Extract (X, y) arrays for fitting, dropping any year with missing data
    in the required columns. Returns the aligned year index too, so
    residual diagnostics can be reported against real calendar years.
    """
    needed = list(x_cols) + [y_col]
    missing = [c for c in needed if c not in annual.columns]
    if missing:
        raise ValueError(f"Missing required columns for fitting: {missing}")

    sub = annual[needed].dropna()
    X = sub[list(x_cols)].to_numpy().T  # shape (n_predictors, n_obs) to match curve_fit convention
    y = sub[y_col].to_numpy()
    years = sub.index.to_numpy()
    return X, y, years


# ============================================================================
# 2. STATE RESPONSE FITTER
# ============================================================================

class StateResponseFitter:
    """
    Fits a family of candidate state response functions
        R(t) = f(Protest(t-lag), Gini(t), Growth(t))
    and compares them honestly: fixed AIC/BIC, coefficient CIs, and
    residual autocorrelation diagnostics (relevant here since this is a
    time series, not iid cross-sectional data).

    Deliberately does NOT hardcode one form as "the" model. Parametric
    candidates are kept for interpretability; a GAM candidate is included
    so each predictor's functional shape can be learned from data rather
    than assumed. Best model is selected on AIC but ties/close calls should
    be judged manually — lower AIC by a hair isn't a good reason to prefer
    a black-box GAM over an interpretable linear form.
    """

    def __init__(self, X, y, years, feature_names=('protest_lag', 'gini', 'growth')):
        self.X = X          # shape (n_features, n_obs)
        self.y = y
        self.years = years
        self.feature_names = list(feature_names)
        self.n = len(y)
        self.results = {}
        # Training-domain bounds per predictor, min/max as actually
        # observed in the fitting data. Used downstream (State.decide_policy)
        # to clip inputs to the range the response function was validated
        # on -- "bounded exponential": the function's SHAPE stays whatever
        # fit best in-range (exponential, if that wins), but it is never
        # *evaluated* outside where the data could inform it. Standard,
        # citable technique (domain-restricted extrapolation / trust
        # region), distinct from and simpler than swapping to a globally-
        # bounded functional form.
        self.domain_bounds_ = {
            name: (float(np.min(X[i])), float(np.max(X[i])))
            for i, name in enumerate(self.feature_names)
        }

    # ---- parametric candidate forms ----
    @staticmethod
    def linear_model(X, a, b, c, intercept):
        protest, gini, growth = X
        return a * protest + b * gini + c * growth + intercept

    @staticmethod
    def logistic_model(X, a, b, c, k, x0):
        """Response saturates at high protest/inequality (bounded-rational reaction)."""
        protest, gini, growth = X
        return a / (1 + np.exp(-k * (protest + b * gini - x0))) + c * growth

    @staticmethod
    def exponential_model(X, a, b, c):
        """Response accelerates at extreme inequality."""
        protest, gini, growth = X
        return a * np.exp(np.clip(b * gini, -50, 50)) + c * protest

    def _aic_bic(self, y_pred, k):
        """Standard Gaussian-likelihood AIC/BIC: n*ln(RSS/n) + penalty."""
        rss = np.sum((self.y - y_pred) ** 2)
        rss = max(rss, 1e-12)  # guard against log(0) on a perfect fit
        aic = self.n * np.log(rss / self.n) + 2 * k
        bic = self.n * np.log(rss / self.n) + k * np.log(self.n)
        return aic, bic, rss

    def _fit_parametric(self, name, model_fn, p0=None):
        try:
            popt, pcov = curve_fit(model_fn, self.X, self.y, p0=p0, maxfev=20000)
            y_pred = model_fn(self.X, *popt)
            r2 = r2_score(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            aic, bic, rss = self._aic_bic(y_pred, k=len(popt))

            # Naive 95% CI per coefficient, straight from curve_fit's pcov.
            # Assumes independent residuals -- kept for comparison only,
            # see hac_ci_95 below for the corrected version.
            perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
            ci = [(p - 1.96 * e, p + 1.96 * e) for p, e in zip(popt, perr)]

            dw = durbin_watson(self.y - y_pred)  # ~2 = no autocorrelation, <1.5 flags concern

            # HAC/Newey-West corrected SEs and CIs -- accounts for the
            # residual autocorrelation Durbin-Watson just flagged, rather
            # than assuming it away.
            hac_se, hac_ci, nw_lags = hac_standard_errors(model_fn, self.X, self.y, popt)

            self.results[name] = {
                'type': 'parametric', 'params': popt,
                'param_ci_95': ci,
                'param_se_hac': hac_se, 'param_ci_95_hac': hac_ci, 'nw_lags': nw_lags,
                'r2': r2, 'rmse': rmse, 'aic': aic, 'bic': bic,
                'durbin_watson': dw, 'y_pred': y_pred,
            }
            print(f"{name:12} | R²: {r2:6.3f} | RMSE: {rmse:8.4f} | "
                  f"AIC: {aic:8.2f} | BIC: {bic:8.2f} | DW: {dw:.2f}")
        except Exception as e:
            print(f"{name:12} | FAILED: {e}")

    def _fit_gam(self):
        if not PYGAM_AVAILABLE:
            return
        try:
            Xg = self.X.T  # pygam wants (n_obs, n_features)
            gam = LinearGAM(s(0) + s(1) + s(2)).gridsearch(Xg, self.y, progress=False)
            y_pred = gam.predict(Xg)
            r2 = r2_score(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            # pygam reports its own AIC (accounts for effective DoF of splines,
            # not raw parameter count) — use it directly rather than our
            # fixed-k formula, which would understate GAM flexibility.
            aic = gam.statistics_['AIC']
            edof = gam.statistics_['edof']
            bic = self.n * np.log(max(np.sum((self.y - y_pred) ** 2), 1e-12) / self.n) + edof * np.log(self.n)
            dw = durbin_watson(self.y - y_pred)

            self.results['gam'] = {
                'type': 'gam', 'model': gam, 'edof': edof,
                'r2': r2, 'rmse': rmse, 'aic': aic, 'bic': bic,
                'durbin_watson': dw, 'y_pred': y_pred,
            }
            print(f"{'gam':12} | R²: {r2:6.3f} | RMSE: {rmse:8.4f} | "
                  f"AIC: {aic:8.2f} | BIC: {bic:8.2f} | DW: {dw:.2f}  (edof={edof:.1f})")
        except Exception as e:
            print(f"{'gam':12} | FAILED: {e}")

    # Whether each candidate's functional form is bounded/saturating as
    # Gini/protest -> extreme values, vs. capable of unbounded (or
    # super-exponential) growth outside the training domain. Not a
    # judgment about in-sample fit -- purely about extrapolation safety.
    # linear: unbounded but grows only linearly (mild). logistic: bounded
    # by construction (R_max ceiling) -- the only candidate consistent
    # with "a state cannot spend >100% of GDP." exponential: unbounded
    # and, given b~18 fitted on a narrow 0.35-0.42 Gini window, explodes
    # by orders of magnitude just outside it (confirmed empirically: the
    # fitted exponential goes from ~3.8 at real Gini=0.42 to ~73,000 at
    # simulated Gini=0.96 once the ABM's wealth mechanics could actually
    # reach that range). gam: unconstrained spline, no boundedness
    # guarantee at all outside the training range.
    _EXTRAPOLATION_SAFE = {'linear': True, 'logistic': True,
                            'exponential': False, 'gam': False}

    def fit_and_compare(self, extrapolation_safe_only=False):
        """
        extrapolation_safe_only: when True, restricts model SELECTION
        (not fitting -- every candidate is still fit and reported) to
        candidates flagged extrapolation-safe above. This is not "assume
        the answer" curve-shopping: every candidate's AIC/BIC/R2 is still
        printed for full transparency, including cases where the
        unconstrained AIC winner is excluded from selection.

        Methodological basis: per Burnham & Anderson (2002, "Model
        Selection and Multimodel Inference"), models within a modest AIC
        gap represent comparable empirical support, and selection among
        them is legitimately informed by considerations beyond raw fit --
        theoretical plausibility, structural soundness, or (as here)
        fitness for the model's intended downstream use. This is not
        "picking a smaller ΔAIC gap than usual" special pleading: this
        project's ABM explicitly needs to evaluate off-historical-
        trajectory states (captured/dictatorship regimes reaching Gini
        levels the training data never saw), so extrapolation safety is
        a real, stated requirement of the use case, not an ad hoc
        preference invoked to override an inconvenient result. The ΔAIC
        between the unconstrained and constrained winners is always
        printed so the tradeoff being made is visible, not hidden.
        """
        print(f"Fitting on n={self.n} annual observations "
              f"({self.years.min()}-{self.years.max()})\n")

        self._fit_parametric('linear', self.linear_model,
                              p0=[1, 1, 1, np.mean(self.y)])
        self._fit_parametric('logistic', self.logistic_model,
                              p0=[np.ptp(self.y), 1, 0.1, 1, np.mean(self.X[1])])
        self._fit_parametric('exponential', self.exponential_model,
                              p0=[1, 1, 0.1])
        self._fit_gam()

        if not self.results:
            print("\n✗ No candidate model fit successfully.")
            return None, {}

        unconstrained_best = min(self.results, key=lambda k: self.results[k]['aic'])

        if not extrapolation_safe_only:
            best_model = unconstrained_best
            print(f"\n✓ Best model by AIC (unconstrained): {best_model}")
        else:
            safe_candidates = [k for k in self.results
                               if self._EXTRAPOLATION_SAFE.get(k, False)]
            if not safe_candidates:
                print("\n⚠ No extrapolation-safe candidate fit successfully; "
                      "falling back to the unconstrained AIC winner.")
                best_model = unconstrained_best
            else:
                best_model = min(safe_candidates, key=lambda k: self.results[k]['aic'])
                delta_aic = self.results[best_model]['aic'] - self.results[unconstrained_best]['aic']
                print(f"\n✓ Best model by AIC, restricted to extrapolation-safe "
                      f"candidates {safe_candidates}: {best_model}")
                if best_model != unconstrained_best:
                    print(f"  (unconstrained AIC winner was '{unconstrained_best}', "
                          f"AIC={self.results[unconstrained_best]['aic']:.2f}; "
                          f"'{best_model}' costs +{delta_aic:.2f} AIC in exchange for a "
                          f"bounded/saturating functional form -- see fit_and_compare "
                          f"docstring for the methodological justification.)")

        best = self.results[best_model]
        if best['durbin_watson'] < 1.5 or best['durbin_watson'] > 2.5:
            print(f"⚠ Durbin-Watson = {best['durbin_watson']:.2f} on the best model — "
                  f"residuals show autocorrelation. With n={self.n} annual points this is "
                  f"common and worth flagging rather than hiding: point estimates are still "
                  f"usable but standard/CI estimates may be too tight.")

        return best_model, self.results

    def summarize(self, model_name):
        """Print a human-readable coefficient table for a parametric model."""
        res = self.results.get(model_name)
        if res is None:
            print(f"No fitted result for '{model_name}'")
            return
        if res['type'] != 'parametric':
            print(f"'{model_name}' is a GAM — inspect via .results['gam']['model'].summary() "
                  f"or partial-dependence plots instead of a coefficient table.")
            return

        print(f"\n{model_name} — coefficient summary")
        print("-" * 60)
        labels = {
            'linear': ['a (protest_lag)', 'b (gini)', 'c (growth)', 'intercept'],
            'logistic': ['a (scale)', 'b (gini weight)', 'c (growth)', 'k (steepness)', 'x0 (midpoint)'],
            'exponential': ['a (scale)', 'b (gini exponent)', 'c (protest)'],
        }.get(model_name, [f'param_{i}' for i in range(len(res['params']))])

        print(f"  {'':22}   {'naive CI (iid residuals)':32} {'HAC/Newey-West CI (' + str(res['nw_lags']) + ' lags)'}")
        for label, val, (lo, hi), (hlo, hhi) in zip(
                labels, res['params'], res['param_ci_95'], res['param_ci_95_hac']):
            naive_str = f"[{lo:8.4f}, {hi:8.4f}]"
            hac_str = f"[{hlo:8.4f}, {hhi:8.4f}]"
            flips = (lo <= 0 <= hi) != (hlo <= 0 <= hhi)
            flag = "  ← CI vs 0 changes!" if flips else ""
            print(f"  {label:22} = {val:10.4f}   {naive_str:32} {hac_str}{flag}")
        print(f"  R² = {res['r2']:.3f}   RMSE = {res['rmse']:.4f}   "
              f"AIC = {res['aic']:.2f}   BIC = {res['bic']:.2f}")


# ============================================================================
# 3. BAYESIAN VAR — joint Protest <-> Redistribution <-> Gini system
# ============================================================================
#
# Why a VAR, and why Bayesian: R(t) depends on lagged Protest; Protest(t)
# plausibly depends on lagged R and on Gini, which all three series also
# trend with (Gini/protest correlate at 0.85 in levels). Fitting each
# equation separately risks both just re-detecting the shared trend rather
# than a real structural relationship. A VAR estimates the system jointly;
# Granger-style checks off it ask "does X help predict Y beyond Y's own
# history" -- a much stronger claim than eyeballing one coefficient's CI.
#
# A classical (OLS) VAR would badly overfit at n~60 for even a 3-variable,
# 1-lag system. A Minnesota-prior Bayesian VAR shrinks coefficients toward
# zero (own-lag less than cross-lag) -- the standard, citable fix for
# small-sample macro VARs (Litterman 1986; Bańbura, Giannone & Reichlin
# 2010), not an ad hoc regularization choice.

def prepare_var_data(annual, columns=('protest_intensity_score',
                                       'redistribution_pct_gdp',
                                       'gini_coefficient')):
    """
    First-differences the requested columns -- the stationarity fix for
    the trend-driven collinearity flagged in the single-equation fits --
    and drops the resulting leading NaN row.
    """
    sub = annual[list(columns)].dropna()
    diffed = sub.diff().dropna()
    diffed.columns = [f'd_{c}' for c in diffed.columns]
    return diffed


class BayesianVAR:
    """
    Small Bayesian VAR with a Minnesota-style shrinkage prior.

    Data handling: input should already be first-differenced (see
    prepare_var_data). Series are standardized (z-scored) internally so a
    single pair of shrinkage hyperparameters applies sensibly across
    variables in very different natural units (a Gini-point change vs. a
    %-GDP change), rather than needing the variable-specific scale
    factors classic Minnesota implementations carry.

    Estimation: closed-form Bayesian linear regression per equation, with
    a diagonal prior precision (own-lag coefficients shrunk less than
    cross-lag ones) and an empirical-Bayes (OLS plug-in) residual
    variance. This is the standard closed-form equivalent of the
    dummy-observation Minnesota BVAR construction, just without the
    dummy-observation bookkeeping.
    """

    def __init__(self, data, lag=1, lambda_overall=0.2, lambda_cross=0.5):
        self.columns = list(data.columns)
        self.lag = lag
        self.lambda_overall = lambda_overall
        self.lambda_cross = lambda_cross

        self.mu = data.mean()
        self.sigma = data.std()
        self.z = (data - self.mu) / self.sigma

        self.coefs_ = {}
        self.coef_cov_ = {}
        self.sigma2_ = {}
        self.n_obs_ = None

    def _design_matrix(self):
        z = self.z.to_numpy()
        n, k = z.shape
        p = self.lag
        rows = n - p
        X = np.ones((rows, 1 + k * p))
        for l in range(1, p + 1):
            X[:, 1 + (l - 1) * k: 1 + l * k] = z[p - l: n - l]
        Y = z[p:]
        return X, Y

    def fit(self):
        X, Y = self._design_matrix()
        n, k = self.z.shape
        p = self.lag
        self.n_obs_ = X.shape[0]

        for eq_idx, eq_name in enumerate(self.columns):
            y = Y[:, eq_idx]

            prior_var = [1e4]  # intercept: diffuse, effectively unshrunk
            for l in range(1, p + 1):
                for j in range(k):
                    is_own = (j == eq_idx)
                    base = (self.lambda_overall / l) ** 2
                    prior_var.append(base if is_own else base * (self.lambda_cross ** 2))
            prior_precision = np.diag(1.0 / np.array(prior_var))

            beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta_ols
            dof = max(len(resid) - X.shape[1], 1)
            sigma2 = max(np.sum(resid ** 2) / dof, 1e-6)

            post_precision = (X.T @ X) / sigma2 + prior_precision
            post_cov = np.linalg.inv(post_precision)
            post_mean = post_cov @ (X.T @ y / sigma2)

            self.coefs_[eq_name] = post_mean
            self.coef_cov_[eq_name] = post_cov
            self.sigma2_[eq_name] = sigma2

        return self

    def coef_names(self):
        names = ['intercept']
        for l in range(1, self.lag + 1):
            for col in self.columns:
                names.append(f'{col}(t-{l})')
        return names

    def summary(self):
        names = self.coef_names()
        print(f"\nBayesian VAR summary (n={self.n_obs_}, lag={self.lag}, "
              f"lambda_overall={self.lambda_overall}, lambda_cross={self.lambda_cross})")
        for eq in self.columns:
            print(f"\nEquation: {eq}(t)")
            print("-" * 60)
            mean = self.coefs_[eq]
            se = np.sqrt(np.diag(self.coef_cov_[eq]))
            for name, m, s in zip(names, mean, se):
                lo, hi = m - 1.96 * s, m + 1.96 * s
                flag = "" if lo <= 0 <= hi else "  *"
                print(f"  {name:28} = {m:8.4f}   95% credible [{lo:7.4f}, {hi:7.4f}]{flag}")

    def granger_check(self, cause, effect):
        """
        Does `cause`'s lagged coefficient(s) in the `effect` equation stay
        credibly away from zero? A lightweight, single-model analog of a
        Granger-causality F-test, consistent with the Bayesian shrinkage
        framework already in use rather than a separate nested-OLS test.
        """
        names = self.coef_names()
        mean = self.coefs_[effect]
        se = np.sqrt(np.diag(self.coef_cov_[effect]))
        hits = []
        for name, m, s in zip(names, mean, se):
            if name.startswith(f'{cause}(t-'):
                lo, hi = m - 1.96 * s, m + 1.96 * s
                hits.append((name, m, lo, hi, not (lo <= 0 <= hi)))
        any_significant = any(h[4] for h in hits)
        verdict = "supports" if any_significant else "does NOT support"
        print(f"\n{cause} -> {effect}: {verdict} a predictive (Granger-style) relationship")
        for name, m, lo, hi, sig in hits:
            flag = " *" if sig else ""
            print(f"    {name:28} = {m:7.4f}  [{lo:7.4f}, {hi:7.4f}]{flag}")
        return any_significant

    def impulse_response(self, shock_var, steps=6):
        """
        One-standard-deviation shock to `shock_var` at t=0, propagated
        through the fitted system (standardized units). Non-orthogonalized
        (no Cholesky ordering) -- fine for a first look at propagation
        shape, not a causal-ordering claim.
        """
        k = len(self.columns)
        p = self.lag
        A = np.zeros((p, k, k))
        c = np.zeros(k)
        for eq_idx, eq in enumerate(self.columns):
            beta = self.coefs_[eq]
            c[eq_idx] = beta[0]
            for l in range(1, p + 1):
                A[l - 1, eq_idx, :] = beta[1 + (l - 1) * k: 1 + l * k]

        history = np.zeros((p, k))
        shock_idx = self.columns.index(shock_var)
        history[-1, shock_idx] = 1.0

        path = [history[-1].copy()]
        for _ in range(steps):
            y_new = c.copy()
            for l in range(1, p + 1):
                y_new += A[l - 1] @ history[-l]
            history = np.vstack([history[1:], y_new])
            path.append(y_new)
        return pd.DataFrame(path, columns=self.columns, index=range(0, steps + 1))




class Regime:
    """
    Three regime types, per the project's stated research question
    (structural reform vs. captured elite bias vs. unresponsive
    repression). Each regime is implemented as a *weighting* on the same
    fitted response function, not a separate model per regime -- this
    keeps the comparison honest: differences in simulated outcomes come
    from how the state weighs its inputs, not from swapping in an
    unrelated equation per regime.
    """
    REPRESENTATIVE = "representative"  # responds to fitted function as calibrated
    CAPTURED = "captured"              # protest signal damped, firm-profit signal upweighted
    DICTATORSHIP = "dictatorship"      # protest/gini signal near-zero, repression high

    ALL = (REPRESENTATIVE, CAPTURED, DICTATORSHIP)

    # (protest_weight_multiplier, police_intensity) per regime. Captured
    # states don't ignore conflict outright (Acemoglu & Robinson-style
    # elites still fear uprising) but respond far less per unit of
    # protest than a representative state; dictatorship both damps the
    # policy channel and raises repression, which feeds back to suppress
    # protest directly (Worker.risk_perception) rather than through
    # policy at all.
    _PARAMS = {
        REPRESENTATIVE: dict(protest_weight=1.0, police_intensity=1.0),
        CAPTURED:       dict(protest_weight=0.35, police_intensity=1.3),
        DICTATORSHIP:   dict(protest_weight=0.05, police_intensity=2.5),
    }

    @classmethod
    def params(cls, regime):
        if regime not in cls._PARAMS:
            raise ValueError(f"Unknown regime '{regime}'. Choose from {cls.ALL}")
        return cls._PARAMS[regime]


class State(mesa.Agent):
    """
    ABM state agent. Plugs in whichever function won the fitting
    comparison above (response_fn, response_params) rather than a
    hardcoded polynomial -- selected empirically, not assumed.

    Two things the original pseudocode was missing that matter for
    "modern ABM standards": (1) the response isn't purely deterministic --
    residual noise is bootstrapped from the fit's own empirical residuals,
    matching the project's math spec (R(t) = f(...) + epsilon(t)) instead
    of silently dropping the noise term; (2) regime type actually changes
    simulated behavior via the Regime weighting above, closing the loop
    the README describes (repression -> worker risk perception ->
    protest -> state observation) rather than leaving "regimes" as a
    label with no mechanism.
    """

    def __init__(self, model, response_fn, response_params, regime=Regime.REPRESENTATIVE,
                 residuals=None, tax_rate=0.25, domain_bounds=None, rng=None):
        super().__init__(model)
        self.random_gen = rng if rng is not None else np.random.default_rng()

        self.response_fn = response_fn
        self.response_params = response_params
        self.regime = regime
        regime_params = Regime.params(regime)
        self.protest_weight = regime_params['protest_weight']
        self.police_intensity = regime_params['police_intensity']

        # Empirical residuals from the fit (if provided) let us draw
        # realistic noise each step instead of either omitting epsilon(t)
        # or inventing an arbitrary noise distribution.
        self.residuals = residuals if residuals is not None else np.array([0.0])

        # Training-domain bounds (StateResponseFitter.domain_bounds_) --
        # inputs are clipped to these before response_fn ever sees them
        # (see decide_policy). "Bounded exponential": the fitted shape
        # itself is whatever won on AIC in-range (exponential currently),
        # left untouched -- it is simply never *evaluated* outside where
        # the data could inform it, rather than being swapped for a
        # globally-bounded functional form. If not provided, no clipping
        # is applied (useful for quick smoke tests) -- flagged since that
        # reintroduces the extrapolation risk this exists to prevent.
        self.domain_bounds = domain_bounds
        if self.domain_bounds is None:
            import warnings as _warnings
            _warnings.warn(
                "State created without domain_bounds -- response_fn will be "
                "evaluated unclipped and can extrapolate arbitrarily far "
                "outside its training range. Pass fitter.domain_bounds_ in "
                "production runs.", stacklevel=2,
            )

        self.tax_rate = tax_rate
        self.redistribution = 0.0
        self.past_protests = []
        self.avg_wage = None
        self.gini = None

        # for Worker._adapt_threshold's reinforcement signal
        self._last_redistribution = 0.0

    def observe(self, avg_wage, gini, protest_intensity, growth):
        self.avg_wage = avg_wage
        self.gini = gini
        self.growth = growth
        self.past_protests.append(protest_intensity)

    def _clip_to_training_domain(self, protest, gini, growth):
        """
        Clips each input to the [min, max] actually observed in the
        historical fitting data (see StateResponseFitter.domain_bounds_).
        Values inside the observed range pass through unchanged; values
        outside are pinned to the nearest boundary, so response_fn is
        only ever asked to extrapolate as far as "flat past the edge of
        what we've seen," never into genuinely unvalidated territory.
        This is what keeps an unbounded-shaped function (exponential)
        from exploding once the ABM's own dynamics (e.g. wealth
        compounding) push simulated Gini past anything in 1960-2025 US
        data.
        """
        if self.domain_bounds is None:
            return protest, gini, growth
        p_lo, p_hi = self.domain_bounds.get('protest_lag', (-np.inf, np.inf))
        g_lo, g_hi = self.domain_bounds.get('gini', (-np.inf, np.inf))
        gr_lo, gr_hi = self.domain_bounds.get('growth', (-np.inf, np.inf))
        return (
            float(np.clip(protest, p_lo, p_hi)),
            float(np.clip(gini, g_lo, g_hi)),
            float(np.clip(growth, gr_lo, gr_hi)),
        )

    def decide_policy(self, lag_periods=1):
        protest_lag = (
            self.past_protests[-lag_periods] if len(self.past_protests) >= lag_periods else 0.0
        )
        # regime-weighted protest input -- captured/dictatorship states
        # see a damped version of the same signal, not a different signal
        weighted_protest = protest_lag * self.protest_weight

        clipped_protest, clipped_gini, clipped_growth = self._clip_to_training_domain(
            weighted_protest, self.gini, self.growth
        )

        X = np.array([[clipped_protest], [clipped_gini], [clipped_growth]])
        target = float(self.response_fn(X, *self.response_params)[0])

        # bootstrapped noise term, matching R(t) = f(...) + epsilon(t)
        noise = self.random_gen.choice(self.residuals)
        target += noise

        self.redistribution = max(0.0, target)

        # feed reinforcement signal to workers before they act next step
        self.model.last_redistribution_delta = self.redistribution - self._last_redistribution
        self._last_redistribution = self.redistribution

    def redistribute(self, workers):
        """
        Distributes redistribution to the lowest-earning quartile of
        currently employed-or-not workers -- proportional wage subsidy,
        not a hardcoded UBI amount.
        """
        if not workers:
            return
        recipients = sorted(workers, key=lambda w: w.wage)[: max(1, len(workers) // 4)]
        subsidy = self.redistribution / len(recipients)
        for w in recipients:
            w.wage += subsidy

    def step(self):
        """Mesa scheduling hook -- observe/decide/redistribute are called
        explicitly by the Model with the current-period aggregates, since
        those aggregates depend on all Worker/Firm steps having already
        run this tick."""
        pass


# ============================================================================
# __main__ — run the full pipeline, print diagnostics
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STATE RESPONSE FUNCTION — DATA PREP + CALIBRATION")
    print("=" * 70)

    annual = load_and_prepare_data("results/us_state_response_data.csv",
                                    protest_lag_years=1)

    print(f"\nAnnual observations: {len(annual)} years "
          f"({annual.index.min()}-{annual.index.max()})")

    print("\nColumn coverage after annual aggregation:")
    coverage = (annual.notna().sum() / len(annual) * 100).sort_values(ascending=False)
    for col, pct in coverage.items():
        print(f"  {col:32} {pct:5.1f}%")

    X, y, years = build_model_inputs(
        annual,
        y_col='redistribution_pct_gdp',
        x_cols=('protest_lag', 'gini_coefficient', 'gdp_growth_yoy_pct'),
    )

    print(f"\nUsable rows for fitting after dropna: {len(y)}")
    if len(y) < 8:
        print("⚠ Very small sample for a 3-4 parameter model — treat coefficients "
              "as exploratory, not confirmatory, until coverage improves.")

    print("\n" + "-" * 70)
    print("MODEL COMPARISON")
    print("-" * 70)
    fitter = StateResponseFitter(X, y, years)
    best_name, results = fitter.fit_and_compare()

    if best_name:
        fitter.summarize(best_name)
        # Also print the simplest (linear) model for reference even if it
        # didn't win, since it's the easiest to cite in the paper.
        if best_name != 'linear' and 'linear' in results:
            fitter.summarize('linear')

    print("\n" + "-" * 70)
    print("JOINT SYSTEM — BAYESIAN VAR (Protest <-> Redistribution <-> Gini)")
    print("-" * 70)

    var_data = prepare_var_data(annual)
    print(f"Fitting on {len(var_data)} first-differenced annual observations")

    bvar = BayesianVAR(var_data, lag=1, lambda_overall=0.2, lambda_cross=0.5).fit()
    bvar.summary()

    print("\n" + "-" * 70)
    print("GRANGER-STYLE CHECKS")
    print("-" * 70)
    d_protest = 'd_protest_intensity_score'
    d_redist = 'd_redistribution_pct_gdp'
    d_gini = 'd_gini_coefficient'
    bvar.granger_check(d_protest, d_redist)
    bvar.granger_check(d_redist, d_protest)
    bvar.granger_check(d_gini, d_protest)
    bvar.granger_check(d_gini, d_redist)

    print("\n" + "-" * 70)
    print("IMPULSE RESPONSE — one s.d. protest shock, propagated 6 periods")
    print("-" * 70)
    irf = bvar.impulse_response(d_protest, steps=6)
    print(irf.round(3).to_string())