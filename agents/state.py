"""
state.py
State Fiscal Response Function — data prep + calibration.

Pipeline:
  1. load_and_prepare_panel() -> reads results/combined_long_panel.csv
                                  (country, year, label, value), pivots
                                  each country to its own annual wide frame,
                                  and builds R(t), Protest(t-lag), Gini(t),
                                  Growth(t), and PoliticalViolence(t-lag).
  2. StateResponseFitter      -> fits linear / logistic / exponential /
                                  GAM candidates for ONE country's data,
                                  with a variable predictor count (see
                                  below), coefficient CIs, and residual
                                  diagnostics.
  3. fit_all_countries()      -> loops StateResponseFitter across every
                                  country in the panel that clears a
                                  per-candidate minimum-N floor, so
                                  data-poor countries still get whatever
                                  candidates are legitimately identifiable
                                  instead of an all-or-nothing gate.
  4. State                    -> ABM agent shell, now driven by whatever
                                  predictor set a given country's fit
                                  actually used (4 or 5 inputs), rather
                                  than a hardcoded 4-tuple.

===============================================================================
NOTES
===============================================================================
Non-US countries have a structurally THINNER and
DIFFERENT schema than the US, not just fewer rows of the same columns:

  - Redistribution: every country (US included, after the collector's
    rename) carries ONE 'redist_gdp_pct' label -- the US's is built by
    summing four program components upstream and non-US countries' are a
    single OECD/welfare-expenditure series, but by the time it reaches this
    file it's the same shape (one number, %GDP, per country-year). Use it
    directly. Do NOT reconstruct it from components here.

  - Protest: the US has TWO measures -- 'protest_intensity_score' (a
    log1p + min-max score built upstream from strike workers/days-idle,
    US-only) and 'civil_unrest_events' (raw annual event counts from
    CNTS-style sources, same measure every OTHER country has too, US
    included). For a cross-country/cross-regime fitter to compare
    apples to apples, we build our OWN harmonized protest measure from
    'civil_unrest_events' for every country (see build_unrest_score
    below) rather than using the US's bespoke strike-based score. This
    means the fitted US protest coefficient here is not directly the
    same measurement as 'protest_intensity_score' elsewhere in this
    codebase -- that's intentional, for cross-country comparability, and
    worth remembering if the two are ever compared side by side.

  - Political violence: 'political_violence_score' (Political Terror
    Scale-style, state violence AGAINST civilians -- the opposite
    causal direction from civil_unrest_events, which is civilians
    acting against the state) exists for most non-US countries but NOT
    currently for the US. It is wired in here as its OWN, separate
    predictor (never merged with protest) for every country generically
    -- there is no US-specific carve-out in the code. If/when US
    political-violence data is added upstream, it will just start being
    used, with no code change needed here. Until then it's simply
    absent from the US's fitted predictor set, exactly like any other
    country whose coverage doesn't clear the inclusion floor below.

  - Regime labels (representative / captured / dictatorship) are
    DELIBERATELY NOT encoded as fitting inputs anywhere in this file.
    They live in regime_periods.csv as a separate, manually-reviewed
    reference table, joined against fitted results only for
    post-hoc grouping/inspection. The point is to let the data speak
    first and inspect it by regime after, not to bend the data toward
    a regime assumption. See load_regime_periods() / regime_periods.csv.
===============================================================================
"""

import numpy as np
import pandas as pd
import mesa
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller


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
# same machinery, no need for a separate linear-only path. Unaffected by
# the multi-country / variable-predictor-count changes below: it treats
# model_fn as a black box called as model_fn(X, *params), which still holds
# regardless of how many predictors X carries for a given country.

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
    with sample size."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def hac_sandwich_covariance(jacobian, residuals, maxlags):
    """
    Newey-West HAC sandwich covariance for a (nonlinear) least-squares fit.
    Returns a (k, k) covariance matrix for the parameter estimates -- same
    role as pcov from curve_fit, but robust to autocorrelated residuals.
    """
    scores = jacobian * residuals[:, None]  # per-observation contributions, (n, k)

    meat = scores.T @ scores  # lag 0
    for lag in range(1, maxlags + 1):
        w = 1 - lag / (maxlags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += w * (gamma + gamma.T)

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
# 1. LONG-PANEL LOADING + PER-COUNTRY ANNUAL PREP
# ============================================================================

# Minimum non-null years a column needs, for a given country, before we'll
# even consider it as a candidate predictor for that country. Below this,
# dropna() in build_model_inputs would gut the sample anyway -- better to
# drop the predictor and keep more years than keep the predictor and lose
# the country.
MIN_PREDICTOR_COVERAGE_YEARS = 10


def build_unrest_score(civil_unrest_events):
    """
    Harmonized cross-country protest measure, built from raw annual
    civil_unrest_events counts: log1p (event counts are heavily
    right-skewed -- a handful of huge-unrest years otherwise dominate),
    then min-max rescaled WITHIN this one country's own series to [0, 1].

    Per-country min-max (not a global z-score across countries) is a
    deliberate choice: it measures each country's protest activity
    relative to ITS OWN historical range, which is the right comparison
    for "did the state respond more after an unusually large protest
    wave, for that country" -- not "who had more protests in absolute
    terms," which raw counts from very different data sources/population
    sizes can't support anyway. This is the same log1p + min-max recipe
    already used upstream for the US's strike-based
    protest_intensity_score (see calculate_strike_severity in the
    collection script) -- kept consistent rather than inventing a
    second normalization scheme.

    Returns np.nan-filled series unchanged if there's insufficient
    variation to rescale (all-equal or all-missing), same guard as the
    upstream strike-severity calc.
    """
    if civil_unrest_events is None:
        return None
    logged = np.log1p(civil_unrest_events.clip(lower=0))
    valid = logged.dropna()
    if len(valid) == 0 or valid.max() == valid.min():
        return pd.Series(np.nan, index=civil_unrest_events.index)
    return (logged - valid.min()) / (valid.max() - valid.min())


def _prep_one_country(annual, protest_lag_years=1):
    """
    Given one country's annual wide frame (columns = labels straight off
    the long panel), add every derived column the fitter needs. Mutates
    and returns `annual`.
    """
    # ---- Redistribution: used directly, never reconstructed here ----
    if 'redist_gdp_pct' in annual.columns:
        annual['redistribution_pct_gdp'] = annual['redist_gdp_pct']
    else:
        annual['redistribution_pct_gdp'] = np.nan
    annual['redistribution_lag'] = annual['redistribution_pct_gdp'].shift(1)

    # ---- Growth: already a rate, never re-differenced ----
    if 'gdp_billions' in annual.columns:
        annual['gdp_growth_yoy_pct'] = annual['gdp_billions'].pct_change() * 100

    # ---- Harmonized protest measure (see build_unrest_score) ----
    if 'civil_unrest_events' in annual.columns:
        annual['unrest_score'] = build_unrest_score(annual['civil_unrest_events'])
        annual['protest_lag'] = annual['unrest_score'].shift(protest_lag_years)

    # ---- Political violence: own predictor, lagged the same way as
    # protest (state repression this year plausibly shapes next year's
    # policy response the same lag-logic way protest does). Not
    # log/min-max transformed -- PTS-style scores are already a bounded
    # small-integer scale (~1-5), not a skewed count. ----
    if 'political_violence_score' in annual.columns:
        annual['political_violence_lag'] = (
            annual['political_violence_score'].shift(protest_lag_years)
        )

    # ---- First differences (stationarity fix). ADF testing (see
    # check_stationarity) generally shows unit roots in the levels of
    # redistribution, gini, and protest/unrest -- fitting on levels
    # risks a spurious regression (two trending series, not a real
    # relationship). gdp_growth_yoy_pct is already a rate and is NOT
    # differenced again. ----
    annual['d_redistribution_pct_gdp'] = annual['redistribution_pct_gdp'].diff()
    annual['d_redistribution_lag'] = annual['d_redistribution_pct_gdp'].shift(1)

    if 'gini_coefficient' in annual.columns:
        annual['d_gini_coefficient'] = annual['gini_coefficient'].diff()

    if 'unrest_score' in annual.columns:
        d_unrest = annual['unrest_score'].diff()
        annual['d_protest_lag'] = d_unrest.shift(protest_lag_years)

    if 'political_violence_score' in annual.columns:
        d_pvs = annual['political_violence_score'].diff()
        annual['d_political_violence_lag'] = d_pvs.shift(protest_lag_years)

    return annual


def load_and_prepare_panel(csv_path="results/combined_long_panel.csv",
                            protest_lag_years=1):
    """
    Load the long-format (country, year, label, value) panel and return a
    dict: {country: annual_df}, one wide annual frame per country, each
    carrying every derived column _prep_one_country adds. No intermediate
    file is written.
    """
    long_df = pd.read_csv(csv_path)
    required = {'country', 'year', 'label', 'value'}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"combined_long_panel.csv missing columns: {missing}")

    by_country = {}
    for country, g in long_df.groupby('country'):
        wide = g.pivot_table(index='year', columns='label', values='value')
        wide = wide.sort_index()
        by_country[country] = _prep_one_country(wide, protest_lag_years=protest_lag_years)

    return by_country


def select_predictors_for_country(annual, min_coverage=MIN_PREDICTOR_COVERAGE_YEARS):
    """
    Decide this country's predictor set. Base four are attempted for
    every country; political_violence is included only if it clears the
    coverage floor -- so the US (no political_violence_score at all
    right now) and a data-poor country both degrade gracefully to the
    4-predictor form, while a well-covered country gets the 5th.

    Returns (x_cols, notes) where notes explains any exclusion, for
    transparent logging by the caller.
    """
    x_cols = []
    notes = []

    base = [
        ('d_protest_lag', 'protest (civil_unrest_events, harmonized)'),
        ('d_gini_coefficient', 'gini'),
        ('gdp_growth_yoy_pct', 'growth'),
        ('d_redistribution_lag', 'redistribution persistence'),
    ]
    for col, desc in base:
        if col in annual.columns and annual[col].notna().sum() >= min_coverage:
            x_cols.append(col)
        else:
            have = annual[col].notna().sum() if col in annual.columns else 0
            notes.append(f"excluded base predictor '{col}' ({desc}): "
                          f"only {have} non-null years (< {min_coverage})")

    optional_col = 'd_political_violence_lag'
    if optional_col in annual.columns and annual[optional_col].notna().sum() >= min_coverage:
        x_cols.append(optional_col)
        notes.append(f"included optional predictor '{optional_col}': "
                      f"{annual[optional_col].notna().sum()} non-null years")
    else:
        have = annual[optional_col].notna().sum() if optional_col in annual.columns else 0
        notes.append(f"political_violence not included ({have} non-null years "
                      f"< {min_coverage}) -- structurally available, just not "
                      f"currently covered for this country")

    return x_cols, notes


def build_model_inputs(annual, y_col='d_redistribution_pct_gdp', x_cols=None):
    """
    Extract (X, y) arrays for fitting, dropping any year with missing data
    in the required columns. Returns the aligned year index too.
    """
    if x_cols is None:
        raise ValueError("x_cols is required (use select_predictors_for_country)")
    needed = list(x_cols) + [y_col]
    missing = [c for c in needed if c not in annual.columns]
    if missing:
        raise ValueError(f"Missing required columns for fitting: {missing}")

    sub = annual[needed].dropna()
    if len(sub) == 0:
        raise ValueError(f"No overlapping non-null years across {needed}")
    X = sub[list(x_cols)].to_numpy().T  # shape (n_predictors, n_obs)
    y = sub[y_col].to_numpy()
    years = sub.index.to_numpy()
    return X, y, years


def check_collinearity(X, feature_names):
    """
    Pairwise correlation + VIF (variance inflation factor) across
    predictors. Unchanged in behavior; still generic to however many
    predictors a given country's X carries.
    """
    n_features = X.shape[0]
    corr = np.corrcoef(X)

    print("\nPairwise correlation between predictors:")
    header = "              " + "".join(f"{name:>14}" for name in feature_names)
    print(header)
    for i, name in enumerate(feature_names):
        row = "".join(f"{corr[i, j]:14.3f}" for j in range(n_features))
        print(f"  {name:12}{row}")

    print("\nVariance Inflation Factors (VIF > 5 concerning, > 10 problematic):")
    for i, name in enumerate(feature_names):
        others = [j for j in range(n_features) if j != i]
        Xi = X[i]
        Xo = X[others].T
        Xo_design = np.column_stack([np.ones(len(Xi)), Xo])
        beta, *_ = np.linalg.lstsq(Xo_design, Xi, rcond=None)
        pred = Xo_design @ beta
        ss_res = np.sum((Xi - pred) ** 2)
        ss_tot = np.sum((Xi - Xi.mean()) ** 2)
        r2_i = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1 / (1 - r2_i) if r2_i < 1 else np.inf
        flag = "  ⚠ HIGH" if vif > 10 else ("  ⚠ moderate" if vif > 5 else "")
        print(f"  {name:22} VIF = {vif:7.2f}{flag}")


def check_stationarity(series, name):
    """
    Augmented Dickey-Fuller test. Unchanged from the single-country
    version -- takes any series, generic to country.

    H0 (null): series has a unit root (non-stationary).
    p < 0.05: reject H0 -> stationary.
    p >= 0.05: possible unit root -- treat any levels-based fit
    involving this series with real suspicion.
    """
    clean = series.dropna()
    if len(clean) < 4:
        print(f"  ADF({name}): skipped, only {len(clean)} non-null obs")
        return None
    result = adfuller(clean, autolag='AIC')
    stat, pvalue, used_lag, nobs = result[0], result[1], result[2], result[3]
    verdict = "stationary" if pvalue < 0.05 else "possible unit root (non-stationary)"
    flag = "" if pvalue < 0.05 else "  ⚠"
    print(f"  ADF({name}): stat={stat:.3f}  p={pvalue:.4f}  "
          f"lags={used_lag}  n={nobs}  -> {verdict}{flag}")
    return pvalue < 0.05


# ============================================================================
# 2. STATE RESPONSE FITTER (variable predictor count)
# ============================================================================
#
# The old model_fn signatures hardcoded exactly 4 predictors
# (protest, gini, growth, redist_lag) by name. That doesn't work once
# political_violence is a 5th predictor for SOME countries and not
# others -- we can't have one fixed function signature per candidate
# shape anymore. Instead:
#
#   X[0] = protest (always the first "core" predictor)
#   X[1] = gini     (always the second "core" predictor)
#   X[2:] = every other included predictor, always additive
#           (growth, redist_lag, and political_violence when present)
#
# linear stays a straight dot-product + intercept regardless of width.
# logistic/exponential keep protest+gini in their saturating core (that
# structural choice -- protest and inequality are what plausibly
# saturate, growth/persistence/repression are additive context -- is
# unchanged from the original design) and add one linear coefficient
# per extra predictor. This means logistic/exponential now correctly
# USE growth and any other extra predictor -- the original
# exponential_model silently ignored growth entirely (it unpacked
# `growth` from X and never used it). That was an oversight, not a
# deliberate omission from the model spec in this file's own docstring
# (R(t) = f(Protest, Gini, Growth)), so it's fixed here rather than
# preserved for backward compatibility.


class StateResponseFitter:
    """
    Fits a family of candidate state response functions for ONE country
        R(t) = f(Protest(t-lag), Gini(t), Growth(t), Redist_lag(t), [PoliticalViolence(t-lag)])
    and compares them honestly: fixed AIC/BIC, coefficient CIs, and
    residual autocorrelation diagnostics.

    feature_names[0] must be the protest predictor, feature_names[1] the
    gini predictor -- everything after that is treated as additive
    "extra" context in the logistic/exponential forms. build_model_inputs
    + select_predictors_for_country already produce X/x_cols in this
    order, so callers normally don't need to think about it directly.
    """

    # Per-candidate minimum-N floor: k_params + this margin. A flat n>=8
    # gate (the old approach) treats a 5-param logistic and a 5-param
    # linear as equally identifiable at the same n, which isn't true --
    # this scales the floor to how many parameters that specific
    # candidate is actually asking the data to pin down, so a data-poor
    # country can still get a legitimate linear fit even where logistic/
    # GAM correctly get excluded rather than reported as a degenerate
    # number.
    MIN_N_MARGIN = 4
    GAM_MIN_N = 20  # GAM has no fixed parameter count to build a formula
                     # floor from (its effective DoF is data-dependent via
                     # the spline gridsearch), so it gets its own flat,
                     # stricter absolute floor instead.

    def __init__(self, X, y, years, feature_names):
        self.X = X          # shape (n_features, n_obs)
        self.y = y
        self.years = years
        self.feature_names = list(feature_names)
        self.n_extra = X.shape[0] - 2  # predictors beyond protest+gini
        self.n = len(y)
        self.results = {}
        self.skipped = {}  # candidate -> reason, for transparency
        self.domain_bounds_ = {
            name: (float(np.min(X[i])), float(np.max(X[i])))
            for i, name in enumerate(self.feature_names)
        }

    # ---- parametric candidate forms (n_extra-aware) ----
    @staticmethod
    def linear_model(X, *params):
        *coefs, intercept = params
        return np.dot(coefs, X) + intercept

    @staticmethod
    def logistic_model(X, *params):
        """Saturates at high protest/inequality; every other predictor
        (growth, redist_lag, political_violence if present) enters
        additively as context, same policy-persistence role as before."""
        n_extra = X.shape[0] - 2
        a, b, k, x0 = params[:4]
        extra_coefs = np.asarray(params[4:4 + n_extra])
        protest, gini = X[0], X[1]
        result = a / (1 + np.exp(-k * (protest + b * gini - x0)))
        if n_extra:
            result = result + extra_coefs @ X[2:]
        return result

    @staticmethod
    def exponential_model(X, *params):
        """Accelerates at extreme inequality; protest enters linearly
        (not inside the exponential -- only gini saturates this way);
        every other predictor is additive, same as logistic."""
        n_extra = X.shape[0] - 2
        a, b, c = params[:3]
        extra_coefs = np.asarray(params[3:3 + n_extra])
        protest, gini = X[0], X[1]
        result = a * np.exp(np.clip(b * gini, -50, 50)) + c * protest
        if n_extra:
            result = result + extra_coefs @ X[2:]
        return result

    def _aic_bic(self, y_pred, k):
        rss = np.sum((self.y - y_pred) ** 2)
        rss = max(rss, 1e-12)
        aic = self.n * np.log(rss / self.n) + 2 * k
        bic = self.n * np.log(rss / self.n) + k * np.log(self.n)
        return aic, bic, rss

    def _min_n_ok(self, name, k_params):
        floor = k_params + self.MIN_N_MARGIN
        if self.n < floor:
            reason = (f"n={self.n} < required {floor} (k={k_params} params + "
                      f"{self.MIN_N_MARGIN} margin)")
            self.skipped[name] = reason
            print(f"{name:12} | SKIPPED | {reason}")
            return False
        return True

    def _fit_parametric(self, name, model_fn, p0):
        if not self._min_n_ok(name, len(p0)):
            return
        try:
            popt, pcov = curve_fit(model_fn, self.X, self.y, p0=p0, maxfev=20000)
            y_pred = model_fn(self.X, *popt)
            r2 = r2_score(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            aic, bic, rss = self._aic_bic(y_pred, k=len(popt))

            perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
            ci = [(p - 1.96 * e, p + 1.96 * e) for p, e in zip(popt, perr)]

            dw = durbin_watson(self.y - y_pred)

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
            self.skipped[name] = f"fit failed: {e}"
            print(f"{name:12} | FAILED: {e}")

    def _fit_gam(self):
        if not PYGAM_AVAILABLE:
            return
        if self.n < self.GAM_MIN_N:
            reason = f"n={self.n} < GAM floor {self.GAM_MIN_N}"
            self.skipped['gam'] = reason
            print(f"{'gam':12} | SKIPPED | {reason}")
            return
        try:
            Xg = self.X.T  # pygam wants (n_obs, n_features)
            n_features = Xg.shape[1]
            terms = s(0)
            for i in range(1, n_features):
                terms = terms + s(i)
            gam = LinearGAM(terms).gridsearch(Xg, self.y, progress=False)
            y_pred = gam.predict(Xg)
            r2 = r2_score(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
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
            self.skipped['gam'] = f"fit failed: {e}"
            print(f"{'gam':12} | FAILED: {e}")

    _EXTRAPOLATION_SAFE = {'linear': True, 'logistic': True,
                            'exponential': False, 'gam': False}

    def fit_and_compare(self, extrapolation_safe_only=False):
        """
        Fits every candidate that clears its own min-N floor (see
        MIN_N_MARGIN / GAM_MIN_N above); candidates that don't clear it
        are skipped with a printed reason and recorded in self.skipped,
        not silently dropped. extrapolation_safe_only restricts final
        SELECTION only, not fitting -- see the original docstring logic
        (Burnham & Anderson 2002) preserved from the single-country
        version.
        """
        print(f"Fitting on n={self.n} annual observations "
              f"({self.years.min()}-{self.years.max()}), "
              f"{2 + self.n_extra} predictors: {self.feature_names}\n")

        n_extra = self.n_extra
        self._fit_parametric('linear', self.linear_model,
                              p0=[1] * (2 + n_extra) + [np.mean(self.y)])
        self._fit_parametric('logistic', self.logistic_model,
                              p0=[np.ptp(self.y), 1, 1, np.mean(self.X[1])] + [0.5] * n_extra)
        self._fit_parametric('exponential', self.exponential_model,
                              p0=[1, 1, 1] + [0.5] * n_extra)
        self._fit_gam()

        if not self.results:
            print("\n✗ No candidate model fit successfully "
                  f"(skipped: {self.skipped}).")
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
                          f"bounded/saturating functional form.)")

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

        extra_labels = [f"extra_{i} ({name})" for i, name in enumerate(self.feature_names[2:])]
        labels = {
            'linear': [f"coef ({name})" for name in self.feature_names] + ['intercept'],
            'logistic': ['a (scale)', 'b (gini weight)', 'k (steepness)', 'x0 (midpoint)'] + extra_labels,
            'exponential': ['a (scale)', 'b (gini exponent)', 'c (protest)'] + extra_labels,
        }.get(model_name, [f'param_{i}' for i in range(len(res['params']))])

        print(f"\n{model_name} — coefficient summary")
        print("-" * 60)
        print(f"  {'':28}   {'naive CI (iid residuals)':32} {'HAC/Newey-West CI (' + str(res['nw_lags']) + ' lags)'}")
        for label, val, (lo, hi), (hlo, hhi) in zip(
                labels, res['params'], res['param_ci_95'], res['param_ci_95_hac']):
            naive_str = f"[{lo:8.4f}, {hi:8.4f}]"
            hac_str = f"[{hlo:8.4f}, {hhi:8.4f}]"
            flips = (lo <= 0 <= hi) != (hlo <= 0 <= hhi)
            flag = "  ← CI vs 0 changes!" if flips else ""
            print(f"  {label:28} = {val:10.4f}   {naive_str:32} {hac_str}{flag}")
        print(f"  R² = {res['r2']:.3f}   RMSE = {res['rmse']:.4f}   "
              f"AIC = {res['aic']:.2f}   BIC = {res['bic']:.2f}")


# ============================================================================
# 2b. CROSS-COUNTRY ORCHESTRATOR
# ============================================================================

# Below this, even a linear+intercept fit (5 params minimum with all
# predictors) is not worth reporting -- this is the absolute floor before
# StateResponseFitter's own per-candidate floors even get a chance to run.
ABSOLUTE_MIN_N = 6


def fit_all_countries(panel_csv="results/combined_long_panel.csv",
                       protest_lag_years=1,
                       extrapolation_safe_only=True,
                       min_predictor_coverage=MIN_PREDICTOR_COVERAGE_YEARS):
    """
    Loop the single-country StateResponseFitter across every country in
    the long panel. Countries/candidates that don't clear their data
    floor are skipped with a printed, recorded reason -- never silently
    dropped and never force-fit past what the data supports.

    Returns {country: {'fitter': StateResponseFitter, 'best': str|None,
                        'results': dict, 'predictors': list, 'n': int}}
    for every country that produced at least one usable fit; countries
    that couldn't clear ABSOLUTE_MIN_N or had no redistribution data at
    all are omitted (reason printed at the time, not returned silently --
    check the console output for the "skip:" lines if a country you
    expected is missing from the returned dict).
    """
    by_country = load_and_prepare_panel(panel_csv, protest_lag_years=protest_lag_years)
    all_results = {}

    for country in sorted(by_country):
        annual = by_country[country]
        print(f"\n{'=' * 70}\n{country}\n{'=' * 70}")

        x_cols, notes = select_predictors_for_country(annual, min_coverage=min_predictor_coverage)
        for note in notes:
            print(f"  {note}")

        try:
            X, y, years = build_model_inputs(annual, x_cols=tuple(x_cols))
        except ValueError as e:
            print(f"  skip: {e}")
            continue

        if len(y) < ABSOLUTE_MIN_N:
            print(f"  skip: only {len(y)} usable annual obs "
                  f"(< ABSOLUTE_MIN_N={ABSOLUTE_MIN_N})")
            continue

        feature_names = ['protest_lag', 'gini'] + [
            c for c in x_cols if c not in ('d_protest_lag', 'd_gini_coefficient')
        ]
        fitter = StateResponseFitter(X, y, years, feature_names=feature_names)
        best_name, results = fitter.fit_and_compare(extrapolation_safe_only=extrapolation_safe_only)

        if best_name is None:
            print(f"  no candidate fit successfully for {country}")
            continue

        all_results[country] = {
            'fitter': fitter, 'best': best_name, 'results': results,
            'predictors': x_cols, 'n': len(y),
        }

    return all_results


# ============================================================================
# 2c. REGIME PERIODS (reference table, NEVER a fitting input)
# ============================================================================
#
# See regime_periods.csv alongside this file. Deliberately NOT sticky /
# assumed -- it's a manually-reviewed, editable reference table of
# (country, year range, regime label, confidence, note), joined against
# fitted/observed data only for post-hoc grouping ("does the fitted
# response differ before/after this transition"), never fed into the
# fitter itself. Boundaries the data doesn't clearly support are marked
# confidence='low' rather than guessed at with false precision.

def load_regime_periods(csv_path="data/format/regime_periods.csv"):
    path = Path(csv_path)
    if not path.exists():
        print(f"⚠ {csv_path} not found -- regime grouping unavailable, "
              f"nothing fabricated. Fitted results are unaffected.")
        return pd.DataFrame(columns=['country', 'start_year', 'end_year',
                                      'regime_label', 'confidence', 'note'])
    return pd.read_csv(path)


def annotate_with_regime(annual, country, regime_periods):
    """
    Left-joins a per-year regime_label/confidence onto one country's
    annual df, purely for inspection (e.g. splitting residuals or
    fitted coefficients by regime-period after the fact). Years outside
    every listed range, or countries with no rows in regime_periods.csv
    yet, get NaN -- not defaulted to 'representative' or any other
    assumed label.
    """
    annual = annual.copy()
    annual['regime_label'] = np.nan
    annual['regime_confidence'] = np.nan
    rows = regime_periods[regime_periods['country'] == country]
    for _, row in rows.iterrows():
        mask = (annual.index >= row['start_year']) & (annual.index <= row['end_year'])
        annual.loc[mask, 'regime_label'] = row['regime_label']
        annual.loc[mask, 'regime_confidence'] = row.get('confidence', np.nan)
    return annual


# ============================================================================
# 3. BAYESIAN VAR — joint Protest <-> Redistribution <-> Gini system
# ============================================================================
# Unchanged in mechanics from the single-country version; prepare_var_data
# now takes one country's already-prepared annual df (from
# load_and_prepare_panel) instead of a US-only global `annual`, so it's
# usable per-country in the same loop shape as fit_all_countries above.

def prepare_var_data(annual, columns=('unrest_score',
                                       'redistribution_pct_gdp',
                                       'gini_coefficient')):
    """
    First-differences the requested columns -- the stationarity fix for
    the trend-driven collinearity flagged in the single-equation fits --
    and drops the resulting leading NaN row. Defaults to unrest_score
    now (the harmonized cross-country protest measure) rather than the
    US-only protest_intensity_score the single-country version used.
    """
    present = [c for c in columns if c in annual.columns]
    missing = [c for c in columns if c not in annual.columns]
    if missing:
        print(f"  ⚠ prepare_var_data: {missing} not available for this country, "
              f"proceeding with {present}")
    sub = annual[present].dropna()
    diffed = sub.diff().dropna()
    diffed.columns = [f'd_{c}' for c in diffed.columns]
    return diffed


class BayesianVAR:
    """
    Small Bayesian VAR with a Minnesota-style shrinkage prior. Unchanged
    from the single-country version -- operates on whatever columns
    prepare_var_data hands it, no US-specific assumptions inside.
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

            prior_var = [1e4]
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
    repression). Each regime is a *weighting* on the same fitted
    response function, not a separate model per regime.

    NOTE: these labels are the ABM's simulation-time regime dial (what
    weighting a simulated State agent uses going forward), and are NOT
    the same thing as regime_periods.csv, which is a historical/
    observational reference table for grouping already-fitted results
    by what regime a real country arguably was in during a given span.
    Don't conflate the two -- one drives simulation, the other is just
    for looking at history.
    """
    REPRESENTATIVE = "representative"
    CAPTURED = "captured"
    DICTATORSHIP = "dictatorship"

    ALL = (REPRESENTATIVE, CAPTURED, DICTATORSHIP)

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
    comparison for a given country (response_fn, response_params,
    feature_names) -- feature_names now drives EVERYTHING about what
    inputs decide_policy builds, rather than a hardcoded 4-tuple, since
    a country's fitted model may have 4 or 5 predictors depending on
    political_violence coverage. The Worker-facing inputs a State needs
    each tick (protest, gini, growth, redistribution persistence,
    optionally political violence) are supplied via `observe()`; State
    only builds X for whichever of those the calibrated fit actually
    used.
    """

    # Maps a feature_name (as produced by StateResponseFitter /
    # select_predictors_for_country) to the attribute on this State
    # instance that holds the CURRENT-TICK value for it. Every entry
    # here must correspond to something decide_policy computes below.
    _FEATURE_TO_ATTR = {
        'protest_lag': '_cur_d_protest',
        'gini': '_cur_d_gini',
        'gdp_growth_yoy_pct': 'growth',
        'd_redistribution_lag': '_last_redistribution_delta',
        'd_political_violence_lag': '_cur_d_political_violence',
    }

    def __init__(self, model, response_fn, response_params, feature_names,
                 regime=Regime.REPRESENTATIVE,
                 residuals=None, tax_rate=0.25, domain_bounds=None, rng=None,
                 redistribution_enabled=True):
        super().__init__(model)
        self.random_gen = rng if rng is not None else np.random.default_rng()

        self.response_fn = response_fn
        self.response_params = response_params
        # Whatever predictor set this country's fit actually used, in
        # the exact order response_fn expects. E.g. ['protest_lag',
        # 'gini', 'gdp_growth_yoy_pct', 'd_redistribution_lag'] for a
        # country without political-violence coverage, or that list
        # plus 'd_political_violence_lag' for one with it.
        self.feature_names = list(feature_names)
        unknown = [f for f in self.feature_names if f not in self._FEATURE_TO_ATTR]
        if unknown:
            raise ValueError(
                f"State got feature_names {unknown} with no known mapping in "
                f"_FEATURE_TO_ATTR -- add it there before calibrating a State "
                f"on a fit that includes this predictor."
            )

        self.regime = regime
        self.redistribution_enabled = redistribution_enabled
        regime_params = Regime.params(regime)
        self.protest_weight = regime_params['protest_weight']
        self.police_intensity = regime_params['police_intensity']

        self.residuals = residuals if residuals is not None else np.array([0.0])

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
        self.past_political_violence = []
        self.avg_wage = None
        self.gini = None
        self.growth = 0.0

        self._last_redistribution = 0.0
        self._last_redistribution_delta = 0.0
        self._prev_gini = None
        self._cur_d_protest = 0.0
        self._cur_d_gini = 0.0
        self._cur_d_political_violence = 0.0

    def observe(self, avg_wage, gini, protest_intensity, growth, political_violence=None):
        """political_violence is optional -- only needs to be supplied
        by the Model if this State's calibrated feature_names actually
        includes 'd_political_violence_lag'; otherwise it's ignored."""
        self.avg_wage = avg_wage
        self.gini = gini
        self.growth = growth
        self.past_protests.append(protest_intensity)
        if political_violence is not None:
            self.past_political_violence.append(political_violence)

    def _clip_to_training_domain(self, values_by_feature):
        """values_by_feature: dict {feature_name: raw_value}. Returns a
        dict with the same keys, clipped to this State's domain_bounds
        (if any) for that feature."""
        if self.domain_bounds is None:
            return dict(values_by_feature)
        clipped = {}
        for name, val in values_by_feature.items():
            lo, hi = self.domain_bounds.get(name, (-np.inf, np.inf))
            clipped[name] = float(np.clip(val, lo, hi))
        return clipped

    def decide_policy(self, lag_periods=1):
        """
        response_fn predicts d_redistribution (a CHANGE), fitted on
        first-differenced series. Builds X dynamically from whatever
        feature_names this State was calibrated with -- 4 predictors for
        a country without political-violence coverage, 5 for one with
        it -- rather than assuming a fixed shape.
        """
        if not self.redistribution_enabled:
            self.redistribution = 0.0
            self._last_redistribution = 0.0
            self._last_redistribution_delta = 0.0
            self.model.last_redistribution_delta = 0.0
            self._prev_gini = self.gini
            return

        protest_now = (
            self.past_protests[-lag_periods] if len(self.past_protests) >= lag_periods else 0.0
        )
        protest_prev = (
            self.past_protests[-lag_periods - 1]
            if len(self.past_protests) >= lag_periods + 1 else 0.0
        )
        self._cur_d_protest = (protest_now - protest_prev) * self.protest_weight
        self._cur_d_gini = (self.gini - self._prev_gini) if self._prev_gini is not None else 0.0

        if 'd_political_violence_lag' in self.feature_names:
            pv_now = (
                self.past_political_violence[-lag_periods]
                if len(self.past_political_violence) >= lag_periods else 0.0
            )
            pv_prev = (
                self.past_political_violence[-lag_periods - 1]
                if len(self.past_political_violence) >= lag_periods + 1 else 0.0
            )
            self._cur_d_political_violence = pv_now - pv_prev

        raw_values = {
            name: getattr(self, self._FEATURE_TO_ATTR[name])
            for name in self.feature_names
        }
        clipped = self._clip_to_training_domain(raw_values)

        X = np.array([[clipped[name]] for name in self.feature_names])
        predicted_delta = float(self.response_fn(X, *self.response_params)[0])

        noise = self.random_gen.choice(self.residuals)
        predicted_delta += noise

        new_redistribution = max(0.0, self._last_redistribution + predicted_delta)

        self.model.last_redistribution_delta = new_redistribution - self._last_redistribution
        self._last_redistribution_delta = new_redistribution - self._last_redistribution
        self._last_redistribution = new_redistribution
        self.redistribution = new_redistribution
        self._prev_gini = self.gini

    def redistribute(self, workers):
        """Distributes redistribution to the lowest-earning quartile of
        currently employed-or-not workers -- proportional wage subsidy."""
        if not workers:
            return

        for w in workers:
            w.transfer_income = 0.0
        recipients = sorted(workers, key=lambda w: w.wage)[: max(1, len(workers) // 4)]
        subsidy = self.redistribution / len(recipients)
        for w in recipients:
             w.transfer_income = subsidy

    def step(self):
        """Mesa scheduling hook -- observe/decide/redistribute are called
        explicitly by the Model with the current-period aggregates."""
        pass


# ============================================================================
# __main__ — run the cross-country pipeline, print diagnostics
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STATE RESPONSE FUNCTION — CROSS-COUNTRY DATA PREP + CALIBRATION")
    print("=" * 70)

    all_results = fit_all_countries("results/combined_long_panel.csv",
                                     protest_lag_years=1,
                                     extrapolation_safe_only=True)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS COUNTRIES")
    print("=" * 70)
    regime_periods = load_regime_periods("data/format/regime_periods.csv")
    for country, res in all_results.items():
        best = res['best']
        r2 = res['results'][best]['r2'] if best else float('nan')
        n_regime_rows = len(regime_periods[regime_periods['country'] == country])
        print(f"  {country:16} best={best or '—':12} n={res['n']:3} "
              f"predictors={res['predictors']} "
              f"R²={r2:.3f}  (regime_periods.csv rows: {n_regime_rows})")

    skipped_countries = set()  # populated implicitly via console "skip:" lines above
    print("\n(See console output above for any country skipped entirely, and "
          "each fitter's `.skipped` dict for individual candidates skipped "
          "within a country that did produce a fit.)")