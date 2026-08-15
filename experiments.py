"""
experiments.py

Experiment suite for the state-fiscal-response ABM. Fits the state
response function + BVAR ONCE (via StateResponseFitter/BayesianVAR in
agents/state.py) and reuses that single fit across every regime and
every replicate seed below -- regime differences in the results come
only from the Regime weighting + police_intensity (agents/state.py),
never from each run silently landing on a slightly different fit.

Implemented this round:
  1. Historical validation          -- run_historical_validation()
  2. Regime comparison, N seeds     -- run_regime_comparison()
  3. Reform vs. revolution test     -- reform_vs_revolution_test()
  4. Lightweight sensitivity sweep  -- sensitivity_sweep()
  5. Permutation null check         -- permutation_null_check()

NOT implemented this round -- flagged, not silently skipped (see the
final print block in __main__ and the chat notes this was planned
against):
  - Period-by-period / rolling-window refit of the empirical response
    function and BVAR on real historical data (pre/post-1980 split etc.)
  - Compositional / indirect correlation analysis (spending mix shifts
    vs. protest, rather than only total redistribution level)
  - Full per-permutation BVAR refit for the null check below (current
    version permutes a single lagged correlation instead of the full
    shrinkage system -- a real but smaller-scope check)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf

from abm_model import StateResponseModel
from agents.state import (
    Regime, StateResponseFitter, BayesianVAR,
    load_and_prepare_data, prepare_var_data, build_model_inputs,
)
from agents.worker import calibrate_population_from_bvar

# Deterministic regime ordering for seed arithmetic below -- NOT Python's
# built-in hash(), which is randomized per-process (PYTHONHASHSEED) and
# would silently break run-to-run reproducibility of "the same seed."
_REGIME_INDEX = {r: i for i, r in enumerate(Regime.ALL)}


# ============================================================================
# SHARED FIT — every experiment below reuses this
# ============================================================================

def fit_once(csv_path="results/combined_long_panel.csv"):
    """
    Single shared fit for every experiment below. Returns everything a
    model instance needs (response_fn, response_params, residuals,
    worker_calibration) plus the fitter/bvar objects for diagnostics.
    """
    annual = load_and_prepare_data(csv_path)
    X, y, years = build_model_inputs(annual)
    fitter = StateResponseFitter(X, y, years)
    best_name, results = fitter.fit_and_compare(extrapolation_safe_only=True)
    best = results[best_name]
    if best["type"] != "parametric":
        raise ValueError(f"Best fit '{best_name}' is a GAM; wrap its predict() "
                          "before using it in the ABM, or force a parametric fit.")
    response_fn = getattr(fitter, f"{best_name}_model")
    response_params = best["params"]
    residuals = y - best["y_pred"]

    var_data = prepare_var_data(annual)
    bvar = BayesianVAR(var_data, lag=1).fit()
    worker_calibration = calibrate_population_from_bvar(bvar)

    initial_gini = float(annual["gini_coefficient"].dropna().iloc[0])

    return dict(
        annual=annual, fitter=fitter, best_name=best_name, bvar=bvar,
        response_fn=response_fn, response_params=response_params,
        residuals=residuals, worker_calibration=worker_calibration,
        initial_gini=initial_gini, domain_bounds=fitter.domain_bounds_,
    )


def build_model(fit, regime, mode="free", n_workers=400, n_firms=40, seed=None):
    historical_data = fit["annual"] if mode == "historical" else None
    return StateResponseModel(
        n_workers=n_workers, n_firms=n_firms, regime=regime,
        response_fn=fit["response_fn"], response_params=fit["response_params"],
        response_residuals=fit["residuals"], response_domain_bounds=fit["domain_bounds"],
        worker_calibration=fit["worker_calibration"],
        mode=mode, historical_data=historical_data,
        initial_gini=fit["initial_gini"], initial_avg_wage=20.0, seed=seed,
    )


# ============================================================================
# 1. HISTORICAL VALIDATION
# ============================================================================

def run_historical_validation(fit, n_workers=400, n_firms=40, seed=0):
    """
    Runs the model in historical mode (Gini/growth pulled from real
    annual data each tick) across the full available span, and compares
    simulated vs. actual redistribution_pct_gdp. Directly answers the
    README's "does the simulated policy path match reality" question.
    """
    annual = fit["annual"]
    model = build_model(fit, Regime.REPRESENTATIVE, mode="historical",
                         n_workers=n_workers, n_firms=n_firms, seed=seed)
    n_steps = len(annual)
    for _ in range(n_steps):
        model.step()

    sim = model.datacollector.get_model_vars_dataframe()["redistribution"].to_numpy()
    actual = annual["redistribution_pct_gdp"].to_numpy()[: len(sim)]

    mask = ~np.isnan(actual) & ~np.isnan(sim)
    rmse = float(np.sqrt(np.mean((sim[mask] - actual[mask]) ** 2)))
    corr = float(np.corrcoef(sim[mask], actual[mask])[0, 1]) if mask.sum() > 2 else np.nan

    print(f"Historical validation (n={int(mask.sum())} comparable years)")
    print(f"  RMSE simulated vs. actual redistribution_pct_gdp: {rmse:.3f}")
    print(f"  Correlation:                                      {corr:.3f}")

    out = pd.DataFrame({
        "year": annual.index[: len(sim)],
        "simulated_redistribution": sim,
        "actual_redistribution": actual,
    })
    return out, dict(rmse=rmse, correlation=corr)


# ============================================================================
# 2. REGIME COMPARISON — multiple seeds, not a single anecdotal run
# ============================================================================

def run_regime_comparison(fit, regimes=Regime.ALL, n_seeds=25, n_steps=200,
                           n_workers=400, n_firms=40, burn_in=50, base_seed=1000):
    """
    Runs N replicate seeds per regime in free mode and summarizes
    post-burn-in outcomes as mean +/- 95% CI (normal approximation,
    appropriate at n_seeds~25) across replicates -- a single run per
    regime is an anecdote, not a result, for a stochastic ABM.

    Returns (raw per-replicate df, summary df, full timeseries dict) --
    the timeseries dict feeds reform_vs_revolution_test() without
    needing to rerun anything.
    """
    records = []
    timeseries = {r: [] for r in regimes}

    for regime in regimes:
        for s in range(n_seeds):
            seed = base_seed + _REGIME_INDEX[regime] * 10_000 + s
            model = build_model(fit, regime, mode="free",
                                 n_workers=n_workers, n_firms=n_firms, seed=seed)
            for _ in range(n_steps):
                model.step()
            df = model.datacollector.get_model_vars_dataframe()
            timeseries[regime].append(df)

            post_burn = df.iloc[burn_in:]
            records.append(dict(
                regime=regime, seed=seed,
                mean_gini=post_burn["gini"].mean(),
                mean_unemployment=post_burn["unemployment_rate"].mean(),
                mean_protest_share=post_burn["protest_share"].mean(),
                mean_redistribution=post_burn["redistribution"].mean(),
                std_redistribution=post_burn["redistribution"].std(),
            ))

    raw = pd.DataFrame(records)

    metric_cols = ["mean_gini", "mean_unemployment", "mean_protest_share",
                    "mean_redistribution", "std_redistribution"]

    def _summ(g):
        out = {}
        for col in metric_cols:
            m = g[col].mean()
            se = g[col].std(ddof=1) / np.sqrt(len(g))
            out[f"{col}_mean"] = m
            out[f"{col}_ci95_lo"] = m - 1.96 * se
            out[f"{col}_ci95_hi"] = m + 1.96 * se
        return pd.Series(out)

    summary = raw.groupby("regime")[metric_cols + ["regime"]].apply(
        lambda g: _summ(g)
    )
    print_regime_summary(summary, n_seeds, n_steps, burn_in)
    return raw, summary, timeseries

def print_regime_summary(summary, n_seeds, n_steps, burn_in):
    """
    Human-legible reformat of the wide regime-comparison summary table --
    one block per metric, regimes as rows, mean + 95% CI in one line,
    instead of one unreadable wide row per regime.
    """
    metric_labels = {
        "mean_gini": "Gini",
        "mean_unemployment": "Unemployment rate",
        "mean_protest_share": "Protest share",
        "mean_redistribution": "Redistribution (mean)",
        "std_redistribution": "Redistribution (volatility)",
    }

    print(f"\nRegime comparison ({n_seeds} seeds/regime, {n_steps} steps, "
          f"burn-in={burn_in})")
    print("=" * 60)

    for metric, label in metric_labels.items():
        print(f"\n{label}")
        print("-" * 60)
        for regime in summary.index:
            m = summary.loc[regime, f"{metric}_mean"]
            lo = summary.loc[regime, f"{metric}_ci95_lo"]
            hi = summary.loc[regime, f"{metric}_ci95_hi"]
            print(f"  {regime:<16} {m:>8.4f}   [{lo:.4f}, {hi:.4f}]")
    print("\n" + "=" * 60)


# ============================================================================
# 3. REFORM VS. REVOLUTION TEST
# ============================================================================

def reform_vs_revolution_test(timeseries, burn_in=50, spike_percentile=90,
                               event_window=10, pre_baseline_window=5, acf_lags=10):
    """
    Two independent ways of asking "does redistribution stick after a
    protest spike, or revert" per regime:

      - Event study: for each protest spike (>= the replicate's own
        90th percentile, post-burn-in), track redistribution relative
        to its own pre-spike baseline for `event_window` ticks after.
        Averaged across all spikes across all replicates in a regime.
        Sticky -> stays elevated; cyclical -> reverts toward 0.
      - ACF of redistribution deviations, averaged across replicates --
        a second measure that doesn't depend on how spikes are defined,
        so it's a useful cross-check on the event study (slow decay =
        sticky, fast decay = cyclical).
    """
    event_results = {}
    acf_results = {}

    for regime, runs in timeseries.items():
        event_curves = []
        acfs = []
        for df in runs:
            post = df.iloc[burn_in:].reset_index(drop=True)
            if len(post) < acf_lags + 5:
                continue
            protest = post["protest_share"].to_numpy()
            redist = post["redistribution"].to_numpy()

            deviation = redist - redist.mean()
            try:
                acfs.append(acf(deviation, nlags=acf_lags, fft=True))
            except Exception:
                pass

            if protest.std() == 0:
                continue
            threshold = np.percentile(protest, spike_percentile)
            if threshold <= 0:
                continue
            spike_ticks = np.where(protest >= threshold)[0]
            for t in spike_ticks:
                if t < pre_baseline_window or t + event_window >= len(redist):
                    continue
                baseline = redist[t - pre_baseline_window: t].mean()
                traj = redist[t: t + event_window + 1] - baseline
                event_curves.append(traj)

        event_results[regime] = (
            np.mean(event_curves, axis=0) if event_curves else None,
            len(event_curves),
        )
        acf_results[regime] = np.mean(acfs, axis=0) if acfs else None

    print(f"\nReform-vs-revolution diagnostics (spike >= p{spike_percentile}, "
          f"{event_window}-tick window)")
    for regime in timeseries:
        curve, n_events = event_results[regime]
        print(f"\n{regime}  ({n_events} protest-spike events pooled)")
        if curve is not None:
            print(f"  redistribution relative to pre-spike baseline, t+0..t+{event_window}:")
            print("   " + "  ".join(f"{v:+.3f}" for v in curve))
        else:
            print("  (not enough spike events to compute a trajectory)")
        acf_vals = acf_results[regime]
        if acf_vals is not None:
            print(f"  redistribution ACF (lags 0-{acf_lags}): "
                  + "  ".join(f"{v:.2f}" for v in acf_vals))

    return event_results, acf_results


# ============================================================================
# 4. LIGHTWEIGHT SENSITIVITY SWEEP
# ============================================================================

def sensitivity_sweep(fit, param_grid=None, n_seeds=6, n_steps=120,
                       n_workers=250, n_firms=25, burn_in=30, base_seed=5000):
    """
    Fragility check on uncalibrated ABM parameters -- never fit to data,
    hand-set defaults: wage_noise_std and the worker threshold_std base.
    Sweeps each across a small grid and reports how much the
    representative-vs-dictatorship gap in mean redistribution moves.

    This is a first-pass check, not an exhaustive sensitivity analysis
    (that's flagged as future work) -- if the regime gap survives this
    small grid, that's reassuring; it isn't proof of robustness beyond
    this grid.
    """
    if param_grid is None:
        param_grid = {
            "wage_noise_std": [0.005, 0.01, 0.02],
            "threshold_std_base": [0.06, 0.08, 0.12],
        }

    combos = [(wn, ts) for wn in param_grid["wage_noise_std"]
              for ts in param_grid["threshold_std_base"]]

    results = []
    for combo_idx, (wn, ts) in enumerate(combos):
        gap_by_seed = []
        for s in range(n_seeds):
            seed = base_seed + combo_idx * 1000 + s
            outcomes = {}
            for regime in (Regime.REPRESENTATIVE, Regime.DICTATORSHIP):
                calib = dict(fit["worker_calibration"])
                calib["threshold_std"] = ts
                model = StateResponseModel(
                    n_workers=n_workers, n_firms=n_firms, regime=regime,
                    response_fn=fit["response_fn"], response_params=fit["response_params"],
                    response_residuals=fit["residuals"], response_domain_bounds=fit["domain_bounds"],
                    worker_calibration=calib,
                    mode="free", initial_gini=fit["initial_gini"],
                    initial_avg_wage=20.0, seed=seed,
                )
                # patched post-construction to sweep wage_noise_std without
                # threading a new Model-level constructor kwarg through for
                # what is, for now, a first-pass fragility check
                for f in model.firms:
                    f.wage_noise_std = wn
                for _ in range(n_steps):
                    model.step()
                df = model.datacollector.get_model_vars_dataframe().iloc[burn_in:]
                outcomes[regime] = df["redistribution"].mean()
            gap_by_seed.append(outcomes[Regime.REPRESENTATIVE] - outcomes[Regime.DICTATORSHIP])
        results.append(dict(wage_noise_std=wn, threshold_std_base=ts,
                             mean_gap=np.mean(gap_by_seed), std_gap=np.std(gap_by_seed)))

    out = pd.DataFrame(results)
    print("\nSensitivity sweep -- representative minus dictatorship mean redistribution gap")
    print(out.round(4).to_string(index=False))
    return out


# ============================================================================
# 5. PERMUTATION NULL CHECK
# ============================================================================

def permutation_null_check(fit, n_permutations=2000, seed=0):
    """
    Empirical permutation test on the real historical
    Protest(t-1) -> Redistribution(t) relationship (first-differenced,
    per the BVAR's own stationarity fix), as a direct answer to "is our
    'no effect' claim actually defensible, or just underpowered."

    Lighter-weight than a full per-permutation refit of the shrinkage
    BVAR (flagged as a follow-up): this permutes the simple lagged-
    correlation coefficient between d_protest and d_redistribution
    instead. If the observed coefficient falls well inside the shuffled
    null distribution, that's real evidence the earlier null finding
    isn't just an artifact of one particular model choice.
    """
    rng = np.random.default_rng(seed)
    var_data = prepare_var_data(fit["annual"])
    protest = var_data["d_protest_intensity_score"].to_numpy()
    redist = var_data["d_redistribution_pct_gdp"].to_numpy()

    protest_lag = protest[:-1]
    redist_t = redist[1:]
    observed = float(np.corrcoef(protest_lag, redist_t)[0, 1])

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(protest_lag)
        null_dist[i] = np.corrcoef(shuffled, redist_t)[0, 1]

    p_value = float(np.mean(np.abs(null_dist) >= abs(observed)))

    print(f"\nPermutation null check (n={len(redist_t)} obs, {n_permutations} shuffles)")
    print(f"  Observed corr(d_protest[t-1], d_redistribution[t]): {observed:.4f}")
    print(f"  Null distribution: mean={null_dist.mean():.4f}, std={null_dist.std():.4f}")
    print(f"  Two-sided empirical p-value: {p_value:.4f}")
    if p_value > 0.10:
        print("  -> Observed correlation is unremarkable relative to pure chance at "
              "this sample size: consistent with the BVAR's null finding, not an "
              "artifact of that specific model choice.")
    else:
        print("  -> Observed correlation falls outside most of the null distribution: "
              "worth reconciling with the BVAR's null Granger-style result.")

    return dict(observed=observed, null_dist=null_dist, p_value=p_value)

# ============================================================================
# 6. GINI GROWTH VS. REDISTRIBUTION TEST
# ============================================================================

def run_gini_growth_test(fit, n_seeds=15, n_steps=500, burn_in=100,
                          n_workers=400, n_firms=40, base_seed=9000):
    """
    Isolates whether the fiscal-response channel matters for Gini's
    long-run trajectory at all, vs. only dampens it, by running the
    SAME representative-regime setup with redistribution fully on vs.
    fully off (State.redistribution_enabled) and comparing each
    condition's post-burn-in Gini TREND (OLS slope of gini vs. step),
    not just its mean level.

    n_steps is much longer than the 200-step regime comparison --
    compounding-driven Gini drift needs runway to separate from
    per-tick noise; 200 steps is fine for level comparisons, not for
    slope estimation.

    Three possible outcomes, all reportable as a real finding:
      - off has a steeper positive Gini slope than on: redistribution
        measurably dampens (but per wealth_growth_check.py, does not
        reverse) rising inequality -- consistent with "insufficient."
      - on and off have statistically indistinguishable slopes:
        redistribution as currently calibrated has no detectable effect
        on the inequality trend at all in this model -- a stronger,
        different claim than "insufficient."
      - off has a FLATTER slope than on: would contradict the intended
        mechanism and needs debugging before any claim is made.
    """
    def _condition_slopes(redistribution_enabled):
        slopes = []
        for s in range(n_seeds):
            seed = base_seed + (0 if redistribution_enabled else 1) * 10_000 + s
            model = StateResponseModel(
                n_workers=n_workers, n_firms=n_firms, regime=Regime.REPRESENTATIVE,
                response_fn=fit["response_fn"], response_params=fit["response_params"],
                response_residuals=fit["residuals"], response_domain_bounds=fit["domain_bounds"],
                worker_calibration=fit["worker_calibration"],
                mode="free", initial_gini=fit["initial_gini"],
                initial_avg_wage=20.0, seed=seed,
                redistribution_enabled=redistribution_enabled,
            )
            for _ in range(n_steps):
                model.step()
            df = model.datacollector.get_model_vars_dataframe()
            post = df.iloc[burn_in:]
            steps = np.arange(len(post))
            slope = linregress(steps, post["gini"].to_numpy()).slope
            slopes.append(slope)
        return np.array(slopes)

    on_slopes = _condition_slopes(True)
    off_slopes = _condition_slopes(False)

    def _mean_ci(x):
        m = x.mean()
        se = x.std(ddof=1) / np.sqrt(len(x))
        return m, m - 1.96 * se, m + 1.96 * se

    on_m, on_lo, on_hi = _mean_ci(on_slopes)
    off_m, off_lo, off_hi = _mean_ci(off_slopes)
    diff = off_slopes - on_slopes
    diff_m, diff_lo, diff_hi = _mean_ci(diff)

    print(f"\nGini trend (post-burn-in OLS slope, {n_seeds} seeds/condition, "
          f"{n_steps} steps, burn-in={burn_in})")
    print(f"  redistribution ON:  slope = {on_m:+.6f}/step   [{on_lo:+.6f}, {on_hi:+.6f}]")
    print(f"  redistribution OFF: slope = {off_m:+.6f}/step   [{off_lo:+.6f}, {off_hi:+.6f}]")
    print(f"  OFF minus ON:       {diff_m:+.6f}   [{diff_lo:+.6f}, {diff_hi:+.6f}]")
    if diff_lo > 0:
        print("  -> OFF's slope is credibly steeper than ON's: redistribution measurably "
              "dampens the rise in Gini in this model (consistent with 'insufficient, "
              "not absent' if wealth_growth_check.py also showed a real Gini uptrend "
              "even with redistribution on).")
    elif diff_hi < 0:
        print("  -> OFF's slope is credibly FLATTER than ON's -- redistribution appears "
              "to be accelerating inequality, opposite of the intended mechanism. This "
              "needs debugging before any claim is made from it.")
    else:
        print("  -> ON and OFF slopes are not credibly different: as calibrated, the "
              "fiscal-response channel has no detectable effect on the Gini trend in "
              "this model, not merely a dampened one.")

    out = pd.DataFrame({
        "condition": ["on"] * n_seeds + ["off"] * n_seeds,
        "seed_index": list(range(n_seeds)) * 2,
        "gini_slope": np.concatenate([on_slopes, off_slopes]),
    })
    return out, dict(on_mean=on_m, on_ci=(on_lo, on_hi),
                      off_mean=off_m, off_ci=(off_lo, off_hi),
                      diff_mean=diff_m, diff_ci=(diff_lo, diff_hi))

# ============================================================================
# 6b. DOSE-RESPONSE CHECK — is the null a scale artifact?
# ============================================================================

def run_redistribution_dose_response(fit, multipliers=(1, 3, 10, 30),
                                      n_seeds=10, n_steps=500, burn_in=100,
                                      n_workers=400, n_firms=40, base_seed=9500):
    """
    Scales State.redistribution post-decision by each multiplier and
    re-checks the Gini slope, to distinguish "redistribution doesn't
    reach the inequality channel at all" (slope stays flat even at 30x)
    from "current calibration is just too weak" (slope responds to scale).
    """
    from scipy.stats import linregress

    results = []
    for mult in multipliers:
        slopes = []
        for s in range(n_seeds):
            seed = base_seed + int(mult * 100) + s
            model = StateResponseModel(
                n_workers=n_workers, n_firms=n_firms, regime=Regime.REPRESENTATIVE,
                response_fn=fit["response_fn"], response_params=fit["response_params"],
                response_residuals=fit["residuals"], response_domain_bounds=fit["domain_bounds"],
                worker_calibration=fit["worker_calibration"],
                mode="free", initial_gini=fit["initial_gini"],
                initial_avg_wage=20.0, seed=seed, redistribution_enabled=True,
            )
            orig_redistribute = model.state.redistribute
            def scaled_redistribute(workers, _orig=orig_redistribute, _m=mult):
                model.state.redistribution *= _m
                _orig(workers)
            model.state.redistribute = scaled_redistribute

            for _ in range(n_steps):
                model.step()
            df = model.datacollector.get_model_vars_dataframe().iloc[burn_in:]
            slope = linregress(np.arange(len(df)), df["gini"].to_numpy()).slope
            slopes.append(slope)
        slopes = np.array(slopes)
        m, se = slopes.mean(), slopes.std(ddof=1) / np.sqrt(len(slopes))
        results.append(dict(multiplier=mult, mean_slope=m,
                             ci95_lo=m - 1.96*se, ci95_hi=m + 1.96*se))
        print(f"  {mult:>4}x redistribution: Gini slope = {m:+.6f}  "
              f"[{m-1.96*se:+.6f}, {m+1.96*se:+.6f}]")

    out = pd.DataFrame(results)
    print("\nDose-response result:")
    print(out.round(6).to_string(index=False))
    if out["mean_slope"].max() - out["mean_slope"].min() < 2 * out["ci95_hi"].sub(out["ci95_lo"]).mean():
        print("-> Slope insensitive to scale even at 30x: redistribution mechanism "
              "appears structurally disconnected from the Gini/inequality channel, "
              "not merely underpowered.")
    else:
        print("-> Slope responds to scale: current real-data calibration is simply "
              "too weak, consistent with 'insufficient, not absent.'")
    return out

# ============================================================================
# 7. TARGETING DIAGNOSTIC — is the ON-condition Gini rise concentrated
#    in the fixed recipient group, and does per-recipient subsidy grow?
# ============================================================================

def run_targeting_diagnostic(fit, n_seeds=10, n_steps=500, burn_in=100,
                              n_workers=400, n_firms=40, base_seed=9800):
    """
    Runs representative-regime, redistribution ON, and checks two things
    against the post-burn-in window:
      1. Does mean_transfer_per_recipient trend upward over time (the
         "momentum via redist_lag" mechanism)?
      2. Does gini_recipients rise faster than gini_non_recipients (the
         "same fixed group getting an ever-larger, more concentrated
         subsidy" mechanism)?
    Both trending the same direction as the overall ON Gini slope from
    run_gini_growth_test would confirm the targeting-concentration
    explanation rather than a general "redistribution backfires" claim.
    """
    subsidy_slopes = []
    gini_recip_slopes = []
    gini_nonrecip_slopes = []
    n_recipients_means = []

    for s in range(n_seeds):
        seed = base_seed + s
        model = StateResponseModel(
            n_workers=n_workers, n_firms=n_firms, regime=Regime.REPRESENTATIVE,
            response_fn=fit["response_fn"], response_params=fit["response_params"],
            response_residuals=fit["residuals"], response_domain_bounds=fit["domain_bounds"],
            worker_calibration=fit["worker_calibration"],
            mode="free", initial_gini=fit["initial_gini"],
            initial_avg_wage=20.0, seed=seed, redistribution_enabled=True,
        )
        for _ in range(n_steps):
            model.step()
        df = model.datacollector.get_model_vars_dataframe().iloc[burn_in:].reset_index(drop=True)
        steps = np.arange(len(df))

        subsidy_slopes.append(linregress(steps, df["mean_transfer_per_recipient"]).slope)
        gini_recip_slopes.append(linregress(steps, df["gini_recipients"]).slope)
        gini_nonrecip_slopes.append(linregress(steps, df["gini_non_recipients"]).slope)
        n_recipients_means.append(df["n_recipients"].mean())

    def _mean_ci(x):
        x = np.array(x)
        m = x.mean()
        se = x.std(ddof=1) / np.sqrt(len(x))
        return m, m - 1.96 * se, m + 1.96 * se

    subs_m, subs_lo, subs_hi = _mean_ci(subsidy_slopes)
    gr_m, gr_lo, gr_hi = _mean_ci(gini_recip_slopes)
    gnr_m, gnr_lo, gnr_hi = _mean_ci(gini_non_recipient_slopes := gini_nonrecip_slopes)
    diff = np.array(gini_recip_slopes) - np.array(gini_nonrecip_slopes)
    diff_m, diff_lo, diff_hi = _mean_ci(diff)

    print(f"\nTargeting diagnostic ({n_seeds} seeds, {n_steps} steps, burn-in={burn_in})")
    print(f"  Mean recipients/tick:              {np.mean(n_recipients_means):.1f} "
          f"(fixed bottom-quartile-by-wage group, n_workers={n_workers})")
    print(f"  Subsidy-per-recipient slope:       {subs_m:+.6f}/step   [{subs_lo:+.6f}, {subs_hi:+.6f}]")
    print(f"  Gini (recipients only) slope:      {gr_m:+.6f}/step   [{gr_lo:+.6f}, {gr_hi:+.6f}]")
    print(f"  Gini (non-recipients only) slope:  {gnr_m:+.6f}/step   [{gnr_lo:+.6f}, {gnr_hi:+.6f}]")
    print(f"  Recipients minus non-recipients:   {diff_m:+.6f}   [{diff_lo:+.6f}, {diff_hi:+.6f}]")

    subsidy_grows = subs_lo > 0
    concentration_confirmed = diff_lo > 0
    if subsidy_grows and concentration_confirmed:
        print("  -> CONFIRMED: per-recipient subsidy grows over time AND inequality "
              "within the recipient group rises faster than among non-recipients. "
              "This supports a targeting-design explanation (fixed-group, "
              "growing, concentrated transfer) for the earlier ON > OFF Gini-slope "
              "result -- not a general 'redistribution backfires' finding.")
    elif concentration_confirmed:
        print("  -> Gini rises faster within recipients than non-recipients, but "
              "subsidy-per-recipient itself is not credibly growing -- some other "
              "within-group divergence (e.g. wage/investment variance among "
              "recipients) is driving it, not simply a growing lump sum. Needs "
              "further decomposition before claiming the targeting mechanism.")
    else:
        print("  -> Neither the subsidy-growth nor the within-group-concentration "
              "pattern is credibly confirmed. The earlier ON > OFF Gini-slope "
              "result is NOT yet explained by this mechanism -- treat it as an "
              "open finding, not attributed to targeting design.")

    out = pd.DataFrame({
        "seed_index": range(n_seeds),
        "subsidy_per_recipient_slope": subsidy_slopes,
        "gini_recipients_slope": gini_recip_slopes,
        "gini_non_recipients_slope": gini_nonrecip_slopes,
        "mean_n_recipients": n_recipients_means,
    })
    return out, dict(subsidy_slope=(subs_m, subs_lo, subs_hi),
                      gini_recipients_slope=(gr_m, gr_lo, gr_hi),
                      gini_non_recipients_slope=(gnr_m, gnr_lo, gnr_hi),
                      diff=(diff_m, diff_lo, diff_hi))

# ============================================================================
# 8. MECHANISM TRACE — variance decomposition + repression-cap check
# ============================================================================

def run_mechanism_trace(fit, regimes=Regime.ALL, n_seeds=10, n_steps=300,
                         burn_in=60, n_workers=400, n_firms=40, base_seed=9900):
    """
    Two mechanism checks, run across all three regimes:
      1. Income-variance decomposition (wage / transfer / owner-capital
         share of total variance) -- tests whether Gini is structurally
         dominated by capital-income variance regardless of transfer
         size (the r>g-style explanation for the redistribution<->Gini
         null).
      2. repression_bound_share -- fraction of workers each tick whose
         protest probability would stay suppressed EVEN AT MAXIMUM
         grievance, given their current risk_perception/threshold. High
         and regime-dependent (captured/dictatorship >> representative)
         would support a repression-caps-the-signal explanation for weak
         protest->redistribution linkage, distinct from the weak
         macro-sensitivity calibration itself.
    """
    records = []
    for regime in regimes:
        for s in range(n_seeds):
            seed = base_seed + _REGIME_INDEX[regime] * 10_000 + s
            model = build_model(fit, regime, mode="free",
                                 n_workers=n_workers, n_firms=n_firms, seed=seed)
            for _ in range(n_steps):
                model.step()
            df = model.datacollector.get_model_vars_dataframe().iloc[burn_in:]
            records.append(dict(
                regime=regime, seed=seed,
                mean_wage_var_share=df["wage_var_share"].mean(),
                mean_transfer_var_share=df["transfer_var_share"].mean(),
                mean_owner_var_share=df["owner_var_share"].mean(),
                mean_repression_bound_share=df["repression_bound_share"].mean(),
            ))

    out = pd.DataFrame(records)
    summary = out.groupby("regime")[[
        "mean_wage_var_share", "mean_transfer_var_share",
        "mean_owner_var_share", "mean_repression_bound_share"
    ]].agg(["mean", "std"])

    print(f"\nMechanism trace ({n_seeds} seeds/regime, {n_steps} steps, burn-in={burn_in})")
    print("=" * 70)
    print("\nIncome variance share by source (of total income variance driving Gini):")
    print(summary[["mean_wage_var_share", "mean_transfer_var_share", "mean_owner_var_share"]]
          .round(4).to_string())
    print("\nRepression-bound share (fraction of workers capped regardless of grievance):")
    print(summary[["mean_repression_bound_share"]].round(4).to_string())

    owner_dominant = (out.groupby("regime")["mean_owner_var_share"].mean() >
                       out.groupby("regime")["mean_transfer_var_share"].mean() * 5).all()
    if owner_dominant:
        print("\n-> Owner/capital-income variance dominates transfer-income variance by "
              ">5x in every regime: consistent with a structural (r>g-style) "
              "explanation for the redistribution<->Gini null -- a bottom-targeted, "
              "capped transfer pool cannot offset unconstrained capital-income "
              "variance at the top, regardless of transfer size. Present as a "
              "modeled hypothesis for real-world investigation, not an empirical "
              "finding on its own.")
    else:
        print("\n-> Owner/capital-income variance does not clearly dominate transfer "
              "variance -- the r>g-style explanation is not well-supported by this "
              "decomposition; the redistribution<->Gini null likely has some other "
              "or additional cause.")

    return out, summary


# ============================================================================
# __main__
# ============================================================================

if __name__ == "__main__":
    os.makedirs("results/experiments", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT SUITE")
    print("=" * 70)

    fit = fit_once()

    print("\n" + "-" * 70)
    print("1. HISTORICAL VALIDATION")
    print("-" * 70)
    hist_df, hist_stats = run_historical_validation(fit)
    hist_df.to_csv("results/experiments/historical_validation.csv", index=False)

    print("\n" + "-" * 70)
    print("2. REGIME COMPARISON")
    print("-" * 70)
    raw, summary, timeseries = run_regime_comparison(fit, n_seeds=25, n_steps=200)
    raw.to_csv("results/experiments/regime_comparison_raw.csv", index=False)
    summary.to_csv("results/experiments/regime_comparison_summary.csv")

    print("\n" + "-" * 70)
    print("3. REFORM VS. REVOLUTION TEST")
    print("-" * 70)
    event_results, acf_results = reform_vs_revolution_test(timeseries)

    print("\n" + "-" * 70)
    print("4. SENSITIVITY SWEEP (lightweight)")
    print("-" * 70)
    sens = sensitivity_sweep(fit)
    sens.to_csv("results/experiments/sensitivity_sweep.csv", index=False)

    print("\n" + "-" * 70)
    print("5. PERMUTATION NULL CHECK")
    print("-" * 70)
    perm = permutation_null_check(fit)

    print("\n" + "-" * 70)
    print("6. GINI GROWTH VS. REDISTRIBUTION TEST")
    print("-" * 70)
    gini_growth_df, gini_growth_stats = run_gini_growth_test(fit)
    gini_growth_df.to_csv("results/experiments/gini_growth_test.csv", index=False)

    print("\n" + "-" * 70)
    print("6b. REDISTRIBUTION DOSE-RESPONSE CHECK")
    print("-" * 70)
    dose_df = run_redistribution_dose_response(fit)
    dose_df.to_csv("results/experiments/redistribution_dose_response.csv", index=False)

    print("\n" + "-" * 70)
    print("7. TARGETING DIAGNOSTIC")
    print("-" * 70)
    targeting_df, targeting_stats = run_targeting_diagnostic(fit)
    targeting_df.to_csv("results/experiments/targeting_diagnostic.csv", index=False)

    print("\n" + "-" * 70)
    print("8. MECHANISM TRACE")
    print("-" * 70)
    mechanism_df, mechanism_summary = run_mechanism_trace(fit)
    mechanism_df.to_csv("results/experiments/mechanism_trace.csv", index=False)

    print("\n" + "=" * 70)
    print("NOT YET IMPLEMENTED (flagged for follow-up, not silently skipped):")
    print("  - Period-by-period / rolling-window refit of empirical response fn + BVAR")
    print("  - Compositional/indirect correlation analysis (spending mix vs protest)")
    print("  - Full per-permutation BVAR refit for the null check (current version")
    print("    permutes a single lagged correlation, not the full shrinkage system)")
    print("=" * 70)