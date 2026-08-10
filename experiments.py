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

def fit_once(csv_path="results/us_state_response_data.csv"):
    """
    Single shared fit for every experiment below. Returns everything a
    model instance needs (response_fn, response_params, residuals,
    worker_calibration) plus the fitter/bvar objects for diagnostics.
    """
    annual = load_and_prepare_data(csv_path)
    X, y, years = build_model_inputs(annual)
    fitter = StateResponseFitter(X, y, years)
    best_name, results = fitter.fit_and_compare()
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
    print(f"\nRegime comparison ({n_seeds} seeds/regime, {n_steps} steps, "
          f"burn-in={burn_in})")
    print(summary.round(4).to_string())
    return raw, summary, timeseries


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

    print("\n" + "=" * 70)
    print("NOT YET IMPLEMENTED (flagged for follow-up, not silently skipped):")
    print("  - Period-by-period / rolling-window refit of empirical response fn + BVAR")
    print("  - Compositional/indirect correlation analysis (spending mix vs protest)")
    print("  - Full per-permutation BVAR refit for the null check (current version")
    print("    permutes a single lagged correlation, not the full shrinkage system)")
    print("=" * 70)