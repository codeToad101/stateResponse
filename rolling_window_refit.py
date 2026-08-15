"""
rolling_window_refit.py

============================================================================
 USES: REAL HISTORICAL DATA ONLY, RE-FITTING THE RESPONSE FUNCTION.
 No ABM simulation runs here. No event-based comparison here.
============================================================================

Tests whether the fitted protest -> redistribution relationship is
STABLE over time, by re-fitting StateResponseFitter on overlapping
rolling sub-windows of the full 1963-2025 span, instead of the single
fixed fit experiments.py's fit_once() uses across the whole period.

This answers a DIFFERENT question than historical_event_study.py:
  - historical_event_study.py asks: "did real redistribution shift
    around real, discrete strike-wave YEARS?"
  - rolling_window_refit.py asks: "does the fitted protest COEFFICIENT
    itself change across different ERAS (e.g. pre/post-1980), or is
    one fixed relationship a reasonable assumption for the full span?"

This directly answers Open Question #4 already flagged in the README
("Period-by-period breakdown ... rolling-window or regime-split fits").

CAVEAT, stated here rather than hidden: at n=63 total annual points,
a rolling window of ~20 years leaves ~15-18 usable rows per window
after dropping NaNs -- enough to fit a low-parameter (linear/logistic/
exponential) form, but coefficient estimates on any single window will
carry real sampling uncertainty. Treat each window's point estimate as
suggestive, not a confident per-era claim, especially for the GAM
candidate which is not attempted here for that reason (too flexible
for a ~15-18 point window).
"""

import numpy as np
import pandas as pd

from agents.state import StateResponseFitter, load_and_prepare_data, build_model_inputs


def rolling_refit(csv_path="results/us_state_response_data.csv",
                   window_years=20, step_years=5, min_obs=12):
    """
    Slides a window_years-wide window across the annual data in
    step_years increments, refitting StateResponseFitter on each
    window independently (each window gets its own AIC-best functional
    form -- not forced to match the full-sample winner, since the best
    form itself might differ by era).
    """
    annual = load_and_prepare_data(csv_path)
    years = annual.index

    start = years.min()
    end = years.max()

    records = []
    window_start = start
    while window_start + window_years <= end:
        window_end = window_start + window_years
        window_annual = annual[(years >= window_start) & (years < window_end)]

        if len(window_annual) < min_obs:
            window_start += step_years
            continue

        X, y, wyears = build_model_inputs(window_annual)
        try:
            fitter = StateResponseFitter(X, y, wyears)
            best_name, results = fitter.fit_and_compare()
            best = results[best_name]
        except Exception as e:
            print(f"  [{window_start}-{window_end}] fit failed: {str(e)[:60]}")
            window_start += step_years
            continue

        if best["type"] != "parametric":
            # GAM winning a ~15-18 point window is a sign of overfitting
            # a window this small, not a real finding -- skip rather than
            # report a coefficient that doesn't exist for a GAM anyway
            print(f"  [{window_start}-{window_end}] best fit was GAM "
                  f"(n={len(window_annual)}) -- skipping, too flexible "
                  f"for this window size")
            window_start += step_years
            continue

        # NOT a fixed param index -- state.py's three parametric forms put
        # the protest term in different places (linear: params[0] directly;
        # exponential: params[2] directly; logistic: protest is INSIDE the
        # sigmoid combined with gini, no standalone coefficient at all).
        # Numerical marginal effect at the window's mean predictor values
        # is the only form-agnostic way to compare "protest's effect on
        # CHANGE in redistribution" (state.py's fitter is now fit on
        # first-differenced series -- see state.py's ADF-driven fix)
        # across windows that may pick different winning forms.
        model_fn = getattr(fitter, f"{best_name}_model")
        x_mean = X.mean(axis=1, keepdims=True)  # (4, 1): d_protest, d_gini, growth, d_redist_lag
        h = 1e-4
        x_plus = x_mean.copy(); x_plus[0, 0] += h
        x_minus = x_mean.copy(); x_minus[0, 0] -= h
        protest_coef = float(
            (model_fn(x_plus, *best["params"])[0] - model_fn(x_minus, *best["params"])[0])
            / (2 * h)
        )

        records.append(dict(
            window=f"{window_start}-{window_end}",
            n_obs=len(window_annual),
            best_form=best_name,
            r_squared=best.get("r2", np.nan),
            protest_coef=protest_coef,
        ))

        window_start += step_years

    return pd.DataFrame(records)


def summarize(out):
    if out.empty:
        print("\nNo windows produced a usable parametric fit.")
        return
    print("\nRolling-window refit results:")
    print(out.round(4).to_string(index=False))

    coefs = out["protest_coef"].dropna()
    if len(coefs) >= 2:
        print(f"\nProtest marginal effect on CHANGE in redistribution (numerical, "
              f"at each window's mean d_protest/d_gini/growth/d_redist_lag) "
              f"across windows: "
              f"mean={coefs.mean():+.4f}, std={coefs.std():.4f}, "
              f"range=[{coefs.min():+.4f}, {coefs.max():+.4f}]")
        if coefs.std() > abs(coefs.mean()):
            print("  -> High variability relative to the mean: consistent "
                  "with an unstable/era-dependent relationship, or with "
                  "a coefficient that is genuinely near zero everywhere "
                  "(both are compatible with the full-sample null finding).")


if __name__ == "__main__":
    print("=" * 70)
    print("ROLLING-WINDOW REFIT -- re-fitting the response function by era")
    print("=" * 70)

    out = rolling_refit()
    summarize(out)
    out.to_csv("results/experiments/rolling_window_refit.csv", index=False)

    print("\n" + "=" * 70)
    print("Reminder: this re-fits the EMPIRICAL RESPONSE FUNCTION on real")
    print("data across sub-periods. It does not look at discrete events")
    print("(see historical_event_study.py) and does not run the ABM.")
    print("=" * 70)