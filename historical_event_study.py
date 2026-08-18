"""
historical_event_study.py

============================================================================
 USES: REAL HISTORICAL DATA ONLY. No ABM simulation runs here.
============================================================================

Tests the disruption-theory question directly: around REAL historical
strike-wave years (identified from actual workers_affected /
civil_unrest_events / protest_participation_rate in the CSV), did actual
redistribution_pct_gdp shift meaningfully in the surrounding years?

This is deliberately separate from:
  - experiments.py's reform_vs_revolution_test(), which runs on SIMULATED
    ABM output (protest_share, redistribution from the model), not real
    history.
  - rolling_window_refit.py, which re-fits the empirical response
    function over sliding time windows rather than looking at discrete
    events.

If this event study and the ABM's reform_vs_revolution_test disagree,
that's not a contradiction to paper over -- it would mean the ABM's
protest/response dynamics don't reproduce the pattern actually observed
in history, which is itself a finding worth reporting.
"""

import numpy as np
import pandas as pd
from agents.state import load_and_prepare_panel


# def load_annual_history(csv_path="results/combined_long_panel.csv"):
#     """
#     Aggregates the quarterly CSV to one row per year, independent of
#     the ABM-fitting pipeline's load_and_prepare_data() -- kept
#     deliberately separate so this event study never silently depends
#     on ABM-fitting internals changing underneath it.

#     redistribution_pct_gdp is NOT a raw CSV column -- mirrors state.py's
#     derivation (sum of redist_eitc/snap/medicaid/ui, divided by
#     gdp_billions) rather than assuming it exists pre-computed.
#     """
#     df = pd.read_csv(csv_path)
#     df["date"] = pd.to_datetime(df.iloc[:, 0])
#     df["year"] = df["date"].dt.year

#     agg_map = {
#         "workers_affected": "sum",
#         "civil_unrest_events": "sum",
#         "protest_participation_rate": "mean",
#         "gini_coefficient": "mean",
#         "gdp_billions": "last",
#     }
#     redist_components = [c for c in
#                           ["redist_eitc", "redist_snap", "redist_medicaid", "redist_ui"]
#                           if c in df.columns]
#     agg_map.update({c: "first" for c in redist_components})

#     agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
#     annual = df.groupby("year").agg(agg_map)

#     if redist_components and "gdp_billions" in annual.columns:
#         annual["redistribution_total_billions"] = annual[redist_components].sum(axis=1)
#         annual["redistribution_pct_gdp"] = (
#             annual["redistribution_total_billions"] / annual["gdp_billions"] * 100
#         )
#     else:
#         annual["redistribution_pct_gdp"] = np.nan

#     return annual.dropna(subset=["redistribution_pct_gdp"])


def identify_strike_wave_years(annual, method="civil_unrest_events",
                                top_pct=0.85):
    """
    Flags REAL historical years as strike-wave years if they fall above
    the top_pct percentile on the chosen severity measure. Default
    measure is protest_participation_rate (the real, unscaled rate --
    see calculate_participation_rate in USdataCollect.py), not the
    min-max-rescaled protest_intensity_score, for the same reason noted
    throughout this project: rescaled indices don't have a meaningful
    absolute threshold.
    """
    col = {
        "participation_rate": "protest_participation_rate",
        "workers_affected": "workers_affected",
        "civil_unrest_events": "civil_unrest_events",
    }[method]

    series = annual[col].dropna()
    threshold = series.quantile(top_pct)
    wave_years = series[series >= threshold].index.tolist()
    return wave_years, threshold

# ============================================================================
# CURATED HISTORICAL LANDMARKS -- real regime-change/uprising dates, NOT
# derived from the data (same "manually reviewed" posture as
# regime_periods.csv). Distinct from identify_strike_wave_years' STATISTICAL
# (top-15%-of-years) approach above -- this is "does fiscal response follow
# a KNOWN real event," not "does it follow a data-defined outlier year."
#
# Checked actual per-country coverage (redistribution_pct_gdp,
# civil_unrest_events, political_violence_score, gini_coefficient, +/-3/+5
# years) before finalizing this list. South Africa was the ORIGINAL and
# only candidate under consideration -- broadened after that check showed
# South Africa has the WORST redistribution_pct_gdp coverage of any
# candidate around its own landmark years (0/9 years covered around 1976,
# 1985, AND 1990; only 4/9 around 1994 itself). Germany (1990) has full
# coverage across all four series; Egypt (2011/2013) nearly so. South
# Africa stays in the list for completeness but expect it to be mostly
# skipped/flagged by landmark_data_coverage below, not a strong test case.
# ============================================================================
LANDMARK_YEARS = {
    'Germany': [1990],                          # reunification
    'Egypt': [2011, 2013],                       # revolution / Sisi takeover
    'Poland': [1989],                            # Round Table transition
    'Russia': [1991, 1992],                      # USSR collapse
    'South Africa': [1976, 1985, 1990, 1994],    # Soweto, states of emergency, transition
}

# Minimum fraction of the [-3, +5]-year window that redistribution_pct_gdp
# needs real (non-null) coverage for, before an event_study around that
# landmark is worth running at all.
LANDMARK_MIN_COVERAGE_FRAC = 0.65


def landmark_data_coverage(annual, landmark_year, window_pre=3, window_post=5,
                            required_cols=('redistribution_pct_gdp', 'civil_unrest_events',
                                            'political_violence_score', 'gini_coefficient')):
    """
    Reports, for one country's one landmark year, how many years in the
    [landmark-window_pre, landmark+window_post] span have real data for
    each required column -- decides whether a landmark event is actually
    testable rather than assuming it is.
    """
    window = list(range(landmark_year - window_pre, landmark_year + window_post + 1))
    coverage = {}
    for col in required_cols:
        if col not in annual.columns:
            coverage[col] = 0
            continue
        avail = set(annual[col].dropna().index)
        coverage[col] = sum(1 for y in window if y in avail)
    coverage['window_len'] = len(window)
    return coverage


def event_study(annual, wave_years, pre_window=2, post_window=5,
                 exclude_confound_years=(2020, 2021)):
    """
    For each REAL strike-wave year, compares redistribution_pct_gdp in
    the post_window years after to the pre_window years immediately
    before -- sticky (reform) vs. reverting (cyclical) shows up as a
    persistent vs. fading gap.

    exclude_confound_years: wave events whose pre/post window overlaps
    any of these years are dropped entirely (not just truncated), since
    an external shock landing inside the window (e.g. COVID-era relief
    spending in 2020-2021) can dominate the mean trajectory on its own
    with only a handful of pooled events -- default excludes 2020-2021
    for that reason. Set to () to disable and see the raw, unfiltered
    result including any such confound.
    """
    records = []
    years_index = annual.index
    excluded = []

    for wy in wave_years:
        pre_years = [y for y in range(wy - pre_window, wy) if y in years_index]
        post_years = [y for y in range(wy, wy + post_window + 1) if y in years_index]

        window_years = set(pre_years) | set(post_years)
        if window_years & set(exclude_confound_years):
            excluded.append(wy)
            continue

        if len(pre_years) < 1 or len(post_years) < 2:
            continue  # not enough real data around this event to compare

        baseline = annual.loc[pre_years, "redistribution_pct_gdp"].mean()
        trajectory = annual.loc[post_years, "redistribution_pct_gdp"] - baseline

        records.append(dict(
            wave_year=wy,
            baseline_redistribution=baseline,
            **{f"t+{i}": v for i, v in enumerate(trajectory.values)}
        ))

    if excluded:
        print(f"  Excluded {len(excluded)} wave event(s) whose window overlaps "
              f"{exclude_confound_years} (e.g. COVID relief spending): {excluded}")

    return pd.DataFrame(records)

def _trajectory_auc(annual, years_index, event_years, pre_window, post_window,
                     exclude_confound_years, min_horizon):
    """
    Shared helper: given a set of event years (real wave years, or a random
    draw for the null), computes each event's deviation trajectory the same
    way event_study() does, restricts to the first `min_horizon` post-window
    points (so every event contributes an equal-length, fully-observed
    trajectory -- no partial/NaN tail entries from events near the end of
    the series), and returns the mean trajectory's AUC (trapezoidal, over
    t+0..t+min_horizon-1).

    Returns None if fewer than 2 events survive filtering -- not enough to
    form a meaningful mean trajectory.
    """
    trajectories = []
    for wy in event_years:
        pre_years = [y for y in range(wy - pre_window, wy) if y in years_index]
        post_years = [y for y in range(wy, wy + min_horizon) if y in years_index]

        window_years = set(pre_years) | set(range(wy - pre_window, wy + post_window + 1))
        if window_years & set(exclude_confound_years):
            continue
        if len(pre_years) < 1 or len(post_years) < min_horizon:
            continue  # requires FULL min_horizon coverage, unlike event_study()

        baseline = annual.loc[pre_years, "redistribution_pct_gdp"].mean()
        trajectory = (annual.loc[post_years, "redistribution_pct_gdp"] - baseline).values
        trajectories.append(trajectory)

    trajectories = [t for t in trajectories if np.all(np.isfinite(t))]
    if len(trajectories) < 2:
        return None

    mean_traj = np.mean(trajectories, axis=0)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    auc = trapz_fn(mean_traj)
    return auc if np.isfinite(auc) else None


def permutation_test_auc(annual, wave_years, pre_window=2, post_window=5,
                          exclude_confound_years=(2020, 2021),
                          min_horizon=4, n_shuffles=2000, seed=0):
    """
    Null check for the event-study trajectory: is the observed mean
    deviation trajectory's AUC unusual relative to AUCs from randomly
    drawn "fake" event years of the same count?

    min_horizon=4 -> uses t+0..t+3 (4 points), the longest horizon with
    full coverage across all 6 real filtered events (2024 only has
    t+0/t+1, so it's excluded from this AUC calculation specifically,
    same tradeoff noted in event_study()'s docstring). This is a
    DIFFERENT, stricter subset than the descriptive t+0..t+5 table
    printed elsewhere -- deliberately, so the test isn't biased by
    horizons with sparse/unequal N.

    Null draws: n_events random years sampled (without replacement) from
    all years with enough surrounding data to be eligible at all, subject
    to the same exclude_confound_years filter as the real event set --
    keeps the null comparable rather than testing against a differently-
    constrained baseline. Overlap between drawn years is allowed (not
    penalized), since real strike waves cluster too and forcing artificial
    spacing in the null would bias the comparison.
    """
    rng = np.random.default_rng(seed)
    years_index = annual.index

    observed_auc = _trajectory_auc(annual, years_index, wave_years, pre_window,
                                    post_window, exclude_confound_years, min_horizon)
    if observed_auc is None:
        print("  Not enough real events with full horizon coverage to run the "
              "permutation test.")
        return None

    # Eligible pool for null draws: any year with enough calendar room for
    # pre_window before and min_horizon after, so a random draw isn't
    # systematically penalized relative to real events just for being near
    # the edge of the series.
    eligible = [y for y in years_index
                if (y - pre_window) in years_index or y - pre_window >= years_index.min()
                if (y + min_horizon - 1) <= years_index.max()]
    # Match the null draw size to how many events actually SURVIVED
    # filtering into observed_auc, not the raw wave_years count -- those
    # differ (10 raw vs 6 filtered here) and comparing AUCs from
    # differently-sized samples isn't a valid test.
    n_events_filtered = sum(
        1 for wy in wave_years
        if not (set(range(wy - pre_window, wy + post_window + 1)) & set(exclude_confound_years))
    )
    n_events = n_events_filtered

    null_aucs = []
    for _ in range(n_shuffles):
        draw = rng.choice(eligible, size=min(n_events, len(eligible)), replace=False)
        auc = _trajectory_auc(annual, years_index, draw, pre_window, post_window,
                               exclude_confound_years, min_horizon)
        if auc is not None:
            null_aucs.append(auc)

    null_aucs = np.array(null_aucs)
    p_value = np.mean(np.abs(null_aucs) >= np.abs(observed_auc))

    print(f"  Permutation test on mean-trajectory AUC (t+0..t+{min_horizon-1}, "
          f"{len(trajectories_note := wave_years)} real events ->"
          f"AUC computed on those surviving full-horizon + confound filtering)")
    print(f"  Observed AUC:        {observed_auc:+.4f}")
    print(f"  Null distribution:   mean={null_aucs.mean():+.4f}  std={null_aucs.std():.4f}  "
          f"(n={len(null_aucs)} valid draws of {n_shuffles})")
    print(f"  Two-sided empirical p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  -> Observed post-wave trajectory AUC is larger than expected under "
              "random year draws: some support for a real cumulative shift, not "
              "just noise from small-n event pooling.")
    else:
        print("  -> Observed AUC is unremarkable relative to random draws: the "
              "apparent post-wave climb is consistent with chance at this sample "
              "size, same caveat as the continuous-regression null elsewhere.")

    return dict(observed_auc=observed_auc, null_mean=null_aucs.mean(),
                null_std=null_aucs.std(), p_value=p_value, n_valid_draws=len(null_aucs))


def summarize(event_df):
    t_cols = [c for c in event_df.columns if c.startswith("t+")]
    if event_df.empty:
        print("  No usable strike-wave events with sufficient surrounding data.")
        return
    print(f"  {len(event_df)} real historical strike-wave events used")
    print("  Mean redistribution_pct_gdp deviation from pre-wave baseline:")
    means = event_df[t_cols].mean()
    print("   " + "  ".join(f"{v:+.3f}" for v in means))
    print("\n  Per-event detail:")
    print(event_df.round(3).to_string(index=False))


if __name__ == "__main__":
    print("=" * 70)
    print("HISTORICAL EVENT STUDY -- REAL DATA (not simulated)")
    print("=" * 70)

    by_country = load_and_prepare_panel()

    print("\n" + "#" * 70)
    print("PART A -- STATISTICAL WAVE YEARS (top 15% by civil_unrest_events)")
    print("#" * 70)
    for country, annual in sorted(by_country.items()):
        if 'redistribution_pct_gdp' not in annual.columns or annual['redistribution_pct_gdp'].dropna().empty:
            print(f"\n{country}: skipped, no redistribution_pct_gdp data at all")
            continue
        print(f"\n{'=' * 70}\n{country}\n{'=' * 70}")

        wave_years, threshold = identify_strike_wave_years(annual, top_pct=0.85)
        print(f"Strike-wave years (civil_unrest_events >= {threshold:.2f}, "
              f"top 15% of real years): {wave_years}")

        print("\nEvent study (COVID-window events excluded by default):")
        event_df = event_study(annual, wave_years)
        summarize(event_df)

        print("\nPermutation null check on the filtered trajectory's AUC:")
        permutation_test_auc(annual, wave_years)

    print("\n" + "#" * 70)
    print("PART B -- CURATED HISTORICAL LANDMARKS (known regime-change/uprising years)")
    print("#" * 70)
    for country, landmark_years in LANDMARK_YEARS.items():
        if country not in by_country:
            print(f"\n{country}: not in panel -- skipped")
            continue
        annual = by_country[country]
        print(f"\n{'=' * 70}\n{country}\n{'=' * 70}")

        testable_years = []
        for ly in landmark_years:
            cov = landmark_data_coverage(annual, ly)
            frac = cov['redistribution_pct_gdp'] / cov['window_len']
            print(f"  {ly}: redistribution_pct_gdp coverage "
                  f"{cov['redistribution_pct_gdp']}/{cov['window_len']} years "
                  f"({frac:.0%}), civil_unrest_events {cov['civil_unrest_events']}/"
                  f"{cov['window_len']}, political_violence_score "
                  f"{cov['political_violence_score']}/{cov['window_len']}, "
                  f"gini_coefficient {cov['gini_coefficient']}/{cov['window_len']}")
            if frac >= LANDMARK_MIN_COVERAGE_FRAC:
                testable_years.append(ly)
            else:
                print(f"    -> below {LANDMARK_MIN_COVERAGE_FRAC:.0%} redistribution "
                      f"coverage threshold, excluded from the event study below")

        if not testable_years:
            print(f"  No landmark year for {country} clears the coverage floor -- "
                  f"no event study run for this country.")
            continue

        print(f"\n  Running event study on: {testable_years}")
        event_df = event_study(annual, testable_years, pre_window=2, post_window=5)
        summarize(event_df)

    print("\n" + "=" * 70)
    print("Reminder: this is REAL HISTORICAL DATA, distinct from the ABM's")
    print("simulated reform_vs_revolution_test() (experiments.py) and from")
    print("rolling_window_refit.py's re-fitted coefficients.")
    print("=" * 70)