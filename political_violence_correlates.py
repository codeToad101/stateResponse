"""
political_violence_correlates.py

============================================================================
 USES: REAL HISTORICAL DATA ONLY. No ABM simulation runs here.
============================================================================

Tests two things per country, contemporaneous and lagged:
  (a) political_violence_score <-> civil_unrest_events -- does state
      violence against civilians correlate with MORE protest (provoked)
      or LESS (suppressed)?
  (b) political_violence_score <-> change in redistribution_pct_gdp --
      do high-repression states substitute violence for fiscal
      concession, i.e. does higher political_violence_score coincide
      with a smaller/negative redistribution response?

political_violence_score is currently absent for the United States (see
state.py's module docstring) -- this script runs on whichever countries
actually have it, and says so rather than silently dropping the US.
"""

import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from agents.state import load_and_prepare_panel, load_regime_periods


def permutation_correlation_test(a, b, n_shuffles=2000, seed=0):
    rng = np.random.default_rng(seed)
    observed_r = np.corrcoef(a, b)[0, 1]
    b_vals = b.to_numpy()
    null_rs = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(b_vals)
        null_rs[i] = np.corrcoef(a, shuffled)[0, 1]
    p_value = float(np.mean(np.abs(null_rs) >= np.abs(observed_r)))
    return float(observed_r), p_value


def lagged_correlation(annual, col_a, col_b, lag=0, min_n=6,
                        exclude_years=None, n_shuffles=2000, seed=0):
    if col_a not in annual.columns or col_b not in annual.columns:
        return None
    a = annual[col_a]
    b = annual[col_b].shift(-lag)  # lag>0: b measured `lag` years AFTER a
    sub = pd_concat_dropna(a, b)
    if exclude_years:
        sub = sub[~sub.index.isin(exclude_years)]
    if len(sub) < min_n:
        return dict(n=len(sub), skipped=True)
    r, p = permutation_correlation_test(sub.iloc[:, 0], sub.iloc[:, 1],
                                         n_shuffles=n_shuffles, seed=seed)
    return dict(n=len(sub), r=r, p=p, skipped=False)


def pd_concat_dropna(a, b):
    import pandas as pd
    return pd.concat([a, b], axis=1).dropna()


CONFOUND_YEARS_REDIST_ONLY = (2020, 2021)


def run_country(country, annual, regime_periods, lags=(0, 1, 2, 3, 4)):
    print(f"\n{'=' * 70}\n{country}\n{'=' * 70}")
    if 'political_violence_score' not in annual.columns or annual['political_violence_score'].dropna().empty:
        print("  no political_violence_score data for this country -- skipped")
        return []

    base = annual.copy()
    base['d_political_violence_score'] = base['political_violence_score'].diff()
    if 'redistribution_pct_gdp' in base.columns:
        base['d_redistribution_pct_gdp'] = base['redistribution_pct_gdp'].diff()

    flat_results = []
    era_masks = {'all_years': None}
    if regime_periods is not None:
        rows = regime_periods[regime_periods['country'] == country]
        for _, row in rows.iterrows():
            label = f"{row['regime_label']}_{row['start_year']}-{row['end_year']}"
            era_masks[label] = (base.index >= row['start_year']) & (base.index <= row['end_year'])

    for era_label, era_mask in era_masks.items():
        era_df = base if era_mask is None else base[era_mask]
        print(f"\n  === era: {era_label} (n_years={len(era_df)}) ===")

        for pvs_col, pvs_label in [('political_violence_score', 'level'),
                                    ('d_political_violence_score', 'differenced')]:
            print(f"\n  --- political_violence_score ({pvs_label}) ---")

            print("  (a) political_violence_score <-> civil_unrest_events (2020-2021 included)")
            for lag in lags:
                res = lagged_correlation(era_df, pvs_col, 'civil_unrest_events', lag=lag)
                if res is None:
                    print("      civil_unrest_events not available -- skipped")
                    break
                tag = f"{country}|{era_label}|{pvs_label}|unrest|lag{lag}"
                if res['skipped']:
                    print(f"    lag={lag}: skipped, only n={res['n']} overlapping years")
                else:
                    print(f"    lag={lag}: r={res['r']:+.3f}  p={res['p']:.4f}  n={res['n']}")
                    flat_results.append(dict(tag=tag, **res))

            print("  (b) political_violence_score <-> change in redistribution_pct_gdp "
                  "(2020-2021 EXCLUDED)")
            if 'redistribution_pct_gdp' in era_df.columns:
                for lag in lags:
                    res = lagged_correlation(era_df, pvs_col, 'd_redistribution_pct_gdp', lag=lag,
                                              exclude_years=CONFOUND_YEARS_REDIST_ONLY)
                    tag = f"{country}|{era_label}|{pvs_label}|redist|lag{lag}"
                    if res['skipped']:
                        print(f"    lag={lag}: skipped, only n={res['n']} overlapping years")
                    else:
                        print(f"    lag={lag}: r={res['r']:+.3f}  p={res['p']:.4f}  n={res['n']}")
                        flat_results.append(dict(tag=tag, **res))
            else:
                print("    redistribution_pct_gdp not available -- skipped")

    return flat_results


if __name__ == "__main__":
    print("=" * 70)
    print("POLITICAL VIOLENCE CORRELATES -- REAL DATA (not simulated)")
    print("=" * 70)
    print("\np-values below are PERMUTATION-based (2000 shuffles), not pearsonr's")
    print("parametric p. 2020-2021 excluded ONLY from the redistribution")
    print("relationship, not civil_unrest_events.")

    by_country = load_and_prepare_panel()
    all_flat_results = []
    regime_periods = load_regime_periods()
    for country in sorted(by_country):
        all_flat_results.extend(run_country(country, by_country[country], regime_periods))

    print("\n" + "=" * 70)
    print("MULTIPLE-COMPARISON CORRECTION (Holm-Bonferroni), across all "
          f"{len(all_flat_results)} tests run above")
    print("=" * 70)
    if all_flat_results:
        raw_p = [res['p'] for res in all_flat_results]
        reject, corrected_p, _, _ = multipletests(raw_p, alpha=0.05, method='holm')
        for res, corr_p, rej in sorted(zip(all_flat_results, corrected_p, reject),
                                        key=lambda x: x[1]):
            flag = "  *** SURVIVES CORRECTION ***" if rej else ""
            print(f"  {res['tag']:36} r={res['r']:+.3f}  raw_p={res['p']:.4f}  "
                  f"holm_p={corr_p:.4f}  n={res['n']}{flag}")
        n_survive = sum(reject)
        print(f"\n{n_survive} / {len(all_flat_results)} tests survive Holm-Bonferroni "
              f"correction at alpha=0.05.")
    else:
        print("  No usable tests were run.")