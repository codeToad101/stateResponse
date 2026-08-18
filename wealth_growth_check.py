"""
wealth_growth_check.py

============================================================================
 USES: REAL HISTORICAL DATA ONLY. No ABM simulation runs here.
============================================================================

Tests the "welfare grows, but not enough" premise directly on real data:
does redistribution_pct_gdp have a statistically real positive trend over
1960-2025, does gini_coefficient also have a statistically real positive
trend over the same span, and if so, is Gini's trend large/persistent
enough that the welfare-growth trend isn't keeping pace?

This is the empirical precondition for the ABM-side reinvestment/on-off
comparison in experiments.py's run_gini_growth_test() -- if either trend
isn't real here, the ABM comparison needs to be interpreted very
differently (or the claim itself needs to be dropped/reframed).
"""

import numpy as np
from scipy.stats import linregress

from agents.state import load_and_prepare_panel, load_regime_periods


def trend_test(annual, col, label=None):
    """
    OLS trend of `col` on year. Returns slope (units/year), R², p-value,
    plus a CAGR-style average annual %-growth figure for interpretability
    alongside the raw slope (a Gini slope and a %-GDP slope aren't on
    comparable scales, so both a raw trend and a relative-growth number
    are reported -- neither alone answers "is it enough").
    """
    label = label or col
    series = annual[col].dropna()
    years = series.index.to_numpy(dtype=float)
    values = series.to_numpy(dtype=float)

    result = linregress(years, values)
    slope, r, p = result.slope, result.rvalue, result.pvalue

    first, last = values[0], values[-1]
    n_years = years[-1] - years[0]
    cagr = (last / first) ** (1 / n_years) - 1 if first > 0 and n_years > 0 else np.nan

    verdict = "real upward trend" if (p < 0.05 and slope > 0) else (
        "real downward trend" if (p < 0.05 and slope < 0) else
        "no statistically significant trend"
    )

    print(f"\n{label} trend, {int(years.min())}-{int(years.max())} (n={len(values)}):")
    print(f"  slope = {slope:+.5f} / year   R\u00b2 = {r**2:.3f}   p = {p:.4f}   -> {verdict}")
    print(f"  {first:.3f} -> {last:.3f}   (CAGR ~ {cagr*100:+.2f}%/yr)")

    return dict(col=col, slope=slope, r2=r**2, p=p, first=first, last=last,
                cagr=cagr, verdict=verdict)

def regime_split_trend_tests(annual, regime_periods, country, min_n=6):
    """
    Runs trend_test separately within each regime-era for this country
    (see regime_periods.csv), in ADDITION to the whole-span test the
    caller already ran. min_n=6 (per your call) -- an era with fewer
    non-null years than that is skipped and reported as skipped, not
    silently fit anyway.
    """
    rows = regime_periods[regime_periods['country'] == country]
    era_results = []
    for _, row in rows.iterrows():
        era_annual = annual[(annual.index >= row['start_year']) & (annual.index <= row['end_year'])]
        for col, label in [('redistribution_pct_gdp', 'Redistribution (% of GDP)'),
                            ('gini_coefficient', 'Gini coefficient')]:
            n_avail = era_annual[col].dropna().shape[0] if col in era_annual.columns else 0
            era_tag = f"{row['regime_label']} {row['start_year']}-{row['end_year']}"
            if n_avail < min_n:
                print(f"    [{era_tag}] {label}: skipped, only {n_avail} "
                      f"non-null years (< {min_n})")
                continue
            print(f"    [{era_tag}, confidence={row['confidence']}]")
            result = trend_test(era_annual, col, label)
            result.update(regime_label=row['regime_label'],
                           era=f"{row['start_year']}-{row['end_year']}")
            era_results.append(result)
    return era_results

if __name__ == "__main__":
    print("=" * 70)
    print("WEALTH/WELFARE GROWTH CHECK -- REAL DATA (not simulated)")
    print("=" * 70)

    by_country = load_and_prepare_panel()
    regime_periods = load_regime_periods()

    for country, annual in sorted(by_country.items()):
        if ('redistribution_pct_gdp' not in annual.columns
                or annual['redistribution_pct_gdp'].dropna().empty):
            print(f"\n{country}: skipped, no redistribution data at all")
            continue

        print(f"\n{'=' * 70}\n{country}\n{'=' * 70}")
        n_rows = annual.notna().any(axis=1).sum()
        print(f"Loaded {n_rows} annual rows ({annual.index.min()}-{annual.index.max()})")

        redist_result = trend_test(annual, "redistribution_pct_gdp", "Redistribution (% of GDP)")
        gini_result = (trend_test(annual, "gini_coefficient", "Gini coefficient")
                       if 'gini_coefficient' in annual.columns else None)

        print("\n  Regime-era breakdown:")
        regime_split_trend_tests(annual, regime_periods, country, min_n=6)

        print("\n  " + "-" * 66)
        print("  INTERPRETATION")
        print("  " + "-" * 66)
        if (gini_result and redist_result["verdict"] == "real upward trend"
                and gini_result["verdict"] == "real upward trend"):
            print(f"  Both series show a real upward trend over the full span for {country} --")
            print("  the empirical precondition for 'welfare grows, but Gini grows too' holds.")
            print(f"  Redistribution CAGR {redist_result['cagr']*100:+.2f}%/yr vs. Gini CAGR "
                  f"{gini_result['cagr']*100:+.2f}%/yr -- not directly comparable (different")
            print("  units), descriptive support only; the ABM on/off comparison tests adequacy.")
        else:
            print(f"  At least one whole-span trend is not statistically significant for "
                  f"{country}.")
            print("  The 'welfare grows but isn't enough' framing needs re-examination for")
            print("  this country specifically before leaning on it in cross-country claims.")