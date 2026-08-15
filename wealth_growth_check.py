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

from agents.state import load_and_prepare_data


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


if __name__ == "__main__":
    print("=" * 70)
    print("WEALTH/WELFARE GROWTH CHECK -- REAL DATA (not simulated)")
    print("=" * 70)

    annual = load_and_prepare_data()
    print(f"\nLoaded {len(annual)} annual observations "
          f"({annual.index.min()}-{annual.index.max()})")

    redist_result = trend_test(annual, "redistribution_pct_gdp", "Redistribution (% of GDP)")
    gini_result = trend_test(annual, "gini_coefficient", "Gini coefficient")

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    if redist_result["verdict"] == "real upward trend" and gini_result["verdict"] == "real upward trend":
        print("Both series show a real upward trend over the full span -- the empirical")
        print("precondition for the 'welfare grows, but Gini grows too, so it's not")
        print("enough' claim holds. Whether redistribution's growth rate")
        print(f"({redist_result['cagr']*100:+.2f}%/yr) is actually 'keeping pace' with Gini's")
        print(f"({gini_result['cagr']*100:+.2f}%/yr) is NOT resolved by comparing these two")
        print("CAGRs directly -- different scales/units. Treat this as descriptive")
        print("support for the premise; the ABM on/off comparison is what actually")
        print("tests adequacy.")
    else:
        print("At least one trend is NOT statistically significant over the full span.")
        print("The 'welfare grows but isn't enough' claim needs re-examination or")
        print("reframing before proceeding to the ABM comparison -- don't build a")
        print("mechanism to explain a premise that isn't actually supported here.")