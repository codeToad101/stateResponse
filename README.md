# State Fiscal Response to Class Conflict: An Agent-Based Model

**Author:** Asiya Kadivar
**Institution:** Carnegie Mellon University
**Status:** Work in progress (private) — data pipeline validated, response function calibrated (v1), ABM implementation not yet started
**Timeline:** July–October 2026
**Target Venue:** Journal of Artificial Societies and Social Simulation (JASSS) or heterodox economics outlets

> **Note to anyone encountering this repo mid-development:** the state response function below is a first calibrated pass, not a final result. Several known gaps are flagged explicitly in "Open Questions & Next Steps" — please read that section before citing any coefficient here. This is an active research project, not a finished finding.

---

## Overview

This project models the **US state's fiscal policy response to labor market conditions and conflict intensity** using agent-based modeling (ABM). The core innovation: instead of treating fiscal policy (taxation, redistribution) as exogenous or optimizing in a rational-expectations framework, we model the state as a *reactive agent* that observes economic indicators (inequality, unemployment, protest) and adjusts redistributive spending accordingly.

**Central research question:** Is state fiscal response to class conflict *structural* (persistent policy reorientation) or *cyclical* (temporary pacification)? And what state regime (representative, captured, unresponsive) best explains observed US policy trajectories?

This goal hasn't changed. What's changed since the original draft is the data pipeline (several real bugs fixed), the fitting methodology (more honest about uncertainty and model flexibility), and — new — a plan to explicitly parameterize *which kind of state* the ABM simulates, rather than assuming one fixed response rule.

### Theoretical Motivation

- **Epstein (2002):** Civil violence models show how state repression + grievance interact at the micro level
- **Acemoglu & Robinson (2000):** Redistribution as political equilibrium—elites tolerate it to avoid uprising
- **Piketty et al. (2014, 2015):** Tax policy responds to political pressure and distributional conflict
- **Alesina & Rodrik (1994):** Tradeoff between redistribution and growth not mechanically determined—policy choice matters

**Gap:** No ABM has endogenized state fiscal response to *endogenous* worker grievance + protest dynamics. Most models treat policy as exogenous or assume rational welfare maximization.

### Ideological Stakes

Two interpretations of results:
1. **Reform optimism:** If redistribution is *sticky* (ratchets up, persists), suggests genuine policy gains possible via democratic struggle
2. **Revolutionary critique:** If redistribution is *cyclical* (state reverses gains when conflict subsides), suggests structural oppression requires systemic change, not reform

We present both without endorsing—the empirics speak. **Caveat as of this draft:** our current fit can't yet distinguish these cleanly — see "Protest Signal" below.

---

## Model Architecture

### Agents

**Workers** (N ≈ 1000–5000)
- Heterogeneous skill, wage, network position
- Grievance = f(wage gap relative to reference point, inequality trend)
- Protest threshold: if (Grievance − Risk) > Threshold, join active protest
- Risk = f(state repression capacity, network visibility)

**Firms** (N ≈ 50–200)
- Heterogeneous productivity
- Wage-setting: mechanical response to labor market tightness, not strategic suppression
- Hiring/firing follows profit dynamics (LIFO)

**State** (N = 1, endogenous decision-maker)
- **Observes:** Gini(t), unemployment(t), protest intensity(t), wage growth(t)
- **Decides:** tax rate τ(t), redistribution R(t)
- **Response function:** calibrated empirically (see Methodology) rather than assumed
- **Regimes (planned, not yet implemented — see Roadmap):** the response function's parameters/who it responds *to* will vary by regime type: representative, captured/aristocratic (biased toward elite preferences over popular grievance), and dictatorship/unresponsive (repression-weighted, largely ignores conflict signal)

### Market Dynamics

- **Labor clearing:** Wages adjust until supply ≈ demand (with frictions)
- **Feedback loops:**
  - High unemployment → low wages → high relative grievance → high protest
  - High protest → state increases redistribution → worker consumption → demand → hiring
  - Inequality widens → grievance increases independent of absolute wages

---

## Data Pipeline (updated)

Source data: FRED (automated pulls), Census/CPS, Tax Foundation, CNTSDA/BLS strike data, SSA/USDA/HHS program spending. Full source list unchanged from original design — see `sources.txt`.

**Fixes applied since first draft:**
- `gdp_billions` was pulling `A191RL1Q225SBEA` (% change from preceding period) instead of a dollar level. Corrected to FRED series `GDP` (Billions of Dollars, SAAR, quarterly).
- Annual source files (Gini, EITC, federal tax rate, Medicaid) were being misparsed: bare-year integer columns (`1970`, `1971`, ...) were passed straight into `pd.to_datetime()`, which silently interprets raw integers as nanosecond epoch offsets rather than years. Every row collapsed to ~1970, and a later hardcoded `is_fiscal_year = True` override compounded it by re-stamping and averaging all years into one blended value per series. Result: `gini_coefficient`, `redist_eitc`, and `redist_medicaid` were flat-lined for the entire 1970–2025 range. Fixed at the date-parsing step; parameter override removed so `is_fiscal_year` is respected per call site.
- `protest_intensity_score` was structurally stuck near 0.5 for the whole series. Cause: `days_idle` (a cumulative person-days-idle figure, often in the thousands) was compared directly against a flat 365-day denominator and clipped — the clip triggered almost every quarter regardless of actual severity. Rebuilt to normalize against total *available* person-days per quarter, log-transform both components (strike activity is heavily right-skewed), and min-max rescale — now spans a real 0–1 range instead of saturating.

### Annual vs. Quarterly (design decision, not a limitation to hide)

The state's fiscal decisions (redistribution, Gini, tax rate) are sourced from genuinely annual data; quarterly figures in the raw CSV are forward-filled duplicates for calendar alignment, not four independent observations. Fitting on the raw ffilled series would inflate N artificially. **Current approach:** collapse to one observation per year for fitting the response function, while aggregating quarterly-resolution predictors (protest, GDP growth) within each year first, so real sub-annual variation still feeds the model rather than being discarded outright.

This does cost some interpretability — we can't yet resolve *within-year* timing (e.g., a Q2 protest spike vs. a Q4 one) on the response side. That resolution matters more for the ABM's agent-level dynamics than for this aggregate fitting step, and is a natural place to reintroduce quarterly granularity once the ABM is running (see Roadmap).

---

## State Response Function — Methodology (v1)

$$R(t) = f(\text{Protest}(t-\tau), \text{Gini}(t), \text{Growth}(t))$$

**Departure from original plan:** rather than committing to one fixed functional form (e.g. a static polynomial), we fit a family of candidates — linear, logistic (saturating), exponential, and a GAM (data-driven smooth shape per predictor) — and compare via AIC/BIC rather than assuming which shape is "right." As of this draft, the exponential form wins on AIC; GAM has higher raw R² but its extra flexibility is penalized appropriately given our sample size (n≈63 annual observations).

**Uncertainty reporting:** initial fits used curve_fit's default covariance, which assumes independent residuals. Durbin-Watson diagnostics showed meaningful residual autocorrelation (expected for a multi-decade macro series — recessions and policy regimes span multiple years). Standard errors and confidence intervals are now also reported via a Newey-West/HAC sandwich estimator, which doesn't assume independence across years. This widened CIs somewhat but did not change the qualitative conclusions below.

### Protest Signal (open question, not yet resolved)

Across every functional form tried so far, the protest coefficient's confidence interval crosses zero — including under the more conservative HAC correction. At present we cannot claim a measurable protest → redistribution effect. Two live hypotheses, not yet distinguished:
1. The effect is genuinely weak/absent at this level of aggregation.
2. It's a measurement problem — our severity score is annual-only for large stretches of the series, and strike-only (CNTSDA) rather than pulling in broader protest-event data (e.g. ACLED).

---

## Open Questions & Next Steps

1. **Protest signal (measurement vs. real absence):** improve resolution before drawing conclusions — recover quarterly granularity for post-1988 strike data (currently annualized despite the source having actual stoppage dates), and evaluate swapping/supplementing CNTSDA with ACLED for broader, more recent event coverage.
2. **Period-by-period breakdown:** the current fit assumes one fixed relationship across the full 1960–2025 span. Given the paper's own reform-vs-revolution framing, this should be tested directly — rolling-window or regime-split fits (e.g. pre/post-1980) rather than assumed stability.
3. **Indirect / compositional correlation:** beyond total redistribution level, test whether protest correlates with *shifts in spending composition* (e.g. EITC up while Medicaid flat) or with *volatility* of spending, rather than only its level — a null result on the level doesn't rule out a compositional or predictability effect.
4. **State regimes in the ABM (new, planned):** implement the `State` agent with selectable regime types — representative (responds to the fitted function as-is), captured/aristocratic (response weighted toward elite/firm preferences over popular grievance, reflecting the "state serves capital" argument in the heterodox literature this project engages with), and dictatorship/unresponsive (repression-prioritized, conflict signal largely ignored). This turns the reform-vs-revolution question into something the simulation can actually run experiments on, rather than only inferring from the historical fit.
5. **ABM implementation:** `Worker`, `Firm`, `State` classes, Mesa scheduling, validation against the historical series once the above are in better shape.

---

## Data Sources & Rationale

*(Unchanged from original — automated FRED pulls for unemployment, LFPR, wages, job openings, GDP, UI/workers' comp/veterans benefits/Medicaid; manually collected Gini, wage percentiles, SNAP, EITC, federal tax rate, strikes/protests. See `sources.txt` for full listing.)*

---

## Mathematical Foundations

### Worker Grievance

$$G_i(t) = \frac{\text{reference\_wage}(t) - w_i(t)}{\text{reference\_wage}(t)}$$

where reference wage = local average wage × (1 − Gini).

### Worker Protest Decision

$$\text{Protest}_i(t) = \begin{cases} 1 & \text{if } G_i(t) - R_i(t) > \theta_i \\ 0 & \text{otherwise} \end{cases}$$

**Source:** Epstein (2002) civil violence model; adapted for labor context.

### State Response Function

$$R(t) = \beta_0 + \beta_1 \cdot \text{Protest}(t - \tau) + \beta_2 \cdot \text{Gini}(t) + \beta_3 \cdot \text{Growth}(t) + \epsilon(t)$$

Fit via multiple functional forms (linear, logistic, exponential, GAM), compared on AIC/BIC with HAC-corrected uncertainty (see Methodology above for current results and caveats).

---

## Known Limitations & Caveats

1. **Firm wage-setting:** mechanical, not strategic. (Mitigation: sensitivity test with mild monopsony, planned.)
2. **State as single agent (partially addressed):** regime types (see Roadmap) will introduce heterogeneity without fully modeling Treasury/Congress/Fed contestation.
3. **Protest data resolution:** see "Protest Signal" above — actively being worked on, not a settled limitation.
4. **Causality unclear:** does redistribution *cause* lower protest, or do high-protest periods *select* into high redistribution? ABM assumes reaction; history is endogenous. Honest limitation, not something the current design resolves.
5. **Demand-side omitted:** no consumption dynamics, credit, or asset bubbles modeled. Scope boundary, not a bug.
6. **Sample size:** n≈63 annual observations for a 3–4 parameter model is workable but not large; treat current coefficients as a first pass, not a confirmed result — this is the main reason items 1–2 in "Open Questions" matter before drawing conclusions for the paper.

---

## Ideological Transparency

This project is grounded in heterodox political economy (Marxist-adjacent but not doctrinaire). We test whether capitalism naturally stabilizes via redistribution (reform) or requires systemic change (revolution).

**We are not neutral on these questions, but the model is.** Results will speak for themselves — which is also why the honest-uncertainty reporting above (wide CIs, unresolved protest signal) is being kept in this README rather than smoothed over: a heterodox project claiming a clean result it can't yet support would undercut its own credibility fastest.

---

## References (Preliminary)

- Acemoglu, D., & Robinson, J. A. (2000). Political losers as a barrier to economic development. *The American Economic Review*, 90(2), 126–130.
- Alesina, A., & Rodrik, D. (1994). Distributive politics and economic growth. *The Quarterly Journal of Economics*, 109(2), 465–490.
- Epstein, J. M. (2002). Modeling civil violence: An agent-based computational approach. *Proceedings of the National Academy of Sciences*, 99(Suppl 3), 7243–7250.
- Piketty, T., Saez, E., & Stantcheva, S. (2014). Optimal taxation of top labor incomes. *Journal of Political Economy*, 122(2), 231–271.
- Manning, A. (2011). Imperfect competition and macroeconomics. *The Economic Journal*, 121(554), 45–65.

---

## Repository Structure

```
state_response_abm/
├── README.md (this file)
├── us_state_response_data.py (data collection — GDP + date-parsing fixes applied)
├── state.py (data aggregation + response function fitting; ABM State agent scaffolded)
├── firm.py / worker.py (ABM agents — pending regime-aware integration)
├── abm_model.py (Mesa ABM implementation) [TO DO]
├── experiments.py (regime comparisons, period-by-period fits) [TO DO]
├── data/
│   ├── us_state_response_data.csv (raw quarterly output)
│   └── manual_collections/ (wage percentiles, Gini, protest data)
└── results/
    ├── plots/ (calibration fits, time series) [TO DO]
    └── experiments/ (regime comparisons) [TO DO]
```

---

**Last updated:** August 2026
**Status:** Response function v1 calibrated; protest signal, period stability, and state regimes flagged as open work before ABM implementation begins.