# State Fiscal Response to Class Conflict: An Agent-Based Model

**Author:** Asiya Kadivar
**Institution:** Carnegie Mellon University
**Status:** Work in progress (private) — data pipeline validated, response function calibrated (v1), ABM implemented and running (Worker/Firm/State agents, three regimes), first full experiment suite complete. Two known bugs from that first run have been fixed; results below are post-fix.
**Timeline:** July–October 2026
**Target Venue:** Journal of Artificial Societies and Social Simulation (JASSS) or heterodox economics outlets

> **Note to anyone encountering this repo mid-development:** the results below are a first post-fix pass, not a final finding. Several open questions are flagged explicitly in "Open Questions & Next Steps" — please read that section before citing any number here. This is an active research project, not a finished result.

---

## Overview

This project models the **US state's fiscal policy response to labor market conditions and conflict intensity** using agent-based modeling (ABM). The core innovation: instead of treating fiscal policy (taxation, redistribution) as exogenous or optimizing in a rational-expectations framework, we model the state as a *reactive agent* that observes economic indicators (inequality, unemployment, protest) and adjusts redistributive spending accordingly.

**Central research question:** Is state fiscal response to class conflict *structural* (persistent policy reorientation) or *cyclical* (temporary pacification)? And what state regime (representative, captured, dictatorship) best explains observed US policy trajectories?

This goal hasn't changed. What's changed since the original draft: the data pipeline (several real bugs fixed), the fitting methodology (more honest about uncertainty and model flexibility), a working ABM with three selectable regimes, a full experiment suite (historical validation, regime comparison, reform-vs-revolution test, sensitivity sweep, permutation null check), and two data-integrity bugs caught and fixed in that suite's first output.

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

We present both without endorsing—the empirics speak. **Caveat as of this draft:** protest is currently rare-to-near-zero across all three regimes post-fix, which limits how cleanly the reform-vs-revolution test can distinguish these two stories right now — see "Current Results" and "Protest Signal" below.

---

## Model Architecture

### Agents

**Workers** (N ≈ 1000–5000)
- Heterogeneous skill, wage, network position
- Grievance = f(wage gap relative to reference point, inequality trend)
- Protest threshold: if (Grievance − Risk) > Threshold, join active protest
- Risk = f(state repression capacity, network visibility)
- Capital-income channel: savings above a consumption floor can be invested (heterogeneous per-agent propensity), earning a stochastic return

**Firms** (N ≈ 50–200)
- Heterogeneous productivity
- Wage-setting: mechanical response to labor market tightness, not strategic suppression
- Hiring/firing follows profit dynamics (LIFO)
- Owner income + owner investment income feed into the population income Gini alongside worker income

**State** (N = 1, endogenous decision-maker)
- **Observes:** Gini(t), unemployment(t), protest intensity(t), wage growth(t)
- **Decides:** tax rate τ(t), redistribution R(t)
- **Response function:** calibrated empirically (see Methodology) rather than assumed
- **Regimes (implemented):** representative, captured (biased toward elite/firm preferences over popular grievance), dictatorship (repression-weighted, largely ignores conflict signal). All three are wired into the ABM and used in regime-comparison experiments below.

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
- Annual source files (Gini, EITC, federal tax rate, Medicaid) were being misparsed: bare-year integer columns (`1970`, `1971`, ...) were passed straight into `pd.to_datetime()`, which silently interprets raw integers as nanosecond epoch offsets rather than years. Fixed at the date-parsing step; parameter override removed so `is_fiscal_year` is respected per call site.
- `protest_intensity_score` was structurally stuck near 0.5 for the whole series. Rebuilt to normalize `days_idle` against total *available* person-days per quarter, log-transform both components, and min-max rescale — now spans a real 0–1 range.

### Annual vs. Quarterly (design decision, not a limitation to hide)

The state's fiscal decisions (redistribution, Gini, tax rate) are sourced from genuinely annual data; quarterly figures in the raw CSV are forward-filled duplicates for calendar alignment, not four independent observations. Current approach: collapse to one observation per year for fitting the response function, while aggregating quarterly-resolution predictors (protest, GDP growth) within each year first.

---

## State Response Function — Methodology (v1)

$$R(t) = f(\text{Protest}(t-\tau), \text{Gini}(t), \text{Growth}(t))$$

We fit a family of candidates — linear, logistic (saturating), exponential, and a GAM — and compare via AIC/BIC rather than assuming which shape is "right." The exponential form wins on AIC (n=63 annual observations, 1963–2025); GAM has higher raw R² but its flexibility is penalized appropriately given sample size. Durbin-Watson on the best model is 0.79, indicating residual autocorrelation typical of a multi-decade macro series — point estimates are usable, but standard/CI estimates from this fit may be too tight.

### Protest Signal (open question, not yet resolved)

Across every functional form tried, the protest coefficient's confidence interval crosses zero, including under HAC correction. A permutation null check on the historical `d_protest(t-1) → d_redistribution(t)` relationship (2000 shuffles) confirms this isn't a modeling artifact: observed correlation = −0.0192, p = 0.87 — indistinguishable from chance. Two live hypotheses, not yet distinguished:
1. The effect is genuinely weak/absent at this level of aggregation.
2. It's a measurement problem — annual-only, strike-only (CNTSDA) severity data for large stretches of the series.

---

## Current Results (first post-fix experiment suite)

Two bugs were caught in the first full run of `experiments.py` and are fixed as of this update:
- **Gini out-of-range values** (means/CIs above 1.0, e.g. 2.68): caused by unfloored negative investment/owner income feeding the Gini calculation. Fixed by clipping the income vector at 0 before computing Gini (not by altering the underlying wealth/investment mechanics, which are left intact) plus an assertion that Gini stays in [0,1].
- **Dictatorship reform-vs-revolution test was invalid**: with protest near-zero for that regime, the 90th-percentile "spike" threshold itself computed to 0, so the spike condition matched every tick rather than real outliers, producing a fabricated monotonic "response" curve. Fixed by requiring the threshold to be strictly positive and the tick itself nonzero; dictatorship now correctly reports 0 usable spike events rather than a fake trajectory.

**Post-fix findings:**
- Regime-comparison Gini is now bounded and plausible (0.471–0.476 across regimes), consistent with real US income-Gini figures (~0.47–0.49).
- Protest share collapsed to near-zero across all three regimes (0.0000–0.0041), down from ~0.20–0.24 pre-fix. This tracks: the old out-of-range Gini was driving `macro_reference = avg_wage × (1 − gini)` negative, artificially inflating worker grievance. With Gini fixed, protest now reflects the model's actual (currently low) grievance-risk gap.
- Regime differentiation in Gini is small and CIs largely overlap (captured 0.4709 vs. representative 0.4760) — theoretically we'd expect captured regimes to produce visibly worse inequality; this convergence is a genuine open question, not settled.
- Reform-vs-revolution event study is no longer degenerate for dictatorship (0 events, correctly reported as insufficient data) but is now event-count-thin for captured/representative too (463 and 388 pooled spikes off a ~0.1–0.4% protest base) — treat those curves as suggestive, not conclusive, until protest calibration is revisited.
- Permutation null check remains solid and consistent pre- and post-fix (p = 0.87).

---

## Open Questions & Next Steps

1. **Protest signal (measurement vs. real absence):** improve resolution before drawing conclusions — recover quarterly granularity for post-1988 strike data, and evaluate swapping/supplementing CNTSDA with ACLED for broader, more recent event coverage.
2. **Near-zero post-fix protest — calibration sanity check (new):** confirm whether the current near-zero protest share is a correct reflection of the fixed grievance dynamics, or whether the logistic protest-decision steepness/threshold parameters need revisiting now that the earlier (buggy) grievance inflation is gone. Needed before leaning on the reform-vs-revolution event counts above.
3. **Regime differentiation in Gini — sanity check (new):** verify that `regime` actually drives income/wealth dynamics and not only protest suppression (via `police_intensity`/risk perception). If captured vs. representative Gini stays this close after the protest-calibration check above, that's a substantive finding worth investigating in the regime response-function weighting itself, not just a threshold artifact.
4. **Period-by-period breakdown:** the current fit assumes one fixed relationship across the full 1960–2025 span. Rolling-window or regime-split fits (e.g. pre/post-1980) rather than assumed stability.
5. **Indirect / compositional correlation:** test whether protest correlates with *shifts in spending composition* (e.g. EITC up while Medicaid flat) or with *volatility* of spending, rather than only its level.
6. **Full per-permutation BVAR refit for the null check:** current version permutes a single lagged correlation rather than refitting the full shrinkage system each permutation — a real but smaller-scope check; a full refit would be more rigorous.

---

## Data Sources & Rationale

*(Unchanged from original — automated FRED pulls for unemployment, LFPR, wages, job openings, GDP, UI/workers' comp/veterans benefits/Medicaid; manually collected Gini, wage percentiles, SNAP, EITC, federal tax rate, strikes/protests. See `sources.txt` for full listing.)*

---

## Mathematical Foundations

### Worker Grievance

$$G_i(t) = \frac{\text{reference\_wage}(t) - w_i(t)}{\text{reference\_wage}(t)}$$

where reference wage = local average wage × (1 − Gini).

### Worker Protest Decision

Probabilistic (logistic), not a hard cutoff — see `worker.py` for the full form:

$$p_i(t) = \sigma\big(k \cdot [(G_i(t) - R_i(t)) - \theta_i]\big)$$

**Source:** Epstein (2002) civil violence model; adapted for labor context.

### State Response Function

$$R(t) = \beta_0 + \beta_1 \cdot \text{Protest}(t - \tau) + \beta_2 \cdot \text{Gini}(t) + \beta_3 \cdot \text{Growth}(t) + \epsilon(t)$$

Fit via multiple functional forms (linear, logistic, exponential, GAM), compared on AIC/BIC with HAC-corrected uncertainty.

---

## Known Limitations & Caveats

1. **Firm wage-setting:** mechanical, not strategic. (Mitigation: sensitivity test with mild monopsony — included in the lightweight sweep above; more exhaustive sweep still open.)
2. **State as single agent (partially addressed):** three regime types now implemented, but this still doesn't fully model Treasury/Congress/Fed contestation.
3. **Protest data resolution:** see "Protest Signal" and Open Question #1 above — actively being worked on, not a settled limitation.
4. **Causality unclear:** does redistribution *cause* lower protest, or do high-protest periods *select* into high redistribution? ABM assumes reaction; history is endogenous.
5. **Demand-side omitted:** no consumption dynamics, credit, or asset bubbles modeled. Scope boundary, not a bug.
6. **Sample size:** n≈63 annual observations for a 3–4 parameter response-function fit is workable but not large; treat current coefficients as a first pass.
7. **Reform-vs-revolution event counts are currently thin** given near-zero post-fix protest (see Open Question #2) — don't over-read the current spike-response curves until that's resolved.

---

## Ideological Transparency

This project is grounded in heterodox political economy (Marxist-adjacent but not doctrinaire). We test whether capitalism naturally stabilizes via redistribution (reform) or requires systemic change (revolution).

**We are not neutral on these questions, but the model is.** Results will speak for themselves — which is also why the honest-uncertainty reporting above (wide CIs, unresolved protest signal, thin post-fix event counts) is being kept in this README rather than smoothed over.

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
├── state.py (data aggregation + response function fitting + State agent)
├── firm.py / worker.py (ABM agents, regime-aware)
├── abm_model.py (Mesa ABM implementation — done; Gini clipping fix applied)
├── experiments.py (historical validation, regime comparison, reform-vs-revolution,
│                    sensitivity sweep, permutation null check — done;
│                    spike-detection threshold fix applied)
├── data/
│   ├── us_state_response_data.csv (raw quarterly output)
│   └── manual_collections/ (wage percentiles, Gini, protest data)
└── results/
    ├── plots/ (calibration fits, time series) [TO DO]
    └── experiments/ (regime comparison outputs, first post-fix run complete)
```

---

**Last updated:** August 2026
**Status:** ABM implemented; first full post-fix experiment suite complete. Protest-calibration sanity check and regime-Gini differentiation are the top open items before drawing paper-level conclusions.