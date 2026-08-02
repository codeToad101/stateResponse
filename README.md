# State Fiscal Response to Class Conflict: An Agent-Based Model

**Author:** Asiya Kadivar  
**Institution:** Carnegie Mellon University  
**Status:** Work in progress (private)  
**Timeline:** July–October 2026  
**Target Venue:** Journal of Artificial Societies and Social Simulation (JASSS) or heterodox economics outlets

---

## Overview

This project models the **US state's fiscal policy response to labor market conditions and conflict intensity** using agent-based modeling (ABM). The core innovation: instead of treating fiscal policy (taxation, redistribution) as exogenous or optimizing in a rational-expectations framework, we model the state as a *reactive agent* that observes economic indicators (inequality, unemployment, protest) and adjusts redistributive spending accordingly.

**Central research question:** 
Is state fiscal response to class conflict *structural* (persistent policy reorientation) or *cyclical* (temporary pacification)? And what state regime (representative, captured, unresponsive) best explains observed US policy trajectories?

### Theoretical Motivation

Recent ABM and political economy literature points to:
- **Epstein (2002):** Civil violence models show how state repression + grievance interact at the micro level
- **Acemoglu & Robinson (2000):** Redistribution as political equilibrium—elites tolerate it to avoid uprising
- **Piketty et al. (2014, 2015):** Tax policy responds to political pressure and distributional conflict
- **Alesina & Rodrik (1994):** Tradeoff between redistribution and growth not mechanically determined—policy choice matters

**Gap:** No ABM has endogenized state fiscal response to *endogenous* worker grievance + protest dynamics. Most models treat policy as exogenous or assume rational welfare maximization.

### Ideological Stakes

Two interpretations of results:
1. **Reform optimism:** If redistribution is *sticky* (ratchets up, persists), suggests genuine policy gains possible via democratic struggle
2. **Revolutionary critique:** If redistribution is *cyclical* (state reverses gains when conflict subsides), suggests structural oppression requires systemic change, not reform

We present both without endorsing—the empirics speak.

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
  - Tight labor → raise wages; slack → cut wages
  - (Optional: mild monopsony power, wage = productivity × (1 − monopsony_power))
- Hiring/firing follows profit dynamics (LIFO)

**State** (N = 1, endogenous decision-maker)
- **Observes:** Gini(t), unemployment(t), protest intensity(t), wage growth(t)
- **Decides:** tax rate τ(t), redistribution R(t)
- **Response function:** R(t) = f(protest(t−lag), Gini(t), growth(t)) + noise
  - *Not* optimizing welfare; using bounded-rational reaction function
  - Lag (2–4 quarters) captures institutional/political delays
- **Budget constraint:** τ × wages ≥ R
- **Regimes:** Representative (responsive), Captured (biased toward elites), Unresponsive (ignores conflict)

### Market Dynamics

- **Labor clearing:** Wages adjust until supply ≈ demand (with frictions)
- **Feedback loops:** 
  - High unemployment → low wages → high relative grievance → high protest
  - High protest → state increases redistribution → worker consumption → demand → hiring
  - Inequality widens → grievance increases independent of absolute wages

---

## Research Questions & Expected Directions

### Primary Questions

1. **State Responsiveness:** Is US fiscal policy better fit by a *reactive* model (responds to past conflict) or *proactive* model (responds to inequality trajectory)?
   - **Method:** Calibrate response functions using historical data; compare AIC/BIC
   - **Expected outcome:** Significant lag effect suggests reactive; pre-emptive redistribution suggests proactive
   
2. **Regime Comparison:** Do different state regimes (representative vs. captured vs. unresponsive) produce observably different distributional outcomes?
   - **Method:** Run identical model with three state types; compare long-run inequality, protest intensity, wage levels
   - **Expected outcome:** Captured state produces higher inequality + cyclical protest; representative state stabilizes lower inequality

3. **Reform vs. Revolution:** Does redistribution "stick" or get reversed?
   - **Method:** Track redistribution over long runs; measure persistence (autocorrelation, trend reversal)
   - **Expected outcome:** 
     - Sticky redistribution → supports reform optimism
     - Cyclical redistribution → supports revolutionary critique

### Secondary Questions

- Does labor market tightness independently suppress protest (even controlling for inequality)?
- How do skill heterogeneity and network structure affect coordination costs for workers?
- What historical periods does the model *fail* to capture? (Model uncertainty assessment)

---

## Data Sources & Rationale

### Automated FRED Pulls

| Variable | FRED ID | Frequency | Rationale |
|----------|---------|-----------|-----------|
| Unemployment | UNRATE | Monthly → quarterly | Direct measure of labor slack; standard state observable |
| LFPR | CIVPART | Monthly → quarterly | Signals hidden unemployment, discouraged workers |
| Avg hourly earnings | CES0500000003 | Monthly → quarterly | Wage dynamics; base for relative grievance calculation |
| Job openings | JTSJOR | Monthly → quarterly | Labor market tightness proxy; available post-2000 only |
| GDP (nominal) | A191RL1Q225SBEA | Quarterly | Denominator for redistribution %; controls for scale |
| UI benefits | IUPBS | Quarterly | Automatic stabilizer; state redistributes via UI |
| Workers comp | WCOMPIB | Quarterly | Insurance-based redistribution |
| Veterans benefits | VETERANS | Quarterly | Long-standing transfer program |
| Medicaid | A091MD3A027NBEA | Quarterly | Means-tested, politically responsive, large (3–4% GDP) |

### Manual Collection Required

| Variable | Source | Frequency | Rationale |
|----------|--------|-----------|-----------|
| **Gini coefficient** | Census Bureau / World Bank | Annual | Direct measure of inequality; state's primary observable for redistribution decision |
| **Wage percentiles** (P10, P50, P90) | Census CPS / Pew Research | Annual | Builds relative grievance (ratio of own wage to reference) |
| **SNAP benefits** | USDA / Census | Annual | Largest discretionary transfer program; politically sensitive |
| **EITC claims** | IRS / Tax Foundation | Annual | Wage subsidy; state's main anti-poverty tool post-1980s |
| **Federal tax rate** | Tax Foundation | Annual | Policy lever; state input for financing redistribution |
| **Strikes/protests** | BLS work stoppages + CNTSDA + ACLED | Quarterly preferred | Direct measure of conflict intensity; fed into state's response function |

### Why Quarterly?

- Captures short-run dynamics (policy lags, hiring cycles)
- Sufficient for 65-year span (260 quarters of data)
- Annual insufficient for detecting state response timing

### Why Start in 1960?

- Captures post-WWII welfare state maturation
- Includes major inflection points: Great Society (1960s) → stagflation (1970s) → neoliberal turn (1980s)
- Sufficient data coverage for reliable calibration

---

## Mathematical Foundations

### Worker Grievance

$$G_i(t) = \frac{\text{reference\_wage}(t) - w_i(t)}{\text{reference\_wage}(t)}$$

where reference wage = local average wage × (1 − Gini). 

**Rationale:** Grievance is *relative*, not absolute. Rising inequality raises grievance even if own wages rise (relative deprivation).

### Worker Protest Decision

$$\text{Protest}_i(t) = \begin{cases} 1 & \text{if } G_i(t) - R_i(t) > \theta_i \\ 0 & \text{otherwise} \end{cases}$$

where:
- $R_i(t)$ = perceived risk of arrest/retaliation = f(police density, protest visibility)
- $\theta_i$ = idiosyncratic threshold (heterogeneous across workers)

**Source:** Epstein (2002) civil violence model; adapted for labor context.

### State Response Function (To Be Calibrated)

$$R(t) = \beta_0 + \beta_1 \cdot \text{Protest}(t - \tau) + \beta_2 \cdot \text{Gini}(t) + \beta_3 \cdot \text{Growth}(t) + \epsilon(t)$$

where:
- $R(t)$ = redistribution spending (% GDP)
- $\tau$ = policy lag (2–4 quarters)
- $\epsilon(t)$ = error (idiosyncratic shocks, unmeasured factors)

**Fitting approach:** 
1. Extract historical R(t), Protest(t), Gini(t), Growth(t) from data
2. Fit multiple functional forms (linear, polynomial, logistic, exponential)
3. Compare AIC/BIC; select best fit
4. Plug fitted function into ABM

**Alternative formulations tested:**
- Logistic: $R = \frac{R_{\max}}{1 + e^{-k(\text{Gini} - x_0)}}$ (redistribution saturates at high inequality)
- Exponential: $R = R_0 e^{\beta \cdot \text{Gini}}$ (aggressive redistribution at extreme inequality)

---

## Expected Timeline

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Data collection & cleaning | 2–3 | us_state_response_data.csv; manual collection checklist |
| State response fitting | 1–2 | Calibrated response function; AIC/BIC comparison plot |
| ABM implementation | 3–4 | Working Mesa model; baseline run reproduces labor market dynamics |
| Validation | 1 | Historical comparison; does simulated US policy path match reality? |
| Experiments | 1–2 | Regime comparisons (representative vs. captured vs. unresponsive); sensitivity analysis |
| Writing | 1 | Results, limitations, interpretation for reform/revolution debate |

---

## Known Limitations & Caveats

1. **Firm wage-setting:** Mechanical (not strategic). Real firms may coordinate to suppress wages. (Mitigation: sensitivity test with mild monopsony.)

2. **State as single agent:** Real state is contested (Treasury vs. Congress vs. Fed). Simplified here. (Mitigation: different "state regimes" proxy heterogeneity.)

3. **Protest data:** Strike data sparse post-2000s; must manually aggregate from ACLED (noisier). May under/overestimate true conflict.

4. **Causality unclear:** Does redistribution *cause* lower protest, or do high-protest periods *select* into high redistribution? ABM assumes reaction, but history is endogenous. (Honest limitation to note.)

5. **Demand-side omitted:** Model focuses on labor supply + state response. Doesn't model consumption dynamics, credit, asset bubbles. (Scope boundary.)

---

## Ideological Transparency

This project is grounded in heterodox political economy (Marxist-adjacent but not doctrinaire). We test whether capitalism naturally stabilizes via redistribution (reform) or requires systemic change (revolution). 

**We are not neutral on these questions, but the model is.** Results will speak for themselves.

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
├── us_state_response_data.py (data collection)
├── state_response_fitter.py (calibration) [TO DO]
├── abm_model.py (Mesa ABM implementation) [TO DO]
├── experiments.py (run scenarios) [TO DO]
├── data/
│   ├── us_state_response_data.csv (raw output)
│   └── manual_collections/ (wage percentiles, Gini, protest data)
└── results/
    ├── plots/ (calibration fits, time series)
    └── experiments/ (regime comparisons)
```

---

**Last updated:** August 2026  
**Status:** Data collection phase
