"""
worker.py

Worker agent for the state-fiscal-response ABM. Rewritten from the
original pseudocode (undefined self.threshold, string literal
"employed" or "unemployed" as a no-op boolean expression, a
count_active_protesters_in_network() call with no implementation) into a
runnable Mesa agent.

Design targets modern complexity-economics ABM standards (SFI's
agent-based computational economics tradition -- Tesfatsion, Axtell,
Epstein):
  - Bounded rationality: grievance/protest decisions come from local
    heuristics and a slowly-adapting reference point, not a solved
    global utility-maximization problem.
  - Heterogeneity: skill, wage, and protest threshold all vary
    agent-to-agent, drawn from distributions rather than shared.
  - Network-embedded behavior: risk perception depends on a worker's
    *local* neighborhood in a social graph, not a global protest count --
    matches Epstein (2002)'s "visible protesters" mechanism properly
    (the original pseudocode referenced this but never defined it).
  - Probabilistic, not deterministic, decisions: protest participation
    is drawn from a logistic probability of the grievance-risk gap
    rather than a hard threshold cutoff, so individual behavior stays
    only partly predictable even when the population distribution is
    well-specified -- this is the "free choice" behavior requested.
  - Mild adaptive learning: a worker's protest threshold drifts slightly
    based on whether recent protest "paid off" (state responded), rather
    than being fixed for the agent's lifetime.
"""

import numpy as np
import networkx as nx
import mesa
from functools import lru_cache
from scipy.optimize import brentq
from scipy.stats import norm


class EmploymentStatus:
    EMPLOYED = "employed"
    UNEMPLOYED = "unemployed"


def occupational_tier(skill):
    """
    Discretizes continuous skill into three occupational tiers
    (administrator / skilled / normal), each with its own pay-premium
    range, following the skill-biased occupational structure used in
    Dosi et al. (2020, "Wage Inequality, Labor Market Polarization and
    Skill-Biased Technological Change") and the Eurace@Unibi labor
    market (general + specific skill tiers). A single continuous
    premium (the earlier 0.7 + 0.6*skill formula) under-produces top-end
    dispersion relative to real occupational wage gaps; a convex,
    tiered premium is closer to what that literature actually models.

    Tier bands (skill ~ Uniform(0.1, 1.0) by construction, so these
    roughly correspond to bottom 50%, next 35%, top 15%):
      skill <  0.50 : "normal"        premium 0.60 - 1.00
      0.50<=skill<0.85: "skilled"      premium 1.00 - 1.60
      skill >= 0.85 : "administrator" premium 1.60 - 3.00
    """
    if skill < 0.50:
        return "normal", 0.60 + 0.40 * (skill / 0.50)
    elif skill < 0.85:
        frac = (skill - 0.50) / 0.35
        return "skilled", 1.00 + 0.60 * frac
    else:
        frac = (skill - 0.85) / 0.15
        return "administrator", 1.60 + 1.40 * frac


# E[skill_premium] under skill ~ Uniform(0.1, 1.0), computed via fine
# numerical integration (deterministic, not a seed-dependent Monte Carlo
# estimate). The old flat-linear premium (0.7 + 0.6*skill) averaged
# ~1.03, close enough to 1.0 that the output formula could ignore it; the
# tiered premium above averages ~1.26 -- firm.py and abm_model.py divide
# by this constant wherever a firm's break-even wage anchor is derived
# from its production function, so the average PAID wage (after
# multiplying by individual skill_premiums) matches average OUTPUT rather
# than silently running ~26% hot, which is what caused a slow-motion
# repeat of the earlier bankruptcy-collapse bug when this was first added.
MEAN_SKILL_PREMIUM = 1.262223

# ============================================================================
# GLOBAL-GAMES COORDINATION (Morris & Shin 1998, 2003; Edmond 2013)
# ============================================================================
#
# Replaces a hand-set critical-mass cutoff with an endogenously-solved
# switching threshold, so that "does the population coordinate" isn't an
# arbitrary constant someone typed in. Two population-shared objects:
#
#   theta  -- the TRUE regime fundamental this tick (public, but each
#             worker only observes it through a private noisy signal).
#             Built from State's existing gini/growth/redistribution
#             trend -- see compute_theta_fundamental.
#   x*     -- the equilibrium private-signal threshold: a worker whose
#             signal falls below x* has "coordination succeed" this
#             tick. Solved from the regime's payoff structure (see
#             solve_switching_equilibrium), not hand-set.
#
# theta* (the critical TRUE fundamental below which a cascade succeeds)
# and x* are the standard two-equation Morris-Shin fixed point. Payoffs
# here let benefit-of-toppling scale with how bad the regime already is
# (1 - theta*), so unlike the textbook diffuse-prior closed form, theta*
# has no closed form anymore and needs the numerical root-find.

def compute_theta_fundamental(state, gini_ref=0.45, growth_scale=5.0,
                               weight_growth=1.0, weight_gini=1.0,
                               weight_redistribution=0.5):
    """
    theta in (0,1): TRUE regime-strength fundamental for this tick.
    Higher theta = stronger fundamentals = harder for a protest cascade
    to succeed. Built entirely from State's existing observed aggregates
    (gini, growth, redistribution trend) -- no new invented series, same
    pattern as reusing police_intensity/protest_weight for payoffs below.

    Which macro signals matter and how much is a documented modeling
    choice, not a fitted quantity (state.py has nothing to fit a
    theta-fundamental target against). Treat these weights the same way
    sigma is treated in Worker -- worth a sensitivity sweep, not
    presented as precision.

    'redistribution gap' is operationalized as the PROPORTIONAL change
    in redistribution (this tick's delta / last level) rather than an
    absolute gap against some reference, since redistribution has no
    fixed absolute scale in this model. Flagging that interpretation
    explicitly since it's a choice, not the only reasonable one.
    """
    if state.gini is None:
        return 0.5  # no observation yet (pre-first-tick) -- neutral prior

    growth = state.growth if state.growth is not None else 0.0
    growth_term = weight_growth * (growth / growth_scale)

    gini_term = weight_gini * (state.gini - gini_ref)

    prev_redist = getattr(state, '_last_redistribution', 0.0)
    redist_delta = getattr(state, '_last_redistribution_delta', 0.0)
    if abs(prev_redist) > 1e-9:
        redist_term = weight_redistribution * (redist_delta / abs(prev_redist))
    else:
        redist_term = 0.0

    raw = growth_term - gini_term + redist_term
    return float(1.0 / (1.0 + np.exp(-raw)))


def _theta_star_residual(theta_star, cost, benefit0, benefit_scale):
    """
    Fixed-point residual for the critical TRUE fundamental theta*:
        Phi(-delta(theta*)) - theta*  ==  0
    where delta(theta*) = Phi^-1( cost / (benefit(theta*) + cost) ) is the
    indifference condition for the marginal agent, and
        benefit(theta*) = benefit0 * (1 + benefit_scale*(1 - theta*))
    lets the payoff for successfully toppling scale with how weak the
    regime already is. That theta*-dependence is what breaks the classic
    Morris-Shin closed form and requires brentq below rather than
    algebra.
    """
    theta_star = float(np.clip(theta_star, 1e-9, 1 - 1e-9))
    benefit = benefit0 * (1.0 + benefit_scale * (1.0 - theta_star))
    ratio = float(np.clip(cost / (benefit + cost), 1e-9, 1 - 1e-9))
    delta = norm.ppf(ratio)
    return norm.cdf(-delta) - theta_star


def _delta_at(theta_star, cost, benefit0, benefit_scale):
    theta_star = float(np.clip(theta_star, 1e-9, 1 - 1e-9))
    benefit = benefit0 * (1.0 + benefit_scale * (1.0 - theta_star))
    ratio = float(np.clip(cost / (benefit + cost), 1e-9, 1 - 1e-9))
    return norm.ppf(ratio)

def sweep_coordination_sensitivity(state, n_workers=500, sigma_values=(0.03, 0.07, 0.15, 0.30),
                                     weight_sets=None, seed=0):
    """
    Quick diagnostic sweep, NOT a calibration -- there's no historical
    target to fit theta weights or sigma against (same caveat as
    compute_theta_fundamental's docstring). Reports, for a fixed state
    snapshot, how sensitive coordination_success and the willing-but-
    falsified gap are to sigma and to the theta weight set, so you can
    eyeball whether the AND-gate is workably permissive before running
    a full abm_model experiment.
    """
    if weight_sets is None:
        weight_sets = {
            'baseline': dict(weight_growth=1.0, weight_gini=1.0, weight_redistribution=0.5),
            'gini_dominant': dict(weight_growth=0.3, weight_gini=2.0, weight_redistribution=0.3),
            'redistribution_dominant': dict(weight_growth=0.5, weight_gini=0.5, weight_redistribution=2.0),
        }
    rng = np.random.default_rng(seed)
    results = []
    for weight_name, weights in weight_sets.items():
        theta = compute_theta_fundamental(state, **weights)
        for sigma in sigma_values:
            cost = 1.0 * state.police_intensity
            theta_star, x_star = solve_switching_equilibrium(cost, 1.0, 1.0, sigma)
            signals = theta + rng.normal(0.0, sigma, size=n_workers)
            coord_success_rate = float(np.mean(signals < x_star))
            results.append(dict(weights=weight_name, sigma=sigma, theta=theta,
                                 theta_star=theta_star, x_star=x_star,
                                 coordination_success_rate=coord_success_rate))
    return results


@lru_cache(maxsize=512)
def solve_switching_equilibrium(cost, benefit0, benefit_scale, sigma):
    """
    Solves the Morris-Shin fixed point (theta*, x*) via a 1-D root-find
    (scipy.optimize.brentq) against the equilibrium indifference
    condition, per Morris & Shin (1998, 2003); payoff structure follows
    Edmond (2013)'s regime-change framing.

    theta*: critical TRUE fundamental -- a cascade succeeds iff the
        tick's true theta < theta*.
    x*: the switching threshold each worker compares their PRIVATE
        signal against (signal < x* -> coordination succeeds).

    Note theta* does not depend on sigma -- in this diffuse-prior/normal-
    noise setup that's a real, known property of the model (the
    "risk-dominant" threshold is pinned by the payoff structure alone),
    not an oversight. x* DOES depend on sigma (x* = theta* - sigma*delta),
    which is where the private-signal noise actually shows up.

    cost:            regime-scaled cost of protesting if the regime
                     survives (coordination_cost0 * state.police_intensity
                     -- reuses the existing regime-differentiated dial
                     rather than inventing a new "revolution cost").
    benefit0/scale:  payoff structure for a successful cascade.
    sigma:           private-signal noise std (Worker.signal_noise_std)
                     -- included only because x* needs it; cached
                     lookups are exact since cost/benefit/sigma are the
                     same float values for every worker within a tick
                     (all read off the one shared State/regime), so this
                     avoids O(N) redundant root-finds per tick.

    Raises via brentq's ValueError if the residual doesn't change sign
    across (0,1) (e.g. a pathological cost/benefit ratio) -- caller
    should treat that as a configuration problem to fix, not silently
    clamp, so it's left unguarded here deliberately.
    """
    theta_star = brentq(_theta_star_residual, 1e-6, 1 - 1e-6,
                         args=(cost, benefit0, benefit_scale))
    delta = _delta_at(theta_star, cost, benefit0, benefit_scale)
    x_star = theta_star - sigma * delta
    return theta_star, x_star

class Worker(mesa.Agent):
    def __init__(self, model, skill, initial_wage,
                 threshold_mean=0.62, threshold_std=0.1,
                 gini_sensitivity=0.0, redistribution_sensitivity=0.0,
                 consumption_floor=None,
                 savings_rate=0.30, investment_threshold_multiple=5.0,
                 propensity_to_invest_alpha=2.0, propensity_to_invest_beta=5.0,
                 idiosyncratic_return_std=0.03,
                 signal_noise_std=0.07,
                 coordination_benefit0=1.0, coordination_benefit_scale=1.0,
                 coordination_cost0=1.0,
                 rng=None):
        super().__init__(model)
        self.random_gen = rng if rng is not None else np.random.default_rng()

        # --- heterogeneous traits, drawn once per agent ---
        self.skill = float(np.clip(skill, 0.0, 1.0))
        # Occupational-tier pay premium (see occupational_tier() above) --
        # replaces the earlier flat-linear premium with a convex,
        # tiered one so top-skill workers pull meaningfully further
        # ahead, matching the skill-biased literature this is drawn from.
        self.occupation, self.skill_premium = occupational_tier(self.skill)
        # idiosyncratic protest threshold theta_i (Epstein 2002-style)
        self.threshold = max(0.01, self.random_gen.normal(threshold_mean, threshold_std))

        # --- data-calibrated macro pathway (see calibrate_population_from_bvar
        # below) -- deliberately small by default; see that function's
        # docstring for why these shouldn't be hand-picked to "look" strong
        self.gini_sensitivity = gini_sensitivity
        self.redistribution_sensitivity = redistribution_sensitivity
        self._last_gini = None

        # --- capital-income channel (worker side) ---
        # Real-world inequality is dominated by capital compounding, not
        # wage dispersion (Piketty's r > g) -- wage stratification alone
        # (even tiered, above) structurally undershoots real Gini. This
        # gives *some* workers a path to accumulate savings and, if they
        # cross a threshold AND are behaviorally inclined to (see
        # propensity_to_invest below), earn a stochastic return on it --
        # not every well-paid worker invests, and investing is risky, not
        # a guaranteed enrichment channel. See _update_wealth_and_investment.
        self.consumption_floor = (
            consumption_floor if consumption_floor is not None else 0.5 * initial_wage
        )
        self.savings_rate = savings_rate
        self.investment_threshold = investment_threshold_multiple * self.consumption_floor
        # Beta(2,5) (mean ~0.29, right-skewed toward low values): most
        # workers, even wealthy ones, are NOT active investors -- matches
        # the request that "some wealthy workers won't necessarily be
        # buying real estate" -- while a smaller tail is genuinely
        # active/willing to invest whenever they clear the threshold.
        # This is a deliberately heterogeneous, per-agent trait, not a
        # single population-wide investment rate.
        self.propensity_to_invest = float(
            self.random_gen.beta(propensity_to_invest_alpha, propensity_to_invest_beta)
        )
        self.idiosyncratic_return_std = idiosyncratic_return_std
        # --- global-games coordination (see solve_switching_equilibrium) ---
        # sigma: the one genuinely free, uncalibratable knob -- treat as
        # a documented sensitivity-analysis parameter (run across a
        # range), not fitted precision, same posture as the VAR's null
        # gini/redistribution result.
        self.signal_noise_std = signal_noise_std
        self.coordination_benefit0 = coordination_benefit0
        self.coordination_benefit_scale = coordination_benefit_scale
        self.coordination_cost0 = coordination_cost0
        self.would_protest_uninhibited = False  # latent willingness (Kuran)
        self.private_signal = None
        self.theta_star = None
        self.x_star = None
        self.preference_falsification_gap = 0.0  # willing but not coordinated
        self.wealth = 0.0
        self.investment_income = 0.0  # realized THIS tick only -- what
        # feeds the income Gini, never the wealth stock itself (an income
        # Gini and a wealth Gini are different quantities; conflating them
        # would make the "0.35-0.42" historical target incomparable)

        # --- mutable state, carried across steps (path dependence) ---
        self.wage = initial_wage
        self.transfer_income = 0.0  # state redistribution -- kept separate
        # from self.wage because Firm._pay_employees fully OVERWRITES wage
        # every tick (w.wage = firm.wage * skill_premium); folding transfers
        # into wage silently erased them before they could ever be measured.
        self.employment_status = EmploymentStatus.EMPLOYED
        self.grievance = 0.0
        self.risk_perception = 0.0
        self.is_protesting = False
        self.reference_wage_ema = initial_wage  # adaptive reference point
        self.firm = None  # set by Firm on hire/fire

    # ------------------------------------------------------------------
    def _update_grievance(self, state):
        """
        Peer-comparison reference wage (Clark & Oswald 1996; Card et al. 2012):
        workers compare themselves to others in their own occupational tier,
        not the whole population. Inequality's effect on grievance is handled
        separately via the calibrated gini_sensitivity pathway below, so it's
        no longer double-counted by also discounting the reference anchor.
        """
        tier_mean = getattr(self.model, "tier_avg_wage", {}).get(
            self.occupation, state.avg_wage
        )
        ema_alpha = 0.15
        self.reference_wage_ema = (
            ema_alpha * tier_mean + (1 - ema_alpha) * self.reference_wage_ema
        )
        if self.employment_status == EmploymentStatus.UNEMPLOYED:
            self.grievance = 1.0
        else:
            self.grievance = max(
                0.0, (self.reference_wage_ema - self.wage) / self.reference_wage_ema
            )

    def _apply_macro_trend_pathway(self, state):
        """
        Direct macro-to-grievance pathway, calibrated from the fitted
        Bayesian VAR (state.py) rather than assumed. This is deliberately
        the *second* grievance channel, not the main one: the wage-gap
        mechanism in _update_grievance above stays dominant, because the
        VAR's Granger-style checks found no credible Gini->Protest or
        Redistribution->Protest relationship at the annual, aggregate
        level (every relevant 95% credible interval straddled zero).

        The honest calibration choice given a null result is NOT to skip
        this pathway entirely (that would hide the finding) and NOT to
        hand-pick a strong coefficient (that would misrepresent it).
        Instead gini_sensitivity/redistribution_sensitivity are set from
        the VAR's small posterior point estimates (see
        calibrate_population_from_bvar), so the agent population reflects
        the same weak, uncertain aggregate signal the data actually shows.
        """
        if self._last_gini is not None:
            d_gini = state.gini - self._last_gini
            self.grievance += self.gini_sensitivity * d_gini
        self._last_gini = state.gini

        d_redist = getattr(self.model, 'last_redistribution_delta', 0.0)
        self.grievance += self.redistribution_sensitivity * d_redist

        self.grievance = float(np.clip(self.grievance, 0.0, 1.5))

    def _update_risk_perception(self, state):
        """
        Risk depends on *local* network visibility of protest -- a worker
        surrounded by active protesters perceives lower marginal risk
        (safety in numbers) than an isolated one at the same state
        repression level. Implements what the original pseudocode called
        but never defined (count_active_protesters_in_network()).
        """
        neighbors = list(self.model.worker_network.neighbors(self.unique_id))
        if neighbors:
            visible = sum(
                1 for n in neighbors if self.model.workers_by_id[n].is_protesting
            )
            local_visibility = visible / len(neighbors)
        else:
            local_visibility = 0.0

        # state.police_intensity is regime-dependent (see state.py) --
        # dictatorships/captured states raise this, damping protest
        # independent of how aggrieved workers actually are
        self.risk_perception = state.police_intensity * (1 - local_visibility)

    def repression_binding(self):
        """
        Diagnostic (not used in decide_protest itself): True if
        risk_perception has driven the protest-probability gap deeply
        negative regardless of grievance -- i.e. repression alone is
        enough to suppress protest before grievance/threshold can matter.
        Used to test whether the protest<->redistribution null partly
        reflects protest being structurally capped upstream (Kuran-style
        preference falsification: repression suppresses the visible
        SIGNAL, not just the state's response to it) rather than only
        reflecting a genuinely weak redistribution response.
        """
        gap_if_max_grievance = (1.5 - self.risk_perception) - self.threshold
        return gap_if_max_grievance < 0

    def _decide_protest(self, state, theta):
        """
        Two-stage decision, replacing the old single-stage logistic
        cutoff (kept as stage 1, unchanged math) with an added global-
        games coordination gate (stage 2):

          1. Latent willingness (Epstein-style logistic gap, identical
             to the old _decide_protest) -- would this worker protest if
             coordination weren't a separate consideration at all.
          2. Coordination (Morris-Shin / Edmond) -- the worker only ACTS
             on that willingness if their private noisy signal of the
             true regime fundamental theta falls below the population's
             endogenous switching threshold x* (see
             solve_switching_equilibrium). x* is solved from payoffs
             each tick, not hand-set -- this is what replaces the old
             ad hoc fixed critical-mass cutoff.

        is_protesting is the AND of both stages. A worker who is willing
        (stage 1) but whose signal doesn't clear coordination (stage 2)
        is Kuran's preference falsification: privately aggrieved,
        outwardly quiescent -- logged in preference_falsification_gap
        rather than silently discarded, so a run can show that gap
        spiking right before a real cascade.
        """
        gap = (self.grievance - self.risk_perception) - self.threshold
        steepness = 8.0  # higher = closer to a hard threshold; kept moderate
        p_protest = 1.0 / (1.0 + np.exp(-steepness * gap))
        self.would_protest_uninhibited = bool(self.random_gen.random() < p_protest)

        cost = self.coordination_cost0 * state.police_intensity
        theta_star, x_star = solve_switching_equilibrium(
            cost, self.coordination_benefit0, self.coordination_benefit_scale,
            self.signal_noise_std,
        )
        self.theta_star = theta_star
        self.x_star = x_star

        self.private_signal = theta + self.random_gen.normal(0.0, self.signal_noise_std)
        coordination_success = self.private_signal < x_star

        self.is_protesting = self.would_protest_uninhibited and coordination_success #too restricted ???
        self.preference_falsification_gap = float(
            self.would_protest_uninhibited and not coordination_success
        )

    def _adapt_threshold(self):
        """
        Mild reinforcement: a worker who protested and the state recently
        increased redistribution becomes slightly more willing to protest
        again; protest with no payoff raises the threshold back
        (fatigue/cost). Small, bounded step -- adaptive drift, not an
        optimizer, and never lets threshold leave (0.01, 0.99).
        """
        if self.is_protesting:
            payoff = getattr(self.model, "last_redistribution_delta", 0.0)
            step = 0.01
            if payoff > 0:
                self.threshold = max(0.01, self.threshold - step)
            else:
                self.threshold = min(0.99, self.threshold + step)

    def _update_wealth_and_investment(self, market_return):
        """
        Savings -> (maybe) investment -> stochastic return, in that order.

        1. Only surplus above consumption_floor gets saved (subsistence
           spending isn't optional) -- so low earners never accumulate
           wealth to invest in the first place, matching real liquidity
           constraints rather than assuming everyone can participate.
        2. Even above the investment_threshold, whether an agent invests
           THIS tick is a per-agent probabilistic draw against their own
           propensity_to_invest -- not everyone who *can* invest *does*,
           every period. This is deliberate: it's the mechanism behind
           "some wealthy workers won't necessarily be buying real estate."
        3. Returns combine a shared macro market_return (the whole
           economy's investment climate that tick) with idiosyncratic
           noise (different assets, different timing/luck) -- so two
           equally wealthy, equally willing investors can still see very
           different outcomes, and a bad draw can genuinely lose money
           (clipped at -100%, no leveraged losses below the invested
           amount).
        """
        # transfer_income runs through the SAME consumption_floor logic as
        # wage -- only the portion of (wage + transfer) above subsistence
        # is save-able, rather than treating transfers as either always-
        # consumed or always-save-eligible by fiat. This means a small
        # routine transfer to an already-above-floor worker adds directly
        # to savable surplus, while a transfer that itself is what pushes
        # someone above floor is partially captured too -- both realistic,
        # neither hardcoded. Unemployed guard (below) still zeroes wage-
        # side surplus, but NOT transfer_income -- a laid-off worker who
        # is a redistribution recipient can still save from that transfer,
        # which is the actual point of a means-tested transfer existing.
        total_income = self.wage + self.transfer_income
        if self.employment_status == EmploymentStatus.UNEMPLOYED:
            # Guards independently of Firm zeroing wage on layoff (see
            # Firm._hire_fire) -- an unemployed worker never accrues new
            # savings from a stale pre-layoff wage figure. Without this,
            # a fired worker's frozen last wage kept "earning" every tick
            # forever, compounding into unbounded wealth over a long run
            # -- a real bug caught during testing, not a hypothetical one.
            total_income = self.transfer_income
        surplus = max(0.0, total_income - self.consumption_floor)
        self.wealth += self.savings_rate * surplus

        self.investment_income = 0.0
        eligible = self.wealth > self.investment_threshold
        chooses_to_invest = self.random_gen.random() < self.propensity_to_invest
        if eligible and chooses_to_invest:
            realized_return = market_return + self.random_gen.normal(
                0.0, self.idiosyncratic_return_std
            )
            realized_return = max(realized_return, -1.0)  # can't lose more than 100%
            self.investment_income = self.wealth * realized_return
            self.wealth = max(0.0, self.wealth + self.investment_income)

    # ------------------------------------------------------------------
    def step(self):
        state = self.model.state
        self._update_grievance(state)
        self._apply_macro_trend_pathway(state)
        self._update_risk_perception(state)
        theta = compute_theta_fundamental(state)
        self._decide_protest(state, theta)
        self._adapt_threshold()
        self._update_wealth_and_investment(self.model.market_return)
        # wage/employment mutations happen in Firm.step(), not here --
        # keeps labor-demand-side logic owned by one agent type so two
        # agents never race on the same worker's wage in a single tick


def calibrate_population_from_bvar(bvar, threshold_mean_base=0.62, threshold_std_base=0.08):
    """
    Derive population-level Worker parameters from the fitted Bayesian
    VAR (state.py's BayesianVAR) instead of hand-picking constants.

    Why not just set gini_sensitivity/redistribution_sensitivity to 0
    given the null Granger result, or to some larger "plausible" number:
      - Zero would silently discard the VAR's actual point estimate and
        pretend we have no information at all, when we do (a small,
        uncertain one).
      - A larger hand-picked number would misrepresent a null result as
        a real effect just to make the ABM's protest dynamics "more
        interesting" -- exactly the kind of shortcut this project has
        been trying to avoid.
      - Using the VAR's own posterior mean keeps the agent population
        consistent with what was actually found: a small, not-clearly-
        nonzero direct macro pathway, sitting underneath the dominant
        wage-gap/network-based protest mechanism.

    threshold_std is widened in proportion to the VAR's posterior
    uncertainty on those coefficients -- since the macro data can't pin
    a population-level sensitivity down precisely, that uncertainty is
    expressed as individual heterogeneity in threshold rather than a
    falsely precise single number. This is an approximation (a fully
    hierarchical treatment would resample thresholds per model run
    rather than bake uncertainty into a fixed std) -- noted here rather
    than presented as more rigorous than it is.
    """
    names = bvar.coef_names()
    gini_row = names.index('d_gini_coefficient(t-1)')
    redist_row = names.index('d_redistribution_pct_gdp(t-1)')

    protest_coefs = bvar.coefs_['d_protest_intensity_score']
    protest_cov = bvar.coef_cov_['d_protest_intensity_score']

    gini_mean = protest_coefs[gini_row]
    gini_se = np.sqrt(protest_cov[gini_row, gini_row])
    redist_mean = protest_coefs[redist_row]
    redist_se = np.sqrt(protest_cov[redist_row, redist_row])

    threshold_std = threshold_std_base + 0.3 * (gini_se + redist_se) / 2

    return dict(
        threshold_mean=threshold_mean_base,
        threshold_std=threshold_std,
        gini_sensitivity=gini_mean,
        redistribution_sensitivity=redist_mean,
    )


def build_worker_network(worker_ids, k=6, rewire_prob=0.1, seed=None):
    """
    Small-world (Watts-Strogatz) social network over worker unique_ids.
    Small-world topology is the standard choice in Epstein-style civil
    violence / protest-diffusion ABMs: mostly-local clustering (workers
    mainly see nearby peers) with a few long-range rewired edges (protest
    visibility/information can still jump across otherwise-distant parts
    of the population), rather than either a fully local lattice or a
    fully random graph.
    """
    n = len(worker_ids)
    k = min(k, n - 1) if n > 1 else 0
    if k < 2:
        g = nx.Graph()
        g.add_nodes_from(worker_ids)
        return g
    g = nx.watts_strogatz_graph(n, k, rewire_prob, seed=seed)
    return nx.relabel_nodes(g, {i: worker_ids[i] for i in range(n)})