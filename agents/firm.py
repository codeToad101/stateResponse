"""
firm.py

Firm agent for the state-fiscal-response ABM. Rewritten from the original
pseudocode (undefined `threshold_tight`, literal `self.hire(2-3 workers)`
and `self.fire(1-2 workers)` as if English were valid Python, no actual
employee list) into a runnable Mesa agent.

Design targets:
  - Bounded rationality: wage-setting and hiring are heuristic responses
    to observable local signals (labor market tightness, own profit), not
    a solved profit-maximization first-order condition.
  - Heterogeneity: productivity varies firm-to-firm, drawn once, so
    identical macro conditions still produce different firm-level
    outcomes -- not one representative firm scaled up.
  - Sticky wages via partial adjustment: wages move a *fraction* of the
    way toward a tightness-implied target each step (adaptive
    expectations), closer to observed real-world wage rigidity than
    instant market clearing.
  - Genuine LIFO firing: an ordered `employees` list actually popped from
    the end, not prose describing what should happen.
  - Optional mild monopsony power (per the project's original design
    note), and stochastic (Poisson) hire/fire magnitude so otherwise-
    similar firms diverge over time instead of moving in lockstep.
"""

import numpy as np
import mesa

from agents.worker import MEAN_SKILL_PREMIUM


class Firm(mesa.Agent):
    def __init__(self, model, productivity, initial_wage, capital=100.0,
                 monopsony_power=0.0, productivity_scale=1.0, wage_noise_std=0.01,
                 profit_share_to_owner=0.4, owner_retention_rate=0.7,
                 owner_investment_threshold_multiple=1.0,
                 owner_propensity_alpha=5.0, owner_propensity_beta=2.0,
                 owner_idiosyncratic_return_std=0.04,
                 protest_output_penalty=0.5, rng=None):
        super().__init__(model)
        self.random_gen = rng if rng is not None else np.random.default_rng()

        self.productivity = float(np.clip(productivity, 0.0, 1.0))
        self.monopsony_power = float(np.clip(monopsony_power, 0.0, 0.5))
        self.productivity_scale = productivity_scale
        self.capital = capital
        self.wage = initial_wage
        self.wage_noise_std = wage_noise_std
        self.profit = 0.0

        # --- capital-income channel ---
        # Fixed by design decision (see chat notes): NOT one designated
        # "top worker" drawing owner pay -- real capitalists don't
        # generally need deep domain skill in the way a top individual
        # contributor does, so tying owner income to any one worker's
        # skill misrepresents the mechanism. Instead it's a separate
        # income stream, owned by the firm itself, added alongside (not
        # instead of) worker wages when the model computes Gini. Floored
        # at 0: owners don't draw negative income when the firm loses
        # money, but they also don't cover the loss out of pocket here
        # (no owner capital injection mechanism -- out of scope for now).
        self.profit_share_to_owner = float(np.clip(profit_share_to_owner, 0.0, 1.0))
        self.owner_income = 0.0

        # --- owner wealth compounding (capital income, not wage income) ---
        # Real-world inequality is dominated by capital compounding
        # (Piketty's r > g), not wage dispersion -- see Dosi et al. (2020)
        # and the wealth-distribution ABM literature. Owners retain a much
        # larger share of their income than workers (owner_retention_rate
        # vs. Worker.savings_rate) and, critically, are drawn from a
        # HIGHER propensity-to-invest distribution -- Beta(5,2), mean
        # ~0.71, vs. workers' Beta(2,5), mean ~0.29 -- reflecting real
        # structural asset-market access differences between capital
        # holders and wage earners, not a claim that every owner invests
        # every period (still a per-tick probabilistic draw).
        self.owner_retention_rate = float(np.clip(owner_retention_rate, 0.0, 1.0))
        self.owner_investment_threshold = owner_investment_threshold_multiple * capital
        self.owner_propensity_to_invest = float(
            self.random_gen.beta(owner_propensity_alpha, owner_propensity_beta)
        )
        self.owner_idiosyncratic_return_std = owner_idiosyncratic_return_std
        self.owner_wealth = 0.0
        self.owner_investment_income = 0.0  # realized THIS tick only, feeds
        # the income Gini alongside owner_income -- not the wealth stock

        # heuristic target, not derived from an optimization -- bounded
        # rationality: firms aim for "reasonably profitable," not optimal
        self.target_profit = 0.05 * capital

        # Free, uncalibrated parameter -- same posture as Worker's sigma/
        # theta weights: undefended design choice, candidate for a
        # sensitivity sweep, not presented as fitted. 0.5 = a protesting
        # employee contributes half their normal output that tick (partial
        # disruption -- not everyone counted in protest_intensity_score is
        # in a full-day work stoppage); 1.0 would model a full strike,
        # 0.0 would silently contradict the strike-based data source this
        # whole model calibrates against.
        self.protest_output_penalty = float(np.clip(protest_output_penalty, 0.0, 1.0))

        # Productivity-linked pay anchor: derived from the SAME
        # production function used in _compute_output_and_profit
        # (assuming an average-skill employee) rather than a separately
        # invented multiplier -- an earlier version pegged this directly
        # to initial_wage instead, which double-counted productivity
        # against the output formula and pushed average wages above what
        # output could sustain, causing a slow-motion repeat of the
        # earlier bankruptcy-collapse bug. Divided by MEAN_SKILL_PREMIUM
        # (imported from worker.py) for the same reason: since actual
        # paid wage = firm.wage * individual skill_premium, and that
        # premium averages ~1.26 under the occupational-tier scheme (not
        # ~1.0), failing to divide it back out here would systematically
        # run the wage bill ~26% hotter than output supports -- exactly
        # the bug this correction is here to prevent, caught in testing.
        assumed_mean_skill = 0.55  # midpoint of Worker skill ~ Uniform(0.1, 1.0)
        self._productivity_anchor = (
            self.productivity * productivity_scale * assumed_mean_skill
            / MEAN_SKILL_PREMIUM
        )

        self.employees = []  # ordered hire queue -> real LIFO firing

    # ------------------------------------------------------------------
    def _set_wage(self, labor_market_tightness):
        """
        Partial-adjustment wage-setting, now blended with a productivity
        anchor rather than tightness alone: a low-productivity firm in a
        tight labor market still competes up, just from a lower baseline,
        and a high-productivity firm doesn't collapse to the market floor
        just because conditions are slack. This is `self.wage` -- the
        firm's *base* pay level; individual worker pay is this base times
        their own skill_premium, set in _pay_employees below.
        """
        tight_threshold = 1.0  # openings roughly matching unemployed searchers
        if labor_market_tightness > tight_threshold:
            pressure = min(0.05, 0.02 * (labor_market_tightness - tight_threshold))
            tightness_target = self.wage * (1 + pressure)
        else:
            slack_cut = min(0.02, 0.01 * (tight_threshold - labor_market_tightness))
            tightness_target = self.wage * (1 - slack_cut)

        target_wage = 0.7 * tightness_target + 0.3 * self._productivity_anchor
        target_wage *= (1 - self.monopsony_power)

        adjustment_speed = 0.3  # sticky wages: partial move per step
        self.wage += adjustment_speed * (target_wage - self.wage)
        # idiosyncratic noise: otherwise identically-initialized firms
        # facing the same tightness signal move in perfect lockstep
        # forever, and the simulated wage distribution never diverges
        self.wage *= (1 + self.random_gen.normal(0, self.wage_noise_std))
        self.wage = max(self.wage, 0.01)

    def _pay_employees(self):
        """
        Within-firm pay stratification: each employee is paid the firm's
        base wage times their own skill_premium (set once per worker in
        Worker.__init__ -- see agents/worker.py), rather than every
        employee at a given firm earning the identical flat wage. This is
        the "stratify pay within worker class @ certain firms" half of
        the fix; _set_wage's productivity anchor above is the "pay @
        various firms themselves" half.
        """
        for w in self.employees:
            w.wage = self.wage * w.skill_premium

    def _hire_fire(self):
        """
        Profit-based hiring/firing. LIFO layoffs implemented as an actual
        pop() from the end of an ordered list. Magnitude is stochastic
        (Poisson) rather than a fixed "2-3 workers," so firms with
        identical profit histories still don't move in perfect lockstep.
        """
        if self.profit > self.target_profit:
            n_hire = self.random_gen.poisson(2)
            for _ in range(n_hire):
                candidate = self.model.draw_unemployed_worker()
                if candidate is None:
                    break
                candidate.firm = self
                candidate.employment_status = "employed"
                candidate.wage = self.wage * candidate.skill_premium
                self.employees.append(candidate)
        elif self.profit < 0:
            n_fire = min(len(self.employees), self.random_gen.poisson(1) + 1)
            for _ in range(n_fire):
                if not self.employees:
                    break
                worker = self.employees.pop()  # LIFO: last hired, first fired
                worker.firm = None
                worker.employment_status = "unemployed"
                worker.wage = 0.0  # no income while unemployed -- see
                # Worker._update_wealth_and_investment for why this
                # matters (a stale nonzero wage would keep "earning"
                # forever otherwise)

    def _compute_output_and_profit(self):
        """
        Closes a real gap: without this, Firm was blind to whether its own
        workforce was protesting at all -- profit/hiring never reacted to
        it, so the protest -> economic disruption -> profit -> layoffs ->
        further grievance loop implied by the strike-derived historical
        data (and the Epstein-style ABM framing this project is built on)
        simply didn't exist on the Firm side.
        """
        output = sum(
            w.skill * self.productivity * self.productivity_scale *
            (1 - self.protest_output_penalty if w.is_protesting else 1.0)
            for w in self.employees
        )
        wage_costs = sum(w.wage for w in self.employees)
        depreciation = 0.01 * self.capital
        self.profit = output - wage_costs - depreciation
        self.owner_income = max(self.profit, 0.0) * self.profit_share_to_owner

    def _update_owner_wealth_and_investment(self, market_return):
        """
        Mirrors Worker._update_wealth_and_investment, but with the
        owner-side asymmetry described in __init__: higher retention,
        lower threshold relative to their typical income, and a
        structurally higher (but still probabilistic, still risky)
        propensity to actually invest each period.
        """
        self.owner_wealth += self.owner_retention_rate * self.owner_income

        self.owner_investment_income = 0.0
        eligible = self.owner_wealth > self.owner_investment_threshold
        chooses_to_invest = self.random_gen.random() < self.owner_propensity_to_invest
        if eligible and chooses_to_invest:
            realized_return = market_return + self.random_gen.normal(
                0.0, self.owner_idiosyncratic_return_std
            )
            realized_return = max(realized_return, -1.0)
            self.owner_investment_income = self.owner_wealth * realized_return
            self.owner_wealth = max(0.0, self.owner_wealth + self.owner_investment_income)

    # ------------------------------------------------------------------
    def step(self):
        labor_market_tightness = self.model.labor_market_tightness
        self._set_wage(labor_market_tightness)
        self._pay_employees()
        self._hire_fire()
        self._compute_output_and_profit()
        self._update_owner_wealth_and_investment(self.model.market_return)