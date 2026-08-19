"""
abm_model.py

Top-level Mesa Model tying together the State/Worker/Firm agents in
agents/. Lives outside the agents/ package since it's the orchestration
layer, not an agent itself.

Multi-country: state.py now fits a separate response function per
country (fit_all_countries). This module caches those fits at module
level (fitting is expensive -- curve_fit + BVAR per country) and builds
a StateResponseModel for a given country's fit, keyed off whatever
feature_names/domain_bounds that country's calibration actually used.

Two run modes, both wired in from the start (not bolted on later):
  - 'free'       : Gini and growth evolve endogenously from the simulated
                    labor market (wage distribution -> Gini; mean firm
                    profit -> growth proxy).
  - 'historical' : Gini and growth are pulled from that country's real
                    annual data each step instead of computed
                    endogenously, so simulated redistribution can be
                    checked against the actual historical series.

Per-tick order matters and is fixed deliberately:
  1. Firms step first (labor market clears: wages, hiring/firing, profit)
  2. This period's aggregate observables are computed/pulled
  3. State observes -> decides -> redistributes
  4. Workers step, reacting to *this* tick's just-updated state
  5. DataCollector records the tick
"""

import numpy as np
import mesa

from agents.state import (
    StateResponseFitter, State, Regime, BayesianVAR,
    load_and_prepare_panel, fit_all_countries, prepare_var_data,
    build_model_inputs, select_predictors_for_country,
)
from agents.worker import Worker, build_worker_network, calibrate_population_from_bvar
from agents.firm import Firm


# ============================================================================
# MODULE-LEVEL FIT CACHE — fit_all_countries() is expensive (curve_fit +
# BVAR per country); every regime/seed/experiment within a process reuses
# the SAME fit per country rather than each silently re-fitting (and
# potentially drifting, since curve_fit isn't perfectly deterministic
# across calls with different starting states). Keyed by (csv_path,
# protest_lag_years) so a different panel/lag choice doesn't collide.
# ============================================================================

_FIT_CACHE = {}


def get_country_fits(csv_path="results/combined_long_panel.csv", protest_lag_years=1,
                      extrapolation_safe_only=True, force_refit=False):
    """
    Cached wrapper around fit_all_countries(). Returns the same dict
    fit_all_countries returns: {country: {'fitter', 'best', 'results',
    'predictors', 'n'}}. Also builds and caches each country's BVAR +
    worker_calibration alongside, since those are needed by every model
    build and are similarly expensive.
    """
    key = (csv_path, protest_lag_years, extrapolation_safe_only)
    if not force_refit and key in _FIT_CACHE:
        return _FIT_CACHE[key]

    country_results = fit_all_countries(
        csv_path, protest_lag_years=protest_lag_years,
        extrapolation_safe_only=extrapolation_safe_only,
    )
    by_country = load_and_prepare_panel(csv_path, protest_lag_years=protest_lag_years)

    out = {}
    for country, res in country_results.items():
        best_name = res["best"]
        best = res["results"][best_name]
        if best["type"] != "parametric":
            # GAM winner: not usable as State.response_fn (different
            # predict() signature) -- flag and skip rather than silently
            # coercing; caller can inspect res["results"] for the GAM if
            # they want it for diagnostics only.
            print(f"[get_country_fits] {country}: best fit '{best_name}' is a GAM, "
                  f"skipping for ABM wiring (GAM has no State-compatible predict()).")
            continue

        fitter = res["fitter"]
        response_fn = getattr(fitter, f"{best_name}_model")
        response_params = best["params"]
        residuals = fitter.y - best["y_pred"]

        annual = by_country[country]
        var_data = prepare_var_data(annual)
        try:
            bvar = BayesianVAR(var_data, lag=1).fit()
            worker_calibration = calibrate_population_from_bvar(bvar)
        except Exception as e:
            print(f"[get_country_fits] {country}: BVAR/worker calibration failed "
                  f"({e}); falling back to Worker defaults for this country.")
            bvar = None
            worker_calibration = {}

        gini_col = annual["gini_coefficient"].dropna() if "gini_coefficient" in annual else None
        initial_gini = float(gini_col.iloc[0]) if gini_col is not None and len(gini_col) else 0.4

        out[country] = dict(
            annual=annual, fitter=fitter, best_name=best_name, bvar=bvar,
            response_fn=response_fn, response_params=response_params,
            residuals=residuals, worker_calibration=worker_calibration,
            initial_gini=initial_gini, domain_bounds=fitter.domain_bounds_,
            feature_names=fitter.feature_names, predictors=res["predictors"],
            n=res["n"],
        )

    _FIT_CACHE[key] = out
    return out


class StateResponseModel(mesa.Model):
    def __init__(self, country, fit=None, n_workers=1000, n_firms=100,
                 regime=Regime.REPRESENTATIVE, worker_calibration=None,
                 monopsony_power=0.0, mode='free',
                 market_return_mean=0.02, market_return_std=0.08, seed=None,
                 redistribution_enabled=True, csv_path="results/combined_long_panel.csv"):
        super().__init__(seed=seed)
        self.random_gen = np.random.default_rng(seed)
        self.country = country

        if fit is None:
            fits = get_country_fits(csv_path)
            if country not in fits:
                raise ValueError(
                    f"No usable fit for country '{country}' (either not present in "
                    f"the panel, didn't clear state.py's data floors, or its best "
                    f"model was a GAM). Available: {sorted(fits)}"
                )
            fit = fits[country]
        self.fit = fit

        if mode not in ('free', 'historical'):
            raise ValueError("mode must be 'free' or 'historical'")

        self.mode = mode
        self.historical_data = fit["annual"] if mode == "historical" else None
        self._historical_row_index = 0
        self.last_redistribution_delta = 0.0
        self.gini = fit["initial_gini"]
        self.market_return_mean = market_return_mean
        self.market_return_std = market_return_std
        self.market_return = 0.0

        worker_calibration = dict(worker_calibration if worker_calibration is not None
                                   else fit["worker_calibration"])
        initial_avg_wage = 20.0

        # --- firms first: labor demand side ---
        productivity_scale = initial_avg_wage / 0.4
        self.firms = [
            Firm(self, productivity=self.random_gen.uniform(0.4, 1.0),
                 initial_wage=initial_avg_wage, monopsony_power=monopsony_power,
                 productivity_scale=productivity_scale, rng=self.random_gen)
            for _ in range(n_firms)
        ]

        # --- workers ---
        self.workers = [
            Worker(self, skill=self.random_gen.uniform(0.1, 1.0),
                   initial_wage=initial_avg_wage, rng=self.random_gen,
                   **worker_calibration)
            for _ in range(n_workers)
        ]
        self.workers_by_id = {w.unique_id: w for w in self.workers}
        self.worker_network = build_worker_network(
            list(self.workers_by_id.keys()), k=6, seed=seed
        )

        for i, w in enumerate(self.workers):
            f = self.firms[i % n_firms]
            f.employees.append(w)
            w.firm = f

        # --- state: built per-country, with THAT country's feature_names
        # and domain_bounds (4 or 5 predictors depending on political-
        # violence coverage -- see state.py's select_predictors_for_country) ---
        self.state = State(
            self, fit["response_fn"], fit["response_params"],
            feature_names=fit["feature_names"], regime=regime,
            residuals=fit["residuals"], domain_bounds=fit["domain_bounds"],
            rng=self.random_gen, redistribution_enabled=redistribution_enabled,
        )
        self._uses_political_violence = "d_political_violence_lag" in fit["feature_names"]

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "country": lambda m: m.country,
                "gini": lambda m: m.gini,
                "avg_wage": lambda m: float(np.mean([w.wage for w in m.workers])),
                "unemployment_rate": lambda m: float(np.mean(
                    [w.employment_status == "unemployed" for w in m.workers])),
                "protest_share": lambda m: float(np.mean(
                    [w.is_protesting for w in m.workers])),
                "mean_grievance": lambda m: float(np.mean([w.grievance for w in m.workers])),
                "repression_bound_share": lambda m: float(np.mean(
                    [w.repression_binding() for w in m.workers])),
                "redistribution": lambda m: m.state.redistribution,
                "total_firm_profit": lambda m: float(sum(f.profit for f in m.firms)),
                "regime": lambda m: m.state.regime,
                "gini_recipients": lambda m: m._last_gini_recipients,
                "gini_non_recipients": lambda m: m._last_gini_non_recipients,
                "mean_transfer_per_recipient": lambda m: m._last_mean_transfer_recipients,
                "n_recipients": lambda m: m._last_n_recipients,
                "wage_var_share": lambda m: m._last_wage_var_share,
                "transfer_var_share": lambda m: m._last_transfer_var_share,
                "owner_var_share": lambda m: m._last_owner_var_share,
                # --- revolt / global-games coordination instrumentation ---
                # theta_star/x_star are identical across workers within a
                # tick (shared payoff structure, cached in
                # solve_switching_equilibrium) -- report the first worker's
                # rather than averaging, to avoid implying they vary.
                "theta_star": lambda m: (m.workers[0].theta_star
                                          if m.workers and m.workers[0].theta_star is not None
                                          else np.nan),
                "x_star": lambda m: (m.workers[0].x_star
                                      if m.workers and m.workers[0].x_star is not None
                                      else np.nan),
                # fraction of workers whose private signal cleared
                # coordination this tick (stage 2 of _decide_protest),
                # distinct from protest_share (which is the AND of both
                # stages)
                "coordination_success_rate": lambda m: float(np.mean(
                    [w.private_signal is not None and w.private_signal < w.x_star
                     for w in m.workers])) if m.workers and m.workers[0].x_star is not None else np.nan,
                # Kuran preference falsification: willing but not
                # coordinated -- the gap between latent grievance and
                # observed protest
                "mean_falsification_gap": lambda m: float(np.mean(
                    [w.preference_falsification_gap for w in m.workers])),
                "mean_would_protest_uninhibited": lambda m: float(np.mean(
                    [w.would_protest_uninhibited for w in m.workers])),
            }
        )

    # ------------------------------------------------------------------
    @property
    def labor_market_tightness(self):
        unemployed = sum(1 for w in self.workers if w.employment_status == "unemployed")
        openings = sum(1 for f in self.firms if f.profit > f.target_profit)
        return openings / max(unemployed, 1)

    def draw_unemployed_worker(self):
        pool = [w for w in self.workers if w.employment_status == "unemployed"]
        return self.random_gen.choice(pool) if len(pool) else None

    @staticmethod
    def _income_variance_shares(wages, transfers, owner_incomes):
        wages = np.asarray(wages, dtype=float)
        transfers = np.asarray(transfers, dtype=float)
        owner = np.asarray(owner_incomes, dtype=float)
        n = max(len(wages), len(transfers), len(owner))
        wages = np.pad(wages, (0, n - len(wages)))
        transfers = np.pad(transfers, (0, n - len(transfers)))
        owner = np.pad(owner, (0, n - len(owner)))
        total = wages + transfers + owner
        total_var = np.var(total)
        if total_var <= 0:
            return dict(wage_share=np.nan, transfer_share=np.nan, owner_share=np.nan)
        return dict(
            wage_share=float(np.var(wages) / total_var),
            transfer_share=float(np.var(transfers) / total_var),
            owner_share=float(np.var(owner) / total_var),
        )

    @staticmethod
    def _compute_gini(wages):
        x = np.sort(np.clip(np.asarray(wages, dtype=float), 0.0, None))
        n = len(x)
        if n == 0 or x.sum() <= 0:
            return 0.0
        cum = np.cumsum(x)
        gini = float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
        assert -1e-9 <= gini <= 1.0 + 1e-9, f"Gini out of bounds: {gini}"
        return float(np.clip(gini, 0.0, 1.0))

    def _pull_historical_row(self):
        idx = min(self._historical_row_index, len(self.historical_data) - 1)
        row = self.historical_data.iloc[idx]
        self._historical_row_index += 1
        return row

    # ------------------------------------------------------------------
    def step(self):
        self.market_return = float(
            self.random_gen.normal(self.market_return_mean, self.market_return_std)
        )

        self._last_gini_recipients = np.nan
        self._last_gini_non_recipients = np.nan
        self._last_mean_transfer_recipients = np.nan
        self._last_n_recipients = 0
        self._last_wage_var_share = np.nan
        self._last_transfer_var_share = np.nan
        self._last_owner_var_share = np.nan

        # 1. firms clear the labor market first
        for f in self.firms:
            f.step()

        # 2. this period's aggregate observables
        political_violence = None
        if self.mode == "historical":
            row = self._pull_historical_row()
            row_gini = row.get("gini_coefficient", np.nan)
            row_growth = row.get("gdp_growth_yoy_pct", np.nan)
            self.gini = float(row_gini) if not (row_gini is None or np.isnan(row_gini)) else self.gini
            growth = float(row_growth) if not (row_growth is None or np.isnan(row_growth)) else 0.0
            if self._uses_political_violence:
                row_pv = row.get("political_violence_score", np.nan)
                political_violence = (float(row_pv) if not (row_pv is None or np.isnan(row_pv))
                                       else None)
        else:
            incomes = (
                [w.wage + w.investment_income + w.transfer_income for w in self.workers]
                + [f.owner_income + f.owner_investment_income for f in self.firms]
            )
            self.gini = self._compute_gini(incomes)

            recipient_mask = [w.transfer_income > 0 for w in self.workers]
            recipient_incomes = [
                w.wage + w.investment_income + w.transfer_income
                for w, is_r in zip(self.workers, recipient_mask) if is_r
            ]
            non_recipient_incomes = (
                [w.wage + w.investment_income
                 for w, is_r in zip(self.workers, recipient_mask) if not is_r]
                + [f.owner_income + f.owner_investment_income for f in self.firms]
            )
            self._last_gini_recipients = self._compute_gini(recipient_incomes)
            self._last_gini_non_recipients = self._compute_gini(non_recipient_incomes)
            self._last_mean_transfer_recipients = (
                float(np.mean([w.transfer_income for w in self.workers if w.transfer_income > 0]))
                if any(recipient_mask) else 0.0
            )
            self._last_n_recipients = int(sum(recipient_mask))
            var_shares = self._income_variance_shares(
                wages=[w.wage + w.investment_income for w in self.workers],
                transfers=[w.transfer_income for w in self.workers],
                owner_incomes=[f.owner_income + f.owner_investment_income for f in self.firms],
            )
            self._last_wage_var_share = var_shares["wage_share"]
            self._last_transfer_var_share = var_shares["transfer_share"]
            self._last_owner_var_share = var_shares["owner_share"]
            growth = float(np.mean([f.profit for f in self.firms]))
            # 'free' mode has no real-world political_violence series to
            # pull from -- left as None (State/Worker already treat None/
            # absent history as a no-op, not a fabricated 0) unless a
            # future endogenous repression-intensity proxy is added.

        avg_wage = float(np.mean([w.wage for w in self.workers]))
        protest_share = (
            float(np.mean([w.is_protesting for w in self.workers]))
            if self.steps > 0 else 0.0
        )

        # 3. state observes -> decides -> redistributes
        self.state.observe(avg_wage=avg_wage, gini=self.gini,
                            protest_intensity=protest_share, growth=growth,
                            political_violence=political_violence)
        self.state.decide_policy(lag_periods=1)
        self.state.redistribute(self.workers)

        tier_wages = {}
        for w in self.workers:
            tier_wages.setdefault(w.occupation, []).append(w.wage)
        self.tier_avg_wage = {
            tier: float(np.mean(wages)) for tier, wages in tier_wages.items()
        }

        # 4. workers react to this tick's just-updated state
        for w in self.workers:
            w.step()

        # 5. record
        self.datacollector.collect(self)


# ============================================================================
# __main__ — smoke-test-scale run across every fitted country, all regimes
# ============================================================================

if __name__ == "__main__": #return to check on UK and Russia fit later iff u wanna
    print("=" * 70)
    print("ABM MODEL — SMOKE RUN (small population, few steps, all countries)")
    print("=" * 70)

    fits = get_country_fits()
    print(f"\nUsable fits: {sorted(fits)}\n")

    for country in sorted(fits):
        for regime in Regime.ALL:
            print(f"\n--- {country} | regime: {regime} (free mode) ---")
            model = StateResponseModel(country, fit=fits[country], regime=regime,
                                        n_workers=200, n_firms=20, mode="free", seed=7)
            for _ in range(10):
                model.step()
            df = model.datacollector.get_model_vars_dataframe()
            print(df[["gini", "unemployment_rate", "protest_share",
                      "redistribution", "coordination_success_rate",
                      "mean_falsification_gap"]].tail(3).round(3).to_string())

        print(f"\n--- {country} | historical mode, representative regime ---")
        model = StateResponseModel(country, fit=fits[country], regime=Regime.REPRESENTATIVE,
                                    n_workers=200, n_firms=20, mode="historical", seed=7)
        n_steps = min(15, len(model.historical_data))
        for _ in range(n_steps):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        print(df[["gini", "redistribution"]].round(3).to_string())

    print("\n✓ Model ran cleanly across all fitted countries, regimes, and both run modes.")

# """
# abm_model.py

# Top-level Mesa Model tying together the State/Worker/Firm agents in
# agents/. Lives outside the agents/ package since it's the orchestration
# layer, not an agent itself.

# Two run modes, both wired in from the start (not bolted on later):
#   - 'free'       : Gini and growth evolve endogenously from the simulated
#                     labor market (wage distribution -> Gini; mean firm
#                     profit -> growth proxy). This is what regime-
#                     comparison experiments need.
#   - 'historical' : Gini and growth are pulled from the real annual data
#                     each step instead of computed endogenously, so
#                     simulated redistribution can be checked against the
#                     actual historical series (the README's validation
#                     question).

# Per-tick order matters and is fixed deliberately:
#   1. Firms step first (labor market clears: wages, hiring/firing, profit)
#   2. This period's aggregate observables are computed/pulled
#   3. State observes -> decides -> redistributes
#   4. Workers step, reacting to *this* tick's just-updated state, not
#      last tick's stale redistribution/repression values
#   5. DataCollector records the tick
# """

# import numpy as np
# import mesa

# from agents.state import (
#     StateResponseFitter, State, Regime, BayesianVAR,
#     load_and_prepare_data, prepare_var_data, build_model_inputs,
# )
# from agents.worker import Worker, build_worker_network, calibrate_population_from_bvar
# from agents.firm import Firm


# class StateResponseModel(mesa.Model):
#     def __init__(self, n_workers=1000, n_firms=100, regime=Regime.REPRESENTATIVE,
#                  response_fn=None, response_params=None, response_residuals=None,
#                  response_domain_bounds=None,
#                  worker_calibration=None, monopsony_power=0.0,
#                  mode='free', historical_data=None,
#                  initial_gini=0.4, initial_avg_wage=20.0,
#                  market_return_mean=0.02, market_return_std=0.08, seed=None,
#                  redistribution_enabled=True):
#         super().__init__(seed=seed)
#         self.random_gen = np.random.default_rng(seed)

#         if response_fn is None or response_params is None:
#             raise ValueError(
#                 "StateResponseModel requires a fitted response_fn/response_params "
#                 "(from StateResponseFitter) -- it does not silently assume a "
#                 "functional form. Use StateResponseModel.from_calibrated_data(...) "
#                 "for the common case of fitting from the CSV and building a model "
#                 "in one call."
#             )
#         if mode not in ('free', 'historical'):
#             raise ValueError("mode must be 'free' or 'historical'")
#         if mode == 'historical' and historical_data is None:
#             raise ValueError("mode='historical' requires historical_data (the annual "
#                               "DataFrame from load_and_prepare_data)")

#         self.mode = mode
#         self.historical_data = historical_data
#         self._historical_row_index = 0
#         self.last_redistribution_delta = 0.0
#         self.gini = initial_gini
#         # Shared macro investment climate, redrawn each tick (see step()) --
#         # this is what makes capital compounding artificial-but-principled
#         # rather than a guaranteed enrichment channel: some ticks are bull
#         # markets, some are bear markets, and individual agents' realized
#         # returns further vary around this via their own idiosyncratic
#         # noise (see Worker/Firm _update_*_wealth_and_investment).
#         self.market_return_mean = market_return_mean
#         self.market_return_std = market_return_std
#         self.market_return = 0.0

#         worker_calibration = dict(worker_calibration or {})

#         # --- firms first: labor demand side ---
#         # productivity_scale converts unitless skill*productivity (~0.4
#         # mean at default draws) to the same order of magnitude as wages,
#         # so firms start near break-even rather than guaranteed
#         # bankrupt -- see Firm's docstring for why this matters.
#         productivity_scale = initial_avg_wage / 0.4
#         self.firms = [
#             Firm(self, productivity=self.random_gen.uniform(0.4, 1.0),
#                  initial_wage=initial_avg_wage, monopsony_power=monopsony_power,
#                  productivity_scale=productivity_scale, rng=self.random_gen)
#             for _ in range(n_firms)
#         ]

#         # --- workers ---
#         self.workers = [
#             Worker(self, skill=self.random_gen.uniform(0.1, 1.0),
#                    initial_wage=initial_avg_wage, rng=self.random_gen,
#                    **worker_calibration)
#             for _ in range(n_workers)
#         ]
#         self.workers_by_id = {w.unique_id: w for w in self.workers}
#         self.worker_network = build_worker_network(
#             list(self.workers_by_id.keys()), k=6, seed=seed
#         )

#         # initial employment: round-robin assignment across firms so no
#         # firm starts empty and no worker starts unassigned
#         for i, w in enumerate(self.workers):
#             f = self.firms[i % n_firms]
#             f.employees.append(w)
#             w.firm = f

#         # --- state (single agent, regime-parameterized) ---
#         residuals = response_residuals if response_residuals is not None else np.array([0.0])
#         self.state = State(self, response_fn, response_params, regime=regime,
#                             residuals=residuals, domain_bounds=response_domain_bounds,
#                             rng=self.random_gen,
#                             redistribution_enabled=redistribution_enabled)

#         # --- data collection ---
#         self.datacollector = mesa.DataCollector(
#             model_reporters={
#                 "gini": lambda m: m.gini,
#                 "avg_wage": lambda m: float(np.mean([w.wage for w in m.workers])),
#                 "unemployment_rate": lambda m: float(np.mean(
#                     [w.employment_status == "unemployed" for w in m.workers])),
#                 "protest_share": lambda m: float(np.mean(
#                     [w.is_protesting for w in m.workers])),
#                 "mean_grievance": lambda m: float(np.mean([w.grievance for w in m.workers])),
#                 "repression_bound_share": lambda m: float(np.mean(
#                     [w.repression_binding() for w in m.workers])),
#                 "redistribution": lambda m: m.state.redistribution,
#                 "total_firm_profit": lambda m: float(sum(f.profit for f in m.firms)),
#                 "regime": lambda m: m.state.regime,
#                 "gini_recipients": lambda m: m._last_gini_recipients,
#                 "gini_non_recipients": lambda m: m._last_gini_non_recipients,
#                 "mean_transfer_per_recipient": lambda m: m._last_mean_transfer_recipients,
#                 "n_recipients": lambda m: m._last_n_recipients,
#                 "wage_var_share": lambda m: m._last_wage_var_share,
#                 "transfer_var_share": lambda m: m._last_transfer_var_share,
#                 "owner_var_share": lambda m: m._last_owner_var_share,
#             }
#         )

#     # ------------------------------------------------------------------
#     # Interfaces Firm/State call on the model (keeps agents decoupled
#     # from each other -- they only ever talk to `self.model`, never
#     # directly to another agent type's internals)
#     # ------------------------------------------------------------------
#     @property
#     def labor_market_tightness(self):
#         unemployed = sum(1 for w in self.workers if w.employment_status == "unemployed")
#         # crude openings proxy: firms currently profitable enough to hire
#         openings = sum(1 for f in self.firms if f.profit > f.target_profit)
#         return openings / max(unemployed, 1)

#     def draw_unemployed_worker(self):
#         pool = [w for w in self.workers if w.employment_status == "unemployed"]
#         return self.random_gen.choice(pool) if len(pool) else None

#     @staticmethod
#     def _income_variance_shares(wages, transfers, owner_incomes):
#         """
#         What share of total cross-sectional income VARIANCE (the quantity
#         Gini is ultimately summarizing) comes from each income source this
#         tick: wages, transfers, owner (capital) income. Computed via each
#         component's contribution to total variance under Var(A+B+C) =
#         Var(A)+Var(B)+Var(C)+2*Cov terms -- reported as each component's
#         own variance as a fraction of total variance of the summed
#         series, not a full covariance decomposition (simpler, and the
#         covariance cross-terms are small here since transfers/owner-income
#         are drawn from largely disjoint populations).
#         """
#         wages = np.asarray(wages, dtype=float)
#         transfers = np.asarray(transfers, dtype=float)
#         owner = np.asarray(owner_incomes, dtype=float)

#         # pad shorter arrays with zeros so all three represent the same
#         # total population length for a fair variance comparison
#         n = max(len(wages), len(transfers), len(owner))
#         wages = np.pad(wages, (0, n - len(wages)))
#         transfers = np.pad(transfers, (0, n - len(transfers)))
#         owner = np.pad(owner, (0, n - len(owner)))

#         total = wages + transfers + owner
#         total_var = np.var(total)
#         if total_var <= 0:
#             return dict(wage_share=np.nan, transfer_share=np.nan, owner_share=np.nan)

#         return dict(
#             wage_share=float(np.var(wages) / total_var),
#             transfer_share=float(np.var(transfers) / total_var),
#             owner_share=float(np.var(owner) / total_var),
#         )

#     @staticmethod
#     def _compute_gini(wages):
#         """Standard discrete Gini from a wage array; used only in 'free' mode."""
#         x = np.sort(np.clip(np.asarray(wages, dtype=float), 0.0, None))
#         n = len(x)
#         if n == 0 or x.sum() <= 0:
#             return 0.0
#         cum = np.cumsum(x)
#         gini = float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
#         assert -1e-9 <= gini <= 1.0 + 1e-9, f"Gini out of bounds: {gini}" #small recipient subset edge case
#         gini = float(np.clip(gini, 0.0, 1.0))
#         return gini

#     def _pull_historical_row(self):
#         idx = min(self._historical_row_index, len(self.historical_data) - 1)
#         row = self.historical_data.iloc[idx]
#         self._historical_row_index += 1
#         return row

#     # ------------------------------------------------------------------
#     def step(self):
#         # macro investment climate for this tick -- drawn once, shared by
#         # every agent's investment mechanism (see __init__ docstring)
#         self.market_return = float(
#             self.random_gen.normal(self.market_return_mean, self.market_return_std)
#         )

#         # Targeting diagnostic defaults -- overwritten below in 'free' mode
#         # only (historical mode has no comparable free-market Gini
#         # decomposition to compute this against).
#         self._last_gini_recipients = np.nan
#         self._last_gini_non_recipients = np.nan
#         self._last_mean_transfer_recipients = np.nan
#         self._last_n_recipients = 0
#         self._last_wage_var_share = np.nan
#         self._last_transfer_var_share = np.nan
#         self._last_owner_var_share = np.nan

#         # 1. firms clear the labor market first
#         for f in self.firms:
#             f.step()

#         # 2. this period's aggregate observables
#         if self.mode == "historical":
#             row = self._pull_historical_row()
#             # .get() only falls back on a *missing* key, not a NaN value
#             # (e.g. pre-1963 rows before Gini coverage starts) -- guard
#             # explicitly so early ticks don't propagate NaN through the
#             # whole state/worker chain
#             row_gini = row.get("gini_coefficient", np.nan)
#             row_growth = row.get("gdp_growth_yoy_pct", np.nan)
#             self.gini = float(row_gini) if not (row_gini is None or np.isnan(row_gini)) else self.gini
#             growth = float(row_growth) if not (row_growth is None or np.isnan(row_growth)) else 0.0
#         else:
#             # Gini now spans worker wages (skill- and firm-stratified),
#             # worker investment income, firm owner profit-share income,
#             # AND owner investment income -- capital compounding is the
#             # dominant real-world driver of inequality (Piketty's r > g),
#             # so this is where the biggest share of the earlier Gini gap
#             # (simulated ~0.12-0.20 vs. real ~0.35-0.42) is meant to close.
#             incomes = (
#                 [w.wage + w.investment_income + w.transfer_income for w in self.workers]
#                 + [f.owner_income + f.owner_investment_income for f in self.firms]
#             )
#             self.gini = self._compute_gini(incomes)

#             # Targeting diagnostic: split Gini into "recipients" (workers
#             # carrying a nonzero transfer_income assigned by LAST tick's
#             # State.redistribute() -- this tick's call hasn't happened yet,
#             # it's step 3 below) vs everyone else. Tests whether a rising
#             # Gini is concentrated within the transfer-recipient group
#             # (fixed bottom-quartile-by-wage, same ~250 workers every tick)
#             # rather than distributed broadly -- see chat notes on the
#             # dose-response result.
#             recipient_mask = [w.transfer_income > 0 for w in self.workers]
#             recipient_incomes = [
#                 w.wage + w.investment_income + w.transfer_income
#                 for w, is_r in zip(self.workers, recipient_mask) if is_r
#             ]
#             non_recipient_incomes = (
#                 [w.wage + w.investment_income
#                  for w, is_r in zip(self.workers, recipient_mask) if not is_r]
#                 + [f.owner_income + f.owner_investment_income for f in self.firms]
#             )
#             self._last_gini_recipients = self._compute_gini(recipient_incomes)
#             self._last_gini_non_recipients = self._compute_gini(non_recipient_incomes)
#             self._last_mean_transfer_recipients = (
#                 float(np.mean([w.transfer_income for w in self.workers if w.transfer_income > 0]))
#                 if any(recipient_mask) else 0.0
#             )
#             self._last_n_recipients = int(sum(recipient_mask))
#             # Variance decomposition: is measured Gini dominated by owner/
#             # capital-income variance, or by wage or transfer variance?
#             var_shares = self._income_variance_shares(
#                 wages=[w.wage + w.investment_income for w in self.workers],
#                 transfers=[w.transfer_income for w in self.workers],
#                 owner_incomes=[f.owner_income + f.owner_investment_income for f in self.firms],
#             )
#             self._last_wage_var_share = var_shares["wage_share"]
#             self._last_transfer_var_share = var_shares["transfer_share"]
#             self._last_owner_var_share = var_shares["owner_share"]
#             # crude endogenous growth proxy: normalized mean firm profit.
#             # Not a real GDP estimate -- flagged here rather than dressed
#             # up as one; fine for regime-comparison experiments where
#             # only relative differences across regimes matter, not the
#             # absolute growth number.
#             growth = float(np.mean([f.profit for f in self.firms]))

#         avg_wage = float(np.mean([w.wage for w in self.workers]))
#         protest_share = (
#             float(np.mean([w.is_protesting for w in self.workers]))
#             if self.steps > 0 else 0.0
#         )

#         # 3. state observes -> decides -> redistributes
#         self.state.observe(avg_wage=avg_wage, gini=self.gini,
#                             protest_intensity=protest_share, growth=growth)
#         self.state.decide_policy(lag_periods=1)
#         self.state.redistribute(self.workers)

#         tier_wages = {}
#         for w in self.workers:
#             tier_wages.setdefault(w.occupation, []).append(w.wage)
#         self.tier_avg_wage = {
#             tier: float(np.mean(wages)) for tier, wages in tier_wages.items()
#         }

#         # 4. workers react to this tick's just-updated state
#         for w in self.workers:
#             w.step()

#         # 5. record
#         self.datacollector.collect(self)

#     # ------------------------------------------------------------------
#     @classmethod
#     def from_calibrated_data(cls, csv_path="results/combined_long_panel.csv",
#                               regime=Regime.REPRESENTATIVE,
#                               n_workers=1000, n_firms=100, mode="free", seed=None,
#                               redistribution_enabled=True):
#         """
#         Convenience constructor: runs the state.py fitting pipeline
#         (StateResponseFitter + BayesianVAR) once and wires the results
#         straight into a new model instance. Use this for regime-
#         comparison experiments so every regime is built from the exact
#         same fit rather than each accidentally re-fitting (and
#         potentially drifting slightly, since curve_fit isn't perfectly
#         deterministic across calls with different starting states).

#         Returns (model, fitter, bvar) -- the fitter/bvar are returned too
#         since experiments.py will want their diagnostics (AIC table,
#         Granger checks) alongside the model itself.
#         """
#         annual = load_and_prepare_data(csv_path)
#         X, y, years = build_model_inputs(annual)
#         fitter = StateResponseFitter(X, y, years)
#         best_name, results = fitter.fit_and_compare()
#         best = results[best_name]

#         if best["type"] != "parametric":
#             raise ValueError(
#                 f"Best-fit model '{best_name}' is a GAM, which has a different "
#                 "predict() signature than State.decide_policy expects. Either "
#                 "wrap gam.predict in a State-compatible callable before using "
#                 "it here, or force a parametric candidate for ABM wiring."
#             )
#         response_fn = getattr(fitter, f"{best_name}_model")
#         response_params = best["params"]
#         residuals = y - best["y_pred"]

#         var_data = prepare_var_data(annual)
#         bvar = BayesianVAR(var_data, lag=1).fit()
#         worker_calibration = calibrate_population_from_bvar(bvar)

#         historical_data = annual if mode == "historical" else None
#         initial_gini = float(annual["gini_coefficient"].dropna().iloc[0])

#         model = cls(
#             n_workers=n_workers, n_firms=n_firms, regime=regime,
#             response_fn=response_fn, response_params=response_params,
#             response_residuals=residuals, response_domain_bounds=fitter.domain_bounds_,
#             worker_calibration=worker_calibration,
#             mode=mode, historical_data=historical_data,
#             initial_gini=initial_gini, initial_avg_wage=20.0, seed=seed,
#             redistribution_enabled=redistribution_enabled,
#         )
#         return model, fitter, bvar


# # ============================================================================
# # __main__ — smoke-test-scale run, not a full experiment (that's
# # experiments.py's job): confirms the model actually runs across all
# # three regimes and both modes, and prints a short summary.
# # ============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("ABM MODEL — SMOKE RUN (small population, few steps)")
#     print("=" * 70)

#     for regime in Regime.ALL:
#         print(f"\n--- regime: {regime} (free mode) ---")
#         model, fitter, bvar = StateResponseModel.from_calibrated_data(
#             regime=regime, n_workers=200, n_firms=20, mode="free", seed=7
#         )
#         for _ in range(10):
#             model.step()
#         df = model.datacollector.get_model_vars_dataframe()
#         print(df[["gini", "unemployment_rate", "protest_share", "redistribution"]]
#               .tail(5).round(3).to_string())

#     print("\n--- historical mode, representative regime ---")
#     model, fitter, bvar = StateResponseModel.from_calibrated_data(
#         regime=Regime.REPRESENTATIVE, n_workers=200, n_firms=20,
#         mode="historical", seed=7,
#     )
#     n_steps = min(15, len(model.historical_data))
#     for _ in range(n_steps):
#         model.step()
#     df = model.datacollector.get_model_vars_dataframe()
#     print(df[["gini", "redistribution"]].round(3).to_string())
#     print("\n✓ Model ran cleanly across all regimes and both run modes.")