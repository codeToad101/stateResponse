from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

class StateResponseFitter:
    def __init__(self, historical_data):
        self.data = historical_data
        self.results = {}
    
    def linear_model(self, x, a, b, c):
        """Redistribution = α·protest + β·gini + γ·growth"""
        protest, gini, growth = x
        return a * protest + b * gini + c * growth
    
    def polynomial_model(self, x, a, b, c, d, e, f):
        """Includes quadratic terms (interaction effects)"""
        protest, gini, growth = x
        return a * protest + b * gini + c * growth + \
               d * (protest ** 2) + e * (gini ** 2) + f * protest * gini
    
    def logistic_model(self, x, a, b, c, k, x0):
        """State response saturates at high inequality/protest"""
        protest, gini, growth = x
        return a / (1 + np.exp(-k * (protest + b * gini - x0))) + c * growth
    
    def exponential_model(self, x, a, b, c):
        """Redistribution accelerates with inequality"""
        protest, gini, growth = x
        return a * np.exp(b * gini) + c * protest
    
    def fit_and_compare(self):
        """Fit all models, compare AIC/BIC"""
        models = {
            'linear': self.linear_model,
            'polynomial': self.polynomial_model,
            'logistic': self.logistic_model,
            'exponential': self.exponential_model,
        }
        
        for name, model in models.items():
            try:
                popt, _ = curve_fit(model, self.data['X'], self.data['y'])
                y_pred = model(self.data['X'], *popt)
                
                r2 = r2_score(self.data['y'], y_pred)
                rmse = np.sqrt(mean_squared_error(self.data['y'], y_pred))
                
                self.results[name] = {
                    'params': popt,
                    'r2': r2,
                    'rmse': rmse,
                    'aic': len(self.data) * np.log(rmse) + 2 * len(popt)
                }
                print(f"{name:12} | R²: {r2:.3f} | RMSE: {rmse:.4f} | AIC: {self.results[name]['aic']:.1f}")
            except Exception as e:
                print(f"{name:12} | FAILED: {e}")
        
        # Rank by AIC (lower = better)
        best_model = min(self.results, key=lambda x: self.results[x]['aic'])
        print(f"\n✓ Best model: {best_model}")
        return best_model, self.results

class State:
    def __init__(self):
        self.tax_rate = 0.25  # baseline (historical US avg)
        self.redistribution = 0
        self.police_intensity = 1.0  # repression capacity
        self.objective = "maximize_stability"  # or "maximize_growth"
    
    def observe(self, workers, firms, time_step):
        # Metrics that inform policy
        self.unemployment_rate = count_unemployed() / len(workers)
        self.gini = calculate_gini([w.wage for w in workers])
        self.protest_intensity = count_active_protesters() / len(workers)
        self.wage_growth_rate = (avg_wage_t - avg_wage_t_1) / avg_wage_t_1
        
        # Store history (state responds to *past* conflict, not present)
        self.past_protests.append(self.protest_intensity)
    
    def decide_policy(self, time_step):
        # STATE OBJECTIVE FUNCTION (calibrated to historical data)
        # Response rule: if past protests high OR inequality rising, increase redistribution
        
        past_protest_avg = mean(self.past_protests[-12:])  # trailing 12 quarters
        
        # Simple linear response (you can make complex later)
        target_redistribution = base_redistribution + \
                               response_coeff * past_protest_avg + \
                               inequality_response * self.gini
        
        # Constraint: budget balance
        total_wages = sum([w.wage for w in workers if w.employed])
        max_redistribution = self.tax_rate * total_wages
        
        self.redistribution = min(target_redistribution, max_redistribution)
        
        # Update tax rate if redistribution changes policy goals
        # (optional: make tax endogenous too)
    
    def redistribute(self, workers):
        # Distribute UBI or wage subsidies to lowest-earning workers
        recipient_workers = sorted(workers, key=lambda w: w.wage)[:len(workers)//4]
        subsidy_per_worker = self.redistribution / len(recipient_workers)
        
        for w in recipient_workers:
            w.wage += subsidy_per_worker