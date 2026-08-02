class Firm:
    def __init__(self, productivity):
        self.productivity = productivity  # 0-1
        self.workers = []
        self.capital = 100
        self.profit = 0
    
    def step(self, labor_market_tightness, wage_pressure):
        # Wage-setting heuristic (bounded rationality)
        if labor_market_tightness > threshold_tight:
            self.wage = self.wage * (1 + wage_pressure)  # raise wages to attract/retain
        else:
            self.wage = self.wage * (1 - 0.02)  # cut wages when slack
        
        # Hiring: expand if profitable, contract if not
        if self.profit > target_profit:
            self.hire(2-3 workers)
        elif self.profit < 0:
            self.fire(1-2 workers)  # last hired, first fired (LIFO)
        
        # Output & profit
        output = sum([w.productivity for w in self.workers])
        costs = sum([w.wage for w in self.workers]) + depreciation
        self.profit = output - costs