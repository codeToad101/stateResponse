class Worker:
    def __init__(self, skill, initial_wage):
        self.skill = skill  # 0-1, determines job access
        self.wage = initial_wage
        self.employment_status = "employed" or "unemployed"
        self.grievance = 0  # accumulates dissatisfaction
        self.risk_perception = 0  # fear of protest consequence
        self.network = []  # neighbors who influence threshold
    
    def step(self, state, firms, inequality_gini):
        # Update grievance: compare wage to reference point (local avg + macro trend)
        reference_wage = state.avg_wage * (1 - inequality_gini)
        self.grievance = max(0, (reference_wage - self.wage) / reference_wage)
        
        # Update risk: depends on protest visibility + state repression
        visible_protesters = count_active_protesters_in_network()
        state_capacity = state.police_intensity  # exogenous, can vary
        self.risk_perception = state_capacity * (1 - visible_protesters / state.population)
        
        # DECISION: Protest if net grievance exceeds threshold
        if self.grievance - self.risk_perception > self.threshold:
            self.protest()  # joins protest, signals to state
        else:
            self.work()  # produces output, receives wage