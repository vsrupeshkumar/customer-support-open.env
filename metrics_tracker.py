class MetricsTracker:
    def __init__(self):
        self.total_reward = 0.0
        self.hazards_prevented = 0
        self.cascading_failures = 0
        self.avg_response_time = 0.0
        self.resource_efficiency = 0.0
        self.action_success_rate = 0.0
        self.step_count = 0
        self.idle_capacity_waste = 0

    def update(self, reward, action, observation, done):
        self.step_count += 1
        self.total_reward += reward
        
        # Calculate failures
        for z_name, z_state in observation.zones.items():
            if z_state.consecutive_failures > 0:
                self.cascading_failures += 1
                
        # Estimate efficiency
        total_resources = (observation.idle_resources.fire_units + observation.busy_resources.fire_units + 
                           observation.idle_resources.ambulances + observation.busy_resources.ambulances)
        
        if total_resources > 0:
            used = observation.busy_resources.fire_units + observation.busy_resources.ambulances
            self.resource_efficiency = (self.resource_efficiency * (self.step_count - 1) + (used / total_resources)) / self.step_count

        if reward > 0:
            self.hazards_prevented += 1

    def get_summary(self):
        stability = max(0.0, 1.0 - (self.cascading_failures / max(1, self.step_count * 3)))
        return {
            "efficiency": self.resource_efficiency if self.step_count > 0 else 0.95,
            "hazards_prevented": self.hazards_prevented,
            "stability": stability
        }
