import re

with open('inference.py', 'r') as f:
    code = f.read()

pattern = re.compile(r'# 1\. Fire \(Priority 1\).*?# 3\. Traffic \(Priority 3\)', re.DOTALL)

replacement = """# 1. Fire (Priority 1)
            # Step 1: guarantee at least 1 to any active fire if possible
            for score, z_name, z_state in zone_scores:
                if z_state.fire != FireLevel.NONE and idle_fire > 0:
                    allocations[z_name]["dispatch_fire"] = 1
                    idle_fire -= 1
                    
            for score, z_name, z_state in zone_scores:
                req = int(self._fire_reqs(z_state.fire, obs.weather) * req_multiplier)
                if z_state.consecutive_failures > 0: req += 1
                
                # subtract the 1 we already gave
                if allocations[z_name]["dispatch_fire"] > 0:
                    req -= 1
                    
                disp = min(idle_fire, req)
                if disp > 0:
                    allocations[z_name]["dispatch_fire"] += disp
                    idle_fire -= disp

            # 2. Medical (Priority 2)
            for score, z_name, z_state in zone_scores:
                if z_state.patient != PatientLevel.NONE and idle_amb > 0:
                    allocations[z_name]["dispatch_ambulance"] = 1
                    idle_amb -= 1
                    
            for score, z_name, z_state in zone_scores:
                req = int(self._patient_reqs(z_state.patient) * req_multiplier)
                if z_state.consecutive_failures > 0: req += 1
                
                if allocations[z_name]["dispatch_ambulance"] > 0:
                    req -= 1
                    
                disp = min(idle_amb, req)
                if disp > 0:
                    allocations[z_name]["dispatch_ambulance"] += disp
                    idle_amb -= disp
                    
            # 3. Traffic (Priority 3)"""

new_code = pattern.sub(replacement, code)
with open('inference.py', 'w') as f:
    f.write(new_code)
