#!/usr/bin/env python3
"""
log_validator.py
================
Asserts zero deviation from the OpenEnv logging specification for standard [START]/[STEP]/[END] markers.
Tracks:
- Presence of task_id (task=)
- Presence of step number, action, and reward
- JSON validity of the action payload
- Presence of final cumulative score (score=)
- Sequential integrity of step execution

Usage:
    python inference.py --task 1 | python log_validator.py
"""

import sys
import re
import json

class LogValidator:
    def __init__(self):
        self.total_steps = 0
        self.start_found = False
        self.end_found = False
        self.current_step = 0
        
    def validate(self, log_content: str) -> bool:
        lines = log_content.strip().split('\n')
        
        # We need to filter lines to only consider [START], [STEP], [END] 
        # as inference.py emits them on stdout.
        # Everything else on stdout might trigger false negatives.
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("[START]"):
                self.start_found = True
                if "task=" not in line:
                    print(f"FAILED: [START] missing task_id. Line: {line}", file=sys.stderr)
                    return False
                    
            elif line.startswith("[STEP]"):
                self.total_steps += 1
                
                # Check for step
                step_match = re.search(r"step=(\d+)", line)
                if not step_match:
                    print(f"FAILED: [STEP] missing step number. Line: {line}", file=sys.stderr)
                    return False
                
                step_num = int(step_match.group(1))
                if step_num != self.current_step + 1:
                    print(f"FAILED: [STEP] out of order. Expected {self.current_step + 1}, got {step_num}", file=sys.stderr)
                    return False
                self.current_step = step_num
                
                # Check for reward
                if "reward=" not in line:
                    print(f"FAILED: [STEP] missing reward. Line: {line}", file=sys.stderr)
                    return False
                    
                # Support capturing up to the next kwarg if present natively eliminating RegEx brittle dependencies
                if "action=" not in line or "reward=" not in line:
                    print(f"FAILED: [STEP] missing or malformed action JSON. Line: {line}", file=sys.stderr)
                    return False
                
                # Split everything between action= and reward= globally parsing any nested JSON depth gracefully
                action_str = line.split("action=")[1].rsplit(" reward=", 1)[0].strip()
                try:
                    parsed = json.loads(action_str)
                    if not isinstance(parsed, dict):
                        print(f"FAILED: [STEP] action JSON is not a structured dict. Action string: {action_str}", file=sys.stderr)
                        return False
                except json.JSONDecodeError:
                    print(f"FAILED: [STEP] action is not valid JSON. Action string: {action_str}", file=sys.stderr)
                    return False
                    
            elif line.startswith("[END]"):
                self.end_found = True
                if "score=" not in line:
                    print(f"FAILED: [END] missing cum_score (score). Line: {line}", file=sys.stderr)
                    return False
                    
        # Final assertions
        if not self.start_found:
            print("FAILED: No [START] marker found.", file=sys.stderr)
            return False
        if not self.end_found:
            print("FAILED: No [END] marker found.", file=sys.stderr)
            return False
        if self.total_steps == 0:
            print("FAILED: No [STEP] markers found.", file=sys.stderr)
            return False
            
        print(f"Compliance: PASSED. Processed {self.total_steps} sequential steps.")
        return True

if __name__ == "__main__":
    if sys.stdin.isatty():
        print("Usage: cat logs.txt | python log_validator.py")
        sys.exit(1)
        
    content = sys.stdin.read()
    validator = LogValidator()
    if not validator.validate(content):
        sys.exit(1)
