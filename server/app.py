import os
import time
import json
import pandas as pd
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CrisisManagementEnv
from env.models import Action, Observation, FireLevel, PatientLevel, TrafficLevel, WeatherCondition, ResourcePool, ZoneDispatch, ZoneState
from inference import StrategicAgent
from metrics_tracker import MetricsTracker

app = FastAPI(title="Adaptive Multi-Zone Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_env = CrisisManagementEnv(task_id=3) # Use task 3 as default for demo
metrics_tracker = MetricsTracker()
step_history = []

@app.get("/api/health")
def health_check(): return {"status": "ok"}

@app.post("/reset")
def reset(task_id: int = 3):
    global api_env
    global metrics_tracker, step_history
    api_env.__init__(task_id=task_id)
    metrics_tracker = MetricsTracker()
    step_history = []
    return api_env.reset().model_dump()

@app.get("/reset")
def get_reset(task_id: int = 3): # support GET just in case
    return reset(task_id)

@app.post("/step")
def step(action: dict):
    # Action looks like: {"allocations": {"Downtown": {...}, ...}}
    act = Action(**action)
    obs, reward, done, info = api_env.step(act)
    metrics_tracker.update(reward, act, obs, done)
    step_history.append({
        "step": len(step_history) + 1,
        "reward": float(reward),
        "done": bool(done),
        "action": action,
        "observation": obs.model_dump()
    })
    return {
        "observation": obs.model_dump(), 
        "reward": float(reward), 
        "done": bool(done), 
        "info": info,
        "metrics": metrics_tracker.get_summary()
    }

@app.get("/state")
def state(): 
    return api_env.state().model_dump()

@app.get("/metrics")
def get_metrics_api():
    summ = metrics_tracker.get_summary()
    return {
        "efficiency": summ["efficiency"],
        "stability": summ["stability"],
        "hazards_prevented": summ["hazards_prevented"],
        "total_reward": metrics_tracker.total_reward
    }

@app.get("/history")
def get_history_api():
    return step_history

@app.get("/live-state")
def get_live_state():
    obs = api_env.state().observation
    return {
        "zones": obs.zones,
        "resources": {
            "idle": obs.idle_resources.model_dump(),
            "busy": obs.busy_resources.model_dump()
        },
        "step": obs.step,
        "max_steps": obs.max_steps,
        "weather": obs.weather.value,
        "metrics": metrics_tracker.get_summary()
    }

smart_agent_instance = SmartAgent()
@app.post("/agent_action")
def get_agent_action(obs: Observation, use_llm: bool = False):
    action, reason = smart_agent_instance.get_action_with_reason(obs, use_llm)
    return {"action": action.model_dump(), "reason": reason}

class SmartAgent(OpenAIAgent):
    def get_action_with_reason(self, obs: Observation, use_llm: bool):
        if not use_llm or not self.client:
            act = BaselineAgent().get_action(obs)
            return act, "(Rule-Based) Triggering heuristic zone allocations sequentially..."

        prompt = f"""
You are the Smart City General AI.
Weather: {obs.weather.value}
Idle Reserves: {obs.idle_resources.model_dump_json()}
Active Zone Statuses (Grid): 
{json.dumps({k: v.model_dump() for k,v in obs.zones.items()}, indent=2)}

Constraint Framework:
- fire: catastrophic=5, high=3, medium=2, low=1 (Hurricane config adds +2, Storm adds +1)
- patient: critical=3, moderate=1 (Gridlock + explicitly missed Police adds +2)
- traffic: heavy/gridlock=True demands police

Allocate available reserves to optimize routing across ALL ZONES.

Return VALID JSON:
{{
  "allocations": {{
    "Downtown": {{"dispatch_fire": int, "dispatch_ambulance": int, "control_traffic": bool}},
    "Suburbs": ...
  }},
  "reasoning": "Explain routing distribution."
}}
"""
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
            raw = resp.choices[0].message.content.strip()
            if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw: raw = raw.replace("```", "").strip()
            data = json.loads(raw)
            
            allocations = {}
            for k, v in data.get("allocations", {}).items():
                allocations[k] = ZoneDispatch(**v)
                
            return Action(allocations=allocations), data.get("reasoning", "Multi-zone allocations matched.")
        except Exception as e:
            return BaselineAgent().get_action(obs), f"LLM Network routing rejected. Base heuristic trigger. {str(e)}"

def render_city_grid(obs: Observation, action: Action = None):
    fire_map = {"none": "✔️ Safe", "low": "🔥 Small", "medium": "🔥🔥 Medium", "high": "🔥🔥🔥 HIGH", "catastrophic": "🌋 CATASTROPHIC"}
    pat_map = {"none": "✔️ Safe", "moderate": "🤕 Moderate", "critical": "💀 CRITICAL", "fatal": "☠️ FATAL"}
    traf_map = {"low": "🟢 Clear", "heavy": "🟡 Heavy", "gridlock": "🔴 GRIDLOCK"}
    weath_map = {"clear": "☀️ Clear", "storm": "⛈️ Storm", "hurricane": "🌀 HURRICANE"}
    
    html = f"""
    <div style="background: rgba(15,20,30,0.9); padding: 15px; border-radius: 12px; border: 2px solid #38bdf8; font-family: sans-serif; min-height: 250px;">
        <h3 style="color: #38bdf8; text-align: center; text-transform: uppercase; margin-top:0;">CITY WIDE EMERGENCY GRID</h3>
        <div style="margin-bottom: 10px; text-align: center;">
            <span style="font-size: 1.2em; border: 1px solid #fbbf24; padding: 5px; border-radius: 5px;">☁️ GLOBAL WEATHER: <span style="color:#fbbf24;">{weath_map[obs.weather.value]}</span></span>
        </div>
        <div style="display: flex; justify-content: space-between; text-align: center; margin-top: 15px; flex-wrap: wrap;">
    """
    
    for z_id, z in obs.zones.items():
        act = action.allocations.get(z_id, ZoneDispatch()) if action and action.allocations else ZoneDispatch()
        
        has_issue = z.fire != FireLevel.NONE or z.patient not in [PatientLevel.NONE, PatientLevel.FATAL] or z.traffic != TrafficLevel.LOW
        panel_color = "rgba(100,0,0,0.5)" if has_issue else "rgba(0,100,0,0.3)"
        
        html += f"""
        <div style="flex: 1; margin: 5px; min-width: 200px; background: {panel_color}; padding: 15px; border-radius: 8px; border: 1px solid #475569;">
            <h4 style="color: #cbd5e1; border-bottom: 1px solid #334155; padding-bottom: 5px;">📍 {z_id.upper()}</h4>
            <div style="font-size: 1.5em; margin: 10px 0; text-shadow: 0 0 10px red;">{fire_map[z.fire.value]}</div>
            <div style="font-size: 1.5em; margin: 10px 0; text-shadow: 0 0 10px magenta;">{pat_map[z.patient.value]}</div>
            <div style="font-size: 1.5em; margin: 10px 0; text-shadow: 0 0 10px yellow;">{traf_map[z.traffic.value]}</div>
            <hr style="border-color: #475569;">
            <div style="color: #fb923c; font-weight: bold;">🚒 Sent: {act.dispatch_fire}</div>
            <div style="color: #f472b6; font-weight: bold;">🚑 Sent: {act.dispatch_ambulance}</div>
            <div style="color: #fbbf24; font-weight: bold;">🚓 Sent: {1 if act.control_traffic else 0}</div>
        </div>
        """
        
    html += f"""
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px; padding: 10px; border-top: 1px solid #334155;">
            <div style="text-align: center; width: 100%;">
                <span style="color: #10b981; font-weight: bold; font-family: monospace; font-size: 1.2em; margin-right: 20px;">🟢 IDLE (BASE): {obs.idle_resources.fire_units}F {obs.idle_resources.ambulances}A {obs.idle_resources.police}P</span>
                <span style="color: #ef4444; font-weight: bold; font-family: monospace; font-size: 1.2em;">🔴 BUSY (FIELD): {obs.busy_resources.fire_units}F {obs.busy_resources.ambulances}A {obs.busy_resources.police}P</span>
            </div>
        </div>
    </div>
    """
    return html

css = """
body, .gradio-container { background-color: #050510 !important; color: #f8fafc !important; }
.gr-button-primary { background: linear-gradient(90deg, #ec4899, #8b5cf6) !important; border: none !important; box-shadow: 0 0 15px rgba(236,72,153,0.5) !important; }
.panel-glow { border: 1px solid #1e293b; background: rgba(15,23,42,0.6); box-shadow: 0 0 20px rgba(139,92,246,0.15); border-radius: 12px; padding: 20px; }
.reasoning-box { background: rgba(5,5,15,0.8); border-left: 4px solid #c084fc; padding: 15px; font-family: monospace; border-radius: 5px; color: #d8b4fe; font-size: 1.1em;}
"""

with gr.Blocks(theme=gr.themes.Base(), css=css, title="Meta Capstone Spatial Routing") as demo:
    gr.HTML("<center><h1 style='color: #ec4899; text-shadow: 0 0 20px #ec4899; font-size:3em; margin-bottom:0;'>🗺️ City-Wide Spatial Orchestrator</h1></center>")
    gr.HTML("<center><p style='color: #8b5cf6; font-size:1.2em; margin-top:5px;'>Advanced Reinforcement Learning Multi-Node Network</p></center>")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel-glow"):
            gr.Markdown("### 🎛️ Architecture Selection")
            agent_select = gr.Radio(choices=["Rule-Based System", "RL Agent (Meta LLaMA)"], value="RL Agent (Meta LLaMA)", label="🤖 Traffic AI Logic")
            task_select = gr.Radio(choices=["1 (Downtown Fire)", "2 (Suburban Storm)", "3 (City-Wide Hurricane)"], value="3 (City-Wide Hurricane)", label="🌐 Simulation Complexity Node")
            
            btn_start = gr.Button("▶ Boot Spatial Network", variant="primary")
            
            with gr.Row():
                score_box = gr.Number(label="Final Matrix Score (%)", value=0, interactive=False)
                eff_box = gr.Number(label="Overall Efficiency (%)", value=0, interactive=False)
            
        with gr.Column(scale=3):
            city_html = gr.HTML(render_city_grid(CrisisManagementEnv(task_id=3).obs))
            
            gr.Markdown("### 🧠 AI Orchestrator Reasoning Network")
            reasoning_box = gr.HTML("<div class='reasoning-box'>Awaiting telemetry simulation to boot...</div>")
            
            gr.Markdown("### 📈 Live Node Telemetrics")
            plot = gr.LinePlot(x="Step", y="Value", color="Metric", title="Model Performance Trajectory", height=250, x_title="Timeline Step")
            
    sim_data = gr.State(pd.DataFrame(columns=["Step", "Value", "Metric"]))

    def run_simulation(agent_type, task_node):
        df = pd.DataFrame(columns=["Step", "Value", "Metric"])
        
        task_id = int(task_node[0])
        env = CrisisManagementEnv(task_id=task_id)
        
        smart_agent = SmartAgent()
        use_llm = "LLaMA" in agent_type
        
        yield render_city_grid(env.obs), "<div class='reasoning-box'>[SPATIAL GRID CONNECTED] Initializing node mapping...</div>", df, 0, 0
        
        while not env.is_done:
            time.sleep(1.5)
            
            action, reason = smart_agent.get_action_with_reason(env.obs.model_copy(deep=True), use_llm)
            reason_html = f"<div class='reasoning-box'><b>Step {env.obs.step+1} Spatial Logistics:</b><br/>{reason}</div>"
            grid_html = render_city_grid(env.obs, action)
            
            yield grid_html, reason_html, df, 0, 0
            
            time.sleep(1.0)
            obs, reward, done, info = env.step(action)
            
            new_rows = pd.DataFrame([
                {"Step": obs.step, "Value": env.total_reward, "Metric": "Cumulative Reward Vector"},
                {"Step": obs.step, "Value": info.get("efficiency", 0) * 100, "Metric": "Network Efficiency %"}
            ])
            df = pd.concat([df, new_rows], ignore_index=True)
            
            yield render_city_grid(obs), reason_html, df, info.get("score", 0)*100, info.get("efficiency", 0)*100
            
        final_reason = "<div class='reasoning-box'><b>[SIMULATION ARCHIVED]</b> All spatial networks stabilized.</div>"
        yield render_city_grid(env.obs), final_reason, df, info.get("score", 0)*100, info.get("efficiency", 0)*100

    btn_start.click(
        run_simulation,
        inputs=[agent_select, task_select],
        outputs=[city_html, reasoning_box, plot, score_box, eff_box]
    )

class ConfigData(BaseModel):
    API_BASE_URL: str
    MODEL_NAME: str
    HF_TOKEN: str

@app.post("/config")
def update_config(cfg: ConfigData):
    import os
    os.environ["API_BASE_URL"] = cfg.API_BASE_URL
    os.environ["MODEL_NAME"] = cfg.MODEL_NAME
    os.environ["HF_TOKEN"] = cfg.HF_TOKEN
    return {"status": "ok"}

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
