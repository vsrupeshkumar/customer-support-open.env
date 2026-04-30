const statusPill = document.getElementById("status-pill");
const sessionInput = document.getElementById("session-id");
const applyButton = document.getElementById("apply-session");
const stateJson = document.getElementById("state-json");
const stepJson = document.getElementById("step-json");
const summaryEl = document.getElementById("summary");
const rewardEl = document.getElementById("reward-breakdown");
const timelineEl = document.getElementById("timeline");
const mapGrid = document.getElementById("map-grid");
const globalMetrics = document.getElementById("global-metrics");

let sessionId = "default";

applyButton.addEventListener("click", () => {
  sessionId = sessionInput.value.trim() || "default";
  fetchTelemetry();
});

function setStatus(text, good = true) {
  statusPill.textContent = text;
  statusPill.style.background = good
    ? "rgba(53, 208, 186, 0.15)"
    : "rgba(255, 107, 107, 0.2)";
  statusPill.style.color = good ? "#35d0ba" : "#ff6b6b";
}

function pill(label, value) {
  return `
    <div class="pill">
      <div class="label">${label}</div>
      <div>${value ?? "-"}</div>
    </div>
  `;
}

function severityScore(zone) {
  const fire = { none: 0, low: 1, medium: 2, high: 3, catastrophic: 4 };
  const patient = { none: 0, moderate: 1, critical: 3, fatal: 4 };
  const traffic = { low: 0, heavy: 1, gridlock: 3 };
  const score =
    (fire[zone.fire] || 0) +
    (patient[zone.patient] || 0) +
    (traffic[zone.traffic] || 0);
  return Math.min(score, 10);
}

function severityColor(score) {
  if (score >= 8) return "#ff6b6b";
  if (score >= 5) return "#ffb347";
  if (score >= 2) return "#35d0ba";
  return "#8ee7ff";
}

function renderSummary(state) {
  if (!state) {
    summaryEl.innerHTML = "";
    return;
  }
  const zones = Object.keys(state.zones || {}).length;
  summaryEl.innerHTML = [
    pill("Zones", zones),
    pill("Weather", state.weather),
    pill("Task", state.task_level),
  ].join("");
}

function renderRewards(lastStep) {
  rewardEl.innerHTML = "";
  if (!lastStep) return;
  const info = lastStep.info || {};
  rewardEl.innerHTML = [
    pill("Reward", lastStep.reward),
    pill("Done", lastStep.done),
    pill("Score", info.score),
    pill("Resolved", `${info.resolved ?? "-"}/${info.total ?? "-"}`),
  ].join("");
}

function renderTimeline(timeline) {
  if (!Array.isArray(timeline)) {
    timelineEl.innerHTML = "";
    return;
  }
  const items = timeline.slice(-8).reverse();
  timelineEl.innerHTML = items
    .map((item, idx) => {
      return `
        <div class="timeline-item">
          <span>Step -${idx + 1}</span>
          <span>Reward ${item.reward ?? "-"}</span>
        </div>
      `;
    })
    .join("");
}

function renderMap(state) {
  mapGrid.innerHTML = "";
  if (!state || !state.zones) return;
  const zones = Object.entries(state.zones);
  mapGrid.innerHTML = zones
    .map(([zoneId, zone]) => {
      const score = severityScore(zone);
      const color = severityColor(score);
      return `
        <div class="zone" style="background:${color}">
          ${zoneId}
          <small>Fire: ${zone.fire}</small>
          <small>Patient: ${zone.patient}</small>
          <small>Traffic: ${zone.traffic}</small>
        </div>
      `;
    })
    .join("");
}

function renderGlobalMetrics(metrics) {
  globalMetrics.innerHTML = "";
  if (!metrics) return;
  globalMetrics.innerHTML = [
    pill("Episodes", metrics.episodes_completed),
    pill("Mean Reward", metrics.mean_reward),
    pill("Completion", metrics.completion_rate),
  ].join("");
}

async function fetchTelemetry() {
  try {
    const response = await fetch(`/enhanced/metrics?session_id=${encodeURIComponent(sessionId)}`);
    if (!response.ok) {
      setStatus("Unavailable", false);
      return;
    }
    const payload = await response.json();
    const telemetry = payload.telemetry || {};
    const state = telemetry.state;
    const lastStep = telemetry.last_step;

    setStatus("Live", true);
    renderSummary(state);
    renderRewards(lastStep);
    renderTimeline(telemetry.timeline || []);
    renderMap(state);
    renderGlobalMetrics(telemetry.global_metrics);
    stateJson.textContent = JSON.stringify(state, null, 2) || "";
    stepJson.textContent = JSON.stringify(lastStep, null, 2) || "";
  } catch (error) {
    setStatus("Error", false);
  }
}

fetchTelemetry();
setInterval(fetchTelemetry, 2000);
