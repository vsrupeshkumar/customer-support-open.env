/**
 * Crisis Management Dashboard — Client-Side Application (v2)
 * ===========================================================
 * Pure ES6 module-pattern application (no frameworks, no external deps).
 *
 * Architecture:
 *   CrisisClient       — Async HTTP API wrapper (fetch-based, never throws)
 *   SaliencyEngine      — Client-side DRR feature-attribution computation
 *   HeatmapRenderer     — Zone severity visualization with CSS class mapping
 *   GaugeRenderer       — Resource pool horizontal bar gauges
 *   SparklineRenderer   — Canvas 2D reward trajectory chart
 *   DispatchPanel       — Manual per-zone dispatch slider interface
 *   ActionLog           — Scrolling event log with color-coded entries
 *   EpisodeSummary      — Modal overlay with grading breakdown
 *   ConnectionMonitor   — Health polling with green/red dot indicator
 *   AutoPlayEngine      — Intelligent auto-dispatch with timed stepping
 *   KeyboardShortcuts   — R/Space/1/2/3/A hotkeys
 *
 * State: Module-scoped variables (no external state library).
 * DOM:   All updates via getElementById + textContent/innerHTML.
 */

(function () {
    'use strict';

    // =========================================================================
    // Module State
    // =========================================================================
    let currentObs = null;
    let stepCount = 0;
    let rewards = [];
    let selectedTask = 1;
    let sessionActive = false;
    let episodeDone = false;
    let lastScore = 0;
    let lastEfficiency = 0;
    let lastInfo = {};
    let autoPlayInterval = null;
    let autoPlayBusy = false;
    let zoneTimeline = {};    // zone → step at which it was resolved

    // =========================================================================
    // 1. CrisisClient — HTTP API Wrapper
    // =========================================================================
    const CrisisClient = {
        /** POST /reset — returns observation dict or {error}. */
        async reset(taskId, seed) {
            try {
                const body = { task_id: taskId };
                if (seed !== null && seed !== undefined && seed !== '') {
                    body.seed = parseInt(seed, 10);
                }
                const res = await fetch('/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                if (!res.ok) {
                    const err = await res.text();
                    return { error: `HTTP ${res.status}: ${err}` };
                }
                return await res.json();
            } catch (e) {
                return { error: e.message };
            }
        },

        /** POST /step — returns {observation, reward, done, info} or {error}. */
        async step(allocations, broadcastMessage) {
            try {
                const body = { allocations };
                if (broadcastMessage && broadcastMessage.trim()) {
                    body.public_broadcast_message = broadcastMessage.trim();
                }
                const res = await fetch('/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                if (!res.ok) {
                    const err = await res.text();
                    return { error: `HTTP ${res.status}: ${err}` };
                }
                return await res.json();
            } catch (e) {
                return { error: e.message };
            }
        },

        /** GET /health — returns health payload or {error}. */
        async getHealth() {
            try {
                const res = await fetch('/health');
                if (!res.ok) return { error: `HTTP ${res.status}` };
                return await res.json();
            } catch (e) {
                return { error: e.message };
            }
        }
    };

    // =========================================================================
    // 2. SaliencyEngine — Client-side DRR Attribution
    // =========================================================================
    // Replicates the exact math from env/reward.py:
    //   _get_required_fire(level, weather)
    //   _get_required_ambulance(level)
    // to compute Dispatch-Requirement Ratio per zone per resource type.
    // =========================================================================
    const SaliencyEngine = {
        /** Required fire units — exact copy of env/reward.py logic. */
        _requiredFire(fireLevel, weather) {
            const base = { catastrophic: 5, high: 3, medium: 2, low: 1, none: 0 };
            const req = base[fireLevel] || 0;
            if (req === 0) return 0;
            const weatherMod = { hurricane: 2, storm: 1, clear: 0 };
            return req + (weatherMod[weather] || 0);
        },

        /** Required ambulances — exact copy of env/reward.py logic. */
        _requiredAmb(patientLevel) {
            const base = { critical: 3, moderate: 1, fatal: 0, none: 0 };
            return base[patientLevel] || 0;
        },

        /**
         * Compute saliency for all zones given pre-step observation + action.
         * Returns: { zoneName: { fire: {sent, req, drr}, medical: {...}, traffic: {...} } }
         */
        compute(obsBeforeStep, allocations, weather) {
            const result = {};
            if (!obsBeforeStep || !obsBeforeStep.zones) return result;

            for (const [zoneId, zoneState] of Object.entries(obsBeforeStep.zones)) {
                const alloc = allocations[zoneId] || { dispatch_fire: 0, dispatch_ambulance: 0, control_traffic: false };

                const fireReq = this._requiredFire(zoneState.fire, weather);
                const ambReq = this._requiredAmb(zoneState.patient);
                const trafficNeed = (zoneState.traffic === 'heavy' || zoneState.traffic === 'gridlock') ? 1 : 0;

                const fireSent = alloc.dispatch_fire || 0;
                const ambSent = alloc.dispatch_ambulance || 0;
                const trafficSent = alloc.control_traffic ? 1 : 0;

                result[zoneId] = {
                    fire: {
                        sent: fireSent,
                        req: fireReq,
                        drr: fireReq > 0 ? fireSent / fireReq : (fireSent > 0 ? Infinity : 0)
                    },
                    medical: {
                        sent: ambSent,
                        req: ambReq,
                        drr: ambReq > 0 ? ambSent / ambReq : (ambSent > 0 ? Infinity : 0)
                    },
                    traffic: {
                        sent: trafficSent,
                        req: trafficNeed,
                        drr: trafficNeed > 0 ? trafficSent / trafficNeed : (trafficSent > 0 ? Infinity : 0)
                    }
                };
            }
            return result;
        },

        /** Render saliency into the panel. */
        render(saliency) {
            const container = document.getElementById('saliency-content');
            if (!saliency || Object.keys(saliency).length === 0) {
                container.innerHTML = '<div class="zone-placeholder">Send a dispatch to see feature attribution analysis</div>';
                return;
            }

            container.innerHTML = '';
            for (const [zoneId, attrs] of Object.entries(saliency)) {
                const card = document.createElement('div');
                card.className = 'sal-zone';
                card.innerHTML = `
                    <div class="sal-zone-name">${zoneId}</div>
                    ${this._barRow('🔥 Fire', attrs.fire)}
                    ${this._barRow('🏥 Med', attrs.medical)}
                    ${this._barRow('🚔 Trf', attrs.traffic)}
                `;
                container.appendChild(card);
            }
        },

        _barRow(label, attr) {
            const { sent, req, drr } = attr;
            let cls = 'zero';
            let pct = 0;

            if (req === 0 && sent === 0) {
                cls = 'zero';
                pct = 0;
            } else if (req === 0 && sent > 0) {
                cls = 'waste';
                pct = 100;
            } else if (drr >= 0.95 && drr <= 1.05) {
                cls = 'perfect';
                pct = 100;
            } else if (drr > 1.05) {
                cls = 'over';
                pct = Math.min(100, drr * 50);
            } else {
                cls = 'under';
                pct = Math.max(5, drr * 100);
            }

            const drrText = req === 0 ? (sent > 0 ? '∞' : '—') : drr.toFixed(2) + 'x';

            return `
                <div class="sal-bar-row">
                    <span class="sal-bar-label">${label}</span>
                    <div class="sal-bar-track">
                        <div class="sal-bar-fill ${cls}" style="width:${pct}%"></div>
                    </div>
                    <span class="sal-bar-value">${sent}/${req} ${drrText}</span>
                </div>
            `;
        }
    };

    // =========================================================================
    // 3. HeatmapRenderer — Zone Severity Visualization
    // =========================================================================
    const HeatmapRenderer = {
        /** Compute the maximum severity class for a zone based on all hazards. */
        _maxSeverity(zone) {
            const fireRank = { none: 0, low: 1, medium: 2, high: 3, catastrophic: 4 };
            const patientRank = { none: 0, moderate: 1, critical: 3, fatal: 4 };
            const trafficRank = { low: 0, heavy: 1, gridlock: 2 };

            const f = fireRank[zone.fire] || 0;
            const p = patientRank[zone.patient] || 0;
            const t = trafficRank[zone.traffic] || 0;
            const maxR = Math.max(f, p, t);

            if (maxR === 0) return 'none';
            if (maxR === 1) return 'low';
            if (maxR === 2) return 'medium';
            if (maxR === 3) return 'high';
            return 'catastrophic';
        },

        /** Build hazard indicator strings for a zone. */
        _indicators(zone) {
            const parts = [];
            if (zone.fire && zone.fire !== 'none') {
                parts.push(`<span class="zone-indicator"><span class="emoji">🔥</span> ${zone.fire}</span>`);
            }
            if (zone.patient && zone.patient !== 'none') {
                const emoji = zone.patient === 'fatal' ? '💀' : '🏥';
                parts.push(`<span class="zone-indicator"><span class="emoji">${emoji}</span> ${zone.patient}</span>`);
            }
            if (zone.traffic && zone.traffic !== 'low') {
                parts.push(`<span class="zone-indicator"><span class="emoji">🚗</span> ${zone.traffic}</span>`);
            }
            if (parts.length === 0) {
                parts.push(`<span class="zone-indicator" style="color:var(--severity-none)">✓ Clear</span>`);
            }
            return parts.join('');
        },

        /** Check if a zone is fully resolved (all hazards at baseline). */
        _isResolved(zone) {
            return (zone.fire === 'none' || !zone.fire) &&
                   (zone.patient === 'none' || !zone.patient) &&
                   (zone.traffic === 'low' || !zone.traffic);
        },

        /** Render all zones into the grid. */
        render(zones) {
            const grid = document.getElementById('zone-grid');
            if (!zones || Object.keys(zones).length === 0) {
                grid.innerHTML = '<div class="zone-placeholder">No zone data</div>';
                return;
            }

            grid.innerHTML = '';
            for (const [zoneId, zoneState] of Object.entries(zones)) {
                const sev = this._maxSeverity(zoneState);
                const card = document.createElement('div');
                card.className = `zone-card sev-${sev}`;
                card.id = `zone-${zoneId.toLowerCase().replace(/\s+/g, '-')}`;
                card.innerHTML = `
                    <div class="zone-name">${zoneId}</div>
                    <div class="zone-indicators">${this._indicators(zoneState)}</div>
                `;
                grid.appendChild(card);

                // Track resolution timeline
                if (this._isResolved(zoneState) && !zoneTimeline[zoneId]) {
                    zoneTimeline[zoneId] = stepCount;
                }
            }
        }
    };

    // =========================================================================
    // 4. GaugeRenderer — Resource Pool Visualization
    // =========================================================================
    const GaugeRenderer = {
        render(idle, busy) {
            if (!idle || !busy) return;

            this._updateGauge('fire', idle.fire_units, busy.fire_units);
            this._updateGauge('amb', idle.ambulances, busy.ambulances);
            this._updateGauge('pol', idle.police, busy.police);
        },

        _updateGauge(type, idleCount, busyCount) {
            const total = idleCount + busyCount;
            const pct = total > 0 ? (idleCount / total) * 100 : 0;

            const fill = document.getElementById(`gauge-${type}`);
            const idleEl = document.getElementById(`${type}-idle`);
            const busyEl = document.getElementById(`${type}-busy`);

            if (fill) fill.style.width = `${pct}%`;
            if (idleEl) idleEl.textContent = idleCount;
            if (busyEl) busyEl.textContent = busyCount;
        }
    };

    // =========================================================================
    // 5. SparklineRenderer — Reward History Chart (Canvas 2D)
    // =========================================================================
    const SparklineRenderer = {
        render(rewardsArr) {
            const canvas = document.getElementById('reward-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');

            // Handle high-DPI displays
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * (window.devicePixelRatio || 1);
            canvas.height = rect.height * (window.devicePixelRatio || 1);
            ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
            const W = rect.width;
            const H = rect.height;

            // Clear
            ctx.clearRect(0, 0, W, H);

            if (rewardsArr.length === 0) {
                ctx.fillStyle = 'rgba(255,255,255,0.1)';
                ctx.font = '12px sans-serif';
                ctx.fillText('No data yet', W / 2 - 30, H / 2);
                return;
            }

            const padding = { top: 12, bottom: 12, left: 12, right: 12 };
            const plotW = W - padding.left - padding.right;
            const plotH = H - padding.top - padding.bottom;

            const minR = Math.min(0, ...rewardsArr);
            const maxR = Math.max(0, ...rewardsArr);
            const range = maxR - minR || 1;

            // Zero line
            const zeroY = padding.top + plotH * (1 - (0 - minR) / range);
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(padding.left, zeroY);
            ctx.lineTo(W - padding.right, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Plot segments (green for positive, red for negative)
            const xStep = rewardsArr.length > 1 ? plotW / (rewardsArr.length - 1) : plotW;

            for (let i = 1; i < rewardsArr.length; i++) {
                const x0 = padding.left + (i - 1) * xStep;
                const x1 = padding.left + i * xStep;
                const y0 = padding.top + plotH * (1 - (rewardsArr[i - 1] - minR) / range);
                const y1 = padding.top + plotH * (1 - (rewardsArr[i] - minR) / range);

                const color = rewardsArr[i] >= 0 ? '#22c55e' : '#ef4444';
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.stroke();
            }

            // Dots
            for (let i = 0; i < rewardsArr.length; i++) {
                const x = padding.left + i * xStep;
                const y = padding.top + plotH * (1 - (rewardsArr[i] - minR) / range);
                const c = rewardsArr[i] >= 0 ? '#22c55e' : '#ef4444';

                ctx.fillStyle = c;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    };

    // =========================================================================
    // 6. DispatchPanel — Manual Dispatch Interface
    // =========================================================================
    const DispatchPanel = {
        /** Render slider controls for each zone. */
        render(zones, idle) {
            const container = document.getElementById('dispatch-zones');
            if (!zones) {
                container.innerHTML = '<div class="zone-placeholder">No zones</div>';
                return;
            }

            container.innerHTML = '';
            const maxFire = idle ? idle.fire_units : 10;
            const maxAmb = idle ? idle.ambulances : 10;

            for (const zoneId of Object.keys(zones)) {
                const safeId = zoneId.toLowerCase().replace(/\s+/g, '-');
                const row = document.createElement('div');
                row.className = 'dispatch-zone-row';
                row.innerHTML = `
                    <div class="dz-label">${zoneId}</div>
                    <div class="dz-slider-group">
                        <div class="dz-slider-label">
                            <span>🚒 Fire</span>
                            <span id="val-fire-${safeId}">0</span>
                        </div>
                        <input type="range" class="dz-slider fire-slider"
                               id="slider-fire-${safeId}" data-zone="${zoneId}" data-type="fire"
                               min="0" max="${maxFire}" value="0">
                    </div>
                    <div class="dz-slider-group">
                        <div class="dz-slider-label">
                            <span>🏥 Amb</span>
                            <span id="val-amb-${safeId}">0</span>
                        </div>
                        <input type="range" class="dz-slider amb-slider"
                               id="slider-amb-${safeId}" data-zone="${zoneId}" data-type="amb"
                               min="0" max="${maxAmb}" value="0">
                    </div>
                    <div class="dz-checkbox-group">
                        <input type="checkbox" class="dz-checkbox"
                               id="chk-traffic-${safeId}" data-zone="${zoneId}">
                        <label for="chk-traffic-${safeId}">🚔</label>
                    </div>
                `;
                container.appendChild(row);
            }

            // Bind slider value displays
            container.querySelectorAll('.dz-slider').forEach(slider => {
                slider.addEventListener('input', () => {
                    const zone = slider.dataset.zone.toLowerCase().replace(/\s+/g, '-');
                    const type = slider.dataset.type;
                    const valEl = document.getElementById(`val-${type}-${zone}`);
                    if (valEl) valEl.textContent = slider.value;
                });
            });
        },

        /** Read current slider/checkbox values into an allocations dict. */
        readAllocations() {
            const allocations = {};
            const container = document.getElementById('dispatch-zones');
            if (!container) return allocations;

            const sliders = container.querySelectorAll('.dz-slider');
            const checkboxes = container.querySelectorAll('.dz-checkbox');

            // Accumulate by zone
            sliders.forEach(s => {
                const zone = s.dataset.zone;
                if (!allocations[zone]) {
                    allocations[zone] = { dispatch_fire: 0, dispatch_ambulance: 0, control_traffic: false };
                }
                if (s.dataset.type === 'fire') {
                    allocations[zone].dispatch_fire = parseInt(s.value, 10);
                } else if (s.dataset.type === 'amb') {
                    allocations[zone].dispatch_ambulance = parseInt(s.value, 10);
                }
            });

            checkboxes.forEach(c => {
                const zone = c.dataset.zone;
                if (!allocations[zone]) {
                    allocations[zone] = { dispatch_fire: 0, dispatch_ambulance: 0, control_traffic: false };
                }
                allocations[zone].control_traffic = c.checked;
            });

            return allocations;
        },

        /** Update slider maximums when idle resources change. */
        updateMaximums(idle) {
            if (!idle) return;
            document.querySelectorAll('.dz-slider.fire-slider').forEach(s => {
                s.max = idle.fire_units;
                if (parseInt(s.value) > idle.fire_units) s.value = idle.fire_units;
            });
            document.querySelectorAll('.dz-slider.amb-slider').forEach(s => {
                s.max = idle.ambulances;
                if (parseInt(s.value) > idle.ambulances) s.value = idle.ambulances;
            });
        }
    };

    // =========================================================================
    // 7. ActionLog — Scrolling Event Log
    // =========================================================================
    const ActionLog = {
        _container: null,

        init() {
            this._container = document.getElementById('log-container');
        },

        append(message, type = 'info') {
            if (!this._container) this.init();
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.textContent = message;
            this._container.appendChild(entry);
            // Auto-scroll to bottom
            this._container.scrollTop = this._container.scrollHeight;
        },

        clear() {
            if (!this._container) this.init();
            this._container.innerHTML = '';
        }
    };

    // =========================================================================
    // 8. EpisodeSummary — Modal with grading breakdown
    // =========================================================================
    const EpisodeSummary = {
        show(info, rewardsArr, stepsUsed, taskLevel) {
            const modal = document.getElementById('episode-modal');
            const body = document.getElementById('modal-body');

            const score = info.score || 0;
            const efficiency = info.efficiency || 0;
            const threshold = 0.50;
            const passed = score >= threshold;

            const cumReward = rewardsArr.reduce((a, b) => a + b, 0);
            const peak = rewardsArr.length > 0 ? Math.max(...rewardsArr) : 0;
            const trough = rewardsArr.length > 0 ? Math.min(...rewardsArr) : 0;

            const maxSteps = taskLevel === 1 ? 10 : taskLevel === 2 ? 15 : 25;

            // Zone resolution timeline text
            let timelineText = '';
            const resolved = Object.entries(zoneTimeline);
            if (resolved.length > 0) {
                timelineText = resolved.map(([z, s]) => `${z}: step ${s}`).join(', ');
            } else {
                timelineText = 'No zones fully resolved';
            }

            body.innerHTML = `
                <div class="modal-stat ${passed ? 'pass' : 'fail'}">
                    <span class="modal-stat-label">Final Score</span>
                    <span class="modal-stat-value">${score.toFixed(4)} ${passed ? '✅ PASS' : '❌ FAIL'}</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-label">Threshold</span>
                    <span class="modal-stat-value">≥ ${threshold.toFixed(2)}</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-label">Efficiency</span>
                    <span class="modal-stat-value">${efficiency.toFixed(4)}</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-label">Steps Used</span>
                    <span class="modal-stat-value">${stepsUsed} / ${maxSteps}</span>
                </div>
                <div class="modal-stat ${cumReward >= 0 ? 'pass' : 'warn'}">
                    <span class="modal-stat-label">Cumulative Reward</span>
                    <span class="modal-stat-value">${cumReward.toFixed(2)}</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-label">Peak / Trough</span>
                    <span class="modal-stat-value">${peak.toFixed(2)} / ${trough.toFixed(2)}</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-label">Zone Resolution</span>
                    <span class="modal-stat-value" style="font-size:0.75rem">${timelineText}</span>
                </div>
            `;

            modal.style.display = 'flex';
        },

        hide() {
            document.getElementById('episode-modal').style.display = 'none';
        }
    };

    // =========================================================================
    // 9. ConnectionMonitor — Health polling with status dot
    // =========================================================================
    const ConnectionMonitor = {
        _intervalId: null,

        start() {
            this._check();
            this._intervalId = setInterval(() => this._check(), 10000);
        },

        async _check() {
            const dot = document.getElementById('conn-dot');
            const result = await CrisisClient.getHealth();
            if (result.error) {
                dot.className = 'conn-dot offline';
                dot.title = 'Server offline: ' + result.error;
            } else {
                dot.className = 'conn-dot online';
                dot.title = `Online | Sessions: ${result.active_sessions}/${result.max_sessions} | RAM: ${result.memory_rss_mb}MB`;
            }
        }
    };

    // =========================================================================
    // 10. AutoPlayEngine — Intelligent auto-dispatch
    // =========================================================================
    const AutoPlayEngine = {
        /** Compute a reasonable auto-dispatch based on zone hazard levels. */
        computeSmartDispatch(obs) {
            if (!obs || !obs.zones) return {};
            const allocations = {};
            const idle = obs.idle_resources || { fire_units: 0, ambulances: 0, police: 0 };
            let fireLeft = idle.fire_units;
            let ambLeft = idle.ambulances;
            const weather = obs.weather || 'clear';

            // Sort zones by severity (worst first) for resource priority
            const zoneEntries = Object.entries(obs.zones);
            const severityOrder = { catastrophic: 5, high: 4, critical: 4, fatal: 3, medium: 3, heavy: 2, moderate: 2, gridlock: 2, low: 1, none: 0 };
            zoneEntries.sort((a, b) => {
                const sa = Math.max(severityOrder[a[1].fire] || 0, severityOrder[a[1].patient] || 0);
                const sb = Math.max(severityOrder[b[1].fire] || 0, severityOrder[b[1].patient] || 0);
                return sb - sa;
            });

            for (const [zoneId, zoneState] of zoneEntries) {
                const fireReq = SaliencyEngine._requiredFire(zoneState.fire, weather);
                const ambReq = SaliencyEngine._requiredAmb(zoneState.patient);
                const needTraffic = (zoneState.traffic === 'heavy' || zoneState.traffic === 'gridlock');

                const fireSend = Math.min(fireReq, fireLeft);
                const ambSend = Math.min(ambReq, ambLeft);

                fireLeft -= fireSend;
                ambLeft -= ambSend;

                allocations[zoneId] = {
                    dispatch_fire: fireSend,
                    dispatch_ambulance: ambSend,
                    control_traffic: needTraffic
                };
            }

            return allocations;
        },

        /** Generate a smart broadcast message. */
        computeBroadcast(obs) {
            if (!obs || !obs.zones) return '';
            const alerts = [];
            for (const [zoneId, z] of Object.entries(obs.zones)) {
                if (z.fire === 'high' || z.fire === 'catastrophic') {
                    alerts.push(`ALERT: ${zoneId} has a ${z.fire} fire. All residents must evacuate immediately.`);
                }
                if (z.patient === 'critical') {
                    alerts.push(`ALERT: ${zoneId} has critical patients. Shelter in place and await medical teams.`);
                }
            }
            return alerts.join(' ');
        },

        start() {
            if (autoPlayInterval) return;
            const btn = document.getElementById('btn-autoplay');
            btn.classList.add('playing');
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                Stop <kbd>A</kbd>
            `;
            setStatus('autoplay');
            ActionLog.append('[Auto-Play] Started — AI dispatch active', 'system');

            autoPlayInterval = setInterval(async () => {
                if (autoPlayBusy || episodeDone || !sessionActive) {
                    this.stop();
                    return;
                }
                autoPlayBusy = true;

                const allocations = this.computeSmartDispatch(currentObs);
                const broadcast = this.computeBroadcast(currentObs);
                document.getElementById('broadcast-input').value = broadcast;

                await executeStep(allocations, broadcast);

                autoPlayBusy = false;

                if (episodeDone) {
                    this.stop();
                }
            }, 2000);
        },

        stop() {
            if (autoPlayInterval) {
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
            }
            autoPlayBusy = false;
            const btn = document.getElementById('btn-autoplay');
            btn.classList.remove('playing');
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/><line x1="19" y1="3" x2="19" y2="21"/></svg>
                Auto-Play <kbd>A</kbd>
            `;
            if (!episodeDone && sessionActive) {
                setStatus('active');
            }
        },

        toggle() {
            if (autoPlayInterval) {
                this.stop();
            } else if (sessionActive && !episodeDone) {
                this.start();
            }
        }
    };

    // =========================================================================
    // Weather emoji helper
    // =========================================================================
    function weatherEmoji(w) {
        if (!w) return '—';
        const map = { clear: '☀️ Clear', storm: '⛈️ Storm', hurricane: '🌀 Hurricane' };
        return map[w] || w;
    }

    // =========================================================================
    // Dynamic Title
    // =========================================================================
    function updateTitle() {
        if (!sessionActive) {
            document.title = 'Crisis Dashboard | Adaptive Crisis Env';
            return;
        }
        const maxSteps = selectedTask === 1 ? 10 : selectedTask === 2 ? 15 : 25;
        const scoreText = lastScore.toFixed(3);
        if (episodeDone) {
            document.title = `✅ Score: ${scoreText} | Crisis Dashboard`;
        } else {
            document.title = `Step ${stepCount}/${maxSteps} | Score: ${scoreText} | Crisis Dashboard`;
        }
    }

    // =========================================================================
    // UI State Updates
    // =========================================================================
    function updateSessionInfo(obs, score, efficiency) {
        document.getElementById('weather-display').textContent = weatherEmoji(obs.weather);
        document.getElementById('score-display').textContent = score.toFixed(3);
        document.getElementById('spark-score').textContent = score.toFixed(3);
        document.getElementById('spark-efficiency').textContent = efficiency.toFixed(3);

        const cumR = rewards.reduce((a, b) => a + b, 0);
        document.getElementById('cumulative-reward').textContent = cumR.toFixed(2);

        const maxSteps = selectedTask === 1 ? 10 : selectedTask === 2 ? 15 : 25;
        document.getElementById('step-counter').textContent = `${stepCount} / ${maxSteps}`;

        updateTitle();
    }

    function setStatus(status) {
        const el = document.getElementById('status-indicator');
        el.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        el.className = `info-value status-${status}`;
    }

    // =========================================================================
    // Core Step Execution (shared by manual and auto-play)
    // =========================================================================
    async function executeStep(allocations, broadcastMsg) {
        // Capture pre-step observation for saliency
        const obsBeforeStep = currentObs;
        const weather = currentObs ? currentObs.weather : 'clear';

        // Summarize dispatch for log
        const dispSummary = Object.entries(allocations)
            .map(([z, d]) => `${z}:F${d.dispatch_fire}/A${d.dispatch_ambulance}${d.control_traffic ? '/P' : ''}`)
            .join(' ');
        ActionLog.append(`[Step ${stepCount + 1}] Dispatching: ${dispSummary}`, 'info');

        const result = await CrisisClient.step(allocations, broadcastMsg);

        if (result.error) {
            ActionLog.append(`ERROR: ${result.error}`, 'error');
            return;
        }

        // Process step response
        stepCount++;
        const reward = result.reward;
        rewards.push(reward);

        currentObs = result.observation;
        const info = result.info || {};
        lastInfo = info;
        lastScore = info.score || lastScore;
        lastEfficiency = info.efficiency || lastEfficiency;

        // Compute & render saliency
        const saliency = SaliencyEngine.compute(obsBeforeStep, allocations, weather);
        SaliencyEngine.render(saliency);

        // Update all renderers
        HeatmapRenderer.render(currentObs.zones);
        GaugeRenderer.render(currentObs.idle_resources, currentObs.busy_resources);
        DispatchPanel.updateMaximums(currentObs.idle_resources);
        SparklineRenderer.render(rewards);
        updateSessionInfo(currentObs, lastScore, lastEfficiency);

        document.getElementById('spark-last-reward').textContent = reward.toFixed(2);

        const rewardType = reward >= 0 ? 'positive' : 'negative';
        ActionLog.append(
            `[Step ${stepCount}] reward=${reward.toFixed(2)} | score=${lastScore.toFixed(3)} | done=${result.done}`,
            rewardType
        );

        if (result.done) {
            episodeDone = true;
            setStatus('done');
            document.getElementById('btn-step').disabled = true;
            document.getElementById('btn-autoplay').disabled = true;
            ActionLog.append(
                `Episode complete. Final score: ${lastScore.toFixed(3)} | Steps: ${stepCount}`,
                'system'
            );
            // Show episode summary modal
            EpisodeSummary.show(info, rewards, stepCount, selectedTask);
        }
    }

    // =========================================================================
    // Event Handlers
    // =========================================================================

    /** RESET — initialize fresh episode */
    async function handleReset() {
        // Stop auto-play if running
        AutoPlayEngine.stop();

        const seed = document.getElementById('seed-input').value;
        const btnReset = document.getElementById('btn-reset');
        const btnStep = document.getElementById('btn-step');
        const btnAutoplay = document.getElementById('btn-autoplay');

        btnReset.disabled = true;
        setStatus('active');
        ActionLog.clear();
        ActionLog.append(`Resetting environment: Task=${selectedTask}, Seed=${seed || 'random'}`, 'system');

        const result = await CrisisClient.reset(selectedTask, seed);

        if (result.error) {
            ActionLog.append(`ERROR: ${result.error}`, 'error');
            setStatus('error');
            btnReset.disabled = false;
            return;
        }

        // Parse observation
        currentObs = result;
        stepCount = 0;
        rewards = [];
        sessionActive = true;
        episodeDone = false;
        lastScore = 0;
        lastEfficiency = 0;
        lastInfo = {};
        zoneTimeline = {};

        // Render all components
        HeatmapRenderer.render(currentObs.zones);
        GaugeRenderer.render(currentObs.idle_resources, currentObs.busy_resources);
        DispatchPanel.render(currentObs.zones, currentObs.idle_resources);
        SparklineRenderer.render(rewards);
        SaliencyEngine.render(null);
        updateSessionInfo(currentObs, 0, 0);

        ActionLog.append(`Environment ready. ${Object.keys(currentObs.zones).length} zones active. Weather: ${currentObs.weather}`, 'info');

        btnReset.disabled = false;
        btnStep.disabled = false;
        btnAutoplay.disabled = false;

        document.getElementById('spark-last-reward').textContent = '—';
    }

    /** STEP — manual dispatch */
    async function handleStep() {
        if (!sessionActive || episodeDone) return;

        const btnStep = document.getElementById('btn-step');
        btnStep.disabled = true;

        const allocations = DispatchPanel.readAllocations();
        const broadcastMsg = document.getElementById('broadcast-input').value;

        await executeStep(allocations, broadcastMsg);

        if (!episodeDone) {
            btnStep.disabled = false;
        }
    }

    // =========================================================================
    // 11. Keyboard Shortcuts
    // =========================================================================
    function initKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't capture when typing in input/textarea
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch (e.key.toLowerCase()) {
                case 'r':
                    e.preventDefault();
                    handleReset();
                    break;
                case ' ':
                    e.preventDefault();
                    if (!autoPlayInterval) handleStep();
                    break;
                case '1':
                    e.preventDefault();
                    selectTask(1);
                    break;
                case '2':
                    e.preventDefault();
                    selectTask(2);
                    break;
                case '3':
                    e.preventDefault();
                    selectTask(3);
                    break;
                case 'a':
                    e.preventDefault();
                    AutoPlayEngine.toggle();
                    break;
                case 'escape':
                    EpisodeSummary.hide();
                    break;
            }
        });
    }

    function selectTask(taskNum) {
        document.querySelectorAll('.btn-task').forEach(b => b.classList.remove('active'));
        const btn = document.getElementById(`btn-task-${taskNum}`);
        if (btn) btn.classList.add('active');
        selectedTask = taskNum;
    }

    // =========================================================================
    // Initialization
    // =========================================================================
    function init() {
        ActionLog.init();

        // Task selector
        document.querySelectorAll('.btn-task').forEach(btn => {
            btn.addEventListener('click', () => {
                selectTask(parseInt(btn.dataset.task, 10));
            });
        });

        // Reset button
        document.getElementById('btn-reset').addEventListener('click', handleReset);

        // Step button
        document.getElementById('btn-step').addEventListener('click', handleStep);

        // Auto-play button
        document.getElementById('btn-autoplay').addEventListener('click', () => AutoPlayEngine.toggle());

        // Modal close
        document.getElementById('modal-close').addEventListener('click', () => {
            EpisodeSummary.hide();
            handleReset();
        });

        // Click outside modal to close
        document.getElementById('episode-modal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('episode-modal')) {
                EpisodeSummary.hide();
            }
        });

        // Keyboard shortcuts
        initKeyboardShortcuts();

        // Connection monitor
        ConnectionMonitor.start();

        // Initial health check log
        CrisisClient.getHealth().then(health => {
            if (health.error) {
                ActionLog.append(`Server health check failed: ${health.error}`, 'error');
                setStatus('error');
            } else {
                ActionLog.append(
                    `Server online. Sessions: ${health.active_sessions}/${health.max_sessions} | Memory: ${health.memory_rss_mb}MB`,
                    'info'
                );
            }
        });
    }

    // Boot when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
