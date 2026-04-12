# Smart City Meta-Orchestration: Analytical Architecture
### Meta AI Hackathon Capstone - Technical Whitepaper

To secure the $7,000 Grand Prize, an application must transcend basic code and prove a deep understanding of mathematical Reinforcement Learning. This document outlines the rigorous academic framework driving this orchestration environment.

## 1. Reinforcement Learning Environment (MDP Architecture)
The core `environment.py` is structured upon a fully bounded Markov Decision Process (MDP).
- **States ($S$)**: Represented via a highly structured multidimensional array `(F, P, T, W, R_idle, R_busy)` denoting Fire Severity, Patient Triaging, Traffic Flow, Weather Chaos, and Resource Pools.
- **Actions ($A$)**: A dispatch vector simulating resource deployment subject strictly to `idle` limitations.
- **State Transition Physics (`step()` & `reset()`)**: The environment mimics continuous temporal mechanics by embedding a `tick_deployment()` array. When an AI dispatches a resource, it is dynamically shifted to $R_{busy}$. A hidden dynamic latency algorithm determines the cooldown steps required to return to $R_{idle}$, integrating friction variables such as `STORM` perturbations and `GRIDLOCK` delays.

## 2. Advanced Transition Dynamics (Non-Stationarity)
To satisfy Phase 3 "Hard Task" requirements, the environment implements non-stationary mechanics:
- **Topology**: Five-zone topology (Downtown, Suburbs, Industrial, Harbor, Residential) using a **State-Space Circular Ring** adjacency map.
- **Resource Depletion**: Fire units decay over time ($N_{fire, t} = N_{fire, 0} - \lfloor t/4 \rfloor$), forcing optimal early sequencing.
- **Inter-Zone Cascading (Stochastic Spread)**: High-severity incidents ($\xi_j > \tau=3$) spread to neighbors with probability $P$:
  $$P(\text{spread}) = \beta \cdot \frac{\xi_j - \tau}{\xi_{max} - \tau}$$
  where $\beta=0.4$ (Cascade Coefficient) and $\xi_{max}=4$.

## 3. Exploit-Resistant Reward Engineering
The `reward.py` and `grader.py` modules implement a multi-layered evaluation framework:
- **Action Diversity Monitor ($\mathcal{D}$)**: Calculated as the ratio of unique action hashes to total steps: $\mathcal{D} = |\{h(a_i)\}| / T$.
- **Monotony Penalty**: Modulates the final score $S$ if diversity falls below $\Gamma=0.3$:
  $$S' = S \cdot \min\left(1.0, \frac{\mathcal{D}}{\Gamma}\right)$$
- **Loop Detection Penalty**: A sliding window ($k=3$) penalizes repeated actions ($\delta=3.0$).

## 4. Adaptive Curriculum Escalation
The environment dynamically adjusts difficulty in-episode to probe agent resilience:
- **Trigger**: When the rolling 5-step reward window mean $\bar{W} > 0.7$.
- **Escalation ($\mathcal{E}$)**: Applies a 20% resource reduction ($Resources \leftarrow \lfloor 0.8 \times Resources \rfloor$) and injects a new crisis event in a clear zone.

## 5. Engineering Sophistication
- **Session Isolation**: UUID-based session store with `asyncio.Lock` prevents state bleeding during concurrent evaluations.
- **Graceful Degradation**: Dual-retry logic with fallback to a Scenario Fallback Pool (Static JSON) ensures high system availability.
- **Observability**: Real-time `/health` and `/metrics` endpoints for production monitoring.

---
*Built to assert total technical dominance in the Meta AI Hackathon.*
