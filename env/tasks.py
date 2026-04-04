"""
env/tasks.py
============
Task registry for the Adaptive Crisis Management Environment.

Determinism Guarantee
---------------------
Every ``Task.generate_initial_observation(seed=N)`` call with the **same**
integer seed must produce the **identical** ``Observation`` object, byte for
byte, every single time — regardless of the Python interpreter state, import
order, or prior calls.

Implementation contract
-----------------------
* Each ``Task`` subclass receives a ``seed`` parameter in
  ``generate_initial_observation``.
* The method must construct a ``random.Random(seed)`` instance internally
  and use **only** that seeded instance for any random choice.
* It must **not** call the module-level ``random.random()`` or
  ``numpy.random.*`` functions directly (those are global and would break
  isolation).
* The ``TaskLevel`` field on the returned ``Observation`` tracks difficulty.

Current task roster
-------------------
Task 1 — EasyTask   : "Single-Zone Emergency"
Task 2 — MediumTask : "Multi-Zone Weather Chaos"
Task 3 — HardTask   : "City-Wide Meta Triage"
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np  # noqa: F401  (imported for completeness; seeding is local)

from env.models import (
    FireLevel,
    Observation,
    PatientLevel,
    ResourcePool,
    TaskLevel,
    TrafficLevel,
    WeatherCondition,
    ZoneState,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Task:
    """Abstract base for all task definitions."""

    task_id: int = 0
    name: str = "unnamed"

    def generate_initial_observation(self, seed: Optional[int] = None) -> Observation:
        """Return the deterministic starting observation for this task.

        Args:
            seed: Integer seed that fully determines every random choice made
                  during incident generation.  The *same* seed must always
                  produce the *same* ``Observation``.

        Returns:
            A freshly constructed ``Observation`` for the beginning of an
            episode under this task.

        Raises:
            NotImplementedError: Must be overridden by concrete subclasses.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1 — Easy
# ---------------------------------------------------------------------------

class EasyTask(Task):
    """Single-Zone Emergency (easy difficulty).

    Deterministic generation
    ------------------------
    When ``seed`` is provided a seeded ``random.Random`` instance draws:

    * Whether a second zone might receive a minor traffic incident (the draw
      keeps the scenario deterministic across seeds).

    The core Downtown fire is always MEDIUM in this task (fixed, not random),
    so the scenario structure is constant.  The RNG is only used to determine
    optional minor variance in the other two zones so that the seed contract
    is honoured even when the outcome is predictable.
    """

    task_id = 1
    name = "Single-Zone Emergency"

    # Fixed resource pool for this difficulty.
    _IDLE_FIRE   = 5
    _IDLE_AMB    = 5
    _IDLE_POLICE = 3
    _MAX_STEPS   = 8

    def generate_initial_observation(self, seed: Optional[int] = None) -> Observation:
        """Generate the deterministic Task 1 starting state.

        The RNG is seeded locally — it does NOT touch the global
        ``random`` module state, ensuring full isolation.

        Args:
            seed: Reproducibility seed.  ``None`` → non-deterministic.

        Returns:
            Observation for a single Downtown fire under clear skies.
        """
        # ---------- Seeded local RNG (isolation!) --------------------------
        rng = random.Random(seed)  # Completely isolated — no global side-effects.

        # ---------- Incident generation (seed-locked) ----------------------
        # Downtown always has a MEDIUM fire in Task 1.
        downtown_fire = FireLevel.MEDIUM

        # Optional minor Suburbs event: 30 % probability of HEAVY traffic.
        # The outcome is determined entirely by the seed.
        suburbs_traffic = (
            TrafficLevel.HEAVY if rng.random() < 0.30 else TrafficLevel.LOW
        )

        zones = {
            "Downtown": ZoneState(
                fire=downtown_fire,
                patient=PatientLevel.NONE,
                traffic=TrafficLevel.LOW,
            ),
            "Suburbs": ZoneState(
                fire=FireLevel.NONE,
                patient=PatientLevel.NONE,
                traffic=suburbs_traffic,
            ),
            "Industrial": ZoneState(
                fire=FireLevel.NONE,
                patient=PatientLevel.NONE,
                traffic=TrafficLevel.LOW,
            ),
        }

        return Observation(
            weather=WeatherCondition.CLEAR,
            zones=zones,
            idle_resources=ResourcePool(
                fire_units=self._IDLE_FIRE,
                ambulances=self._IDLE_AMB,
                police=self._IDLE_POLICE,
            ),
            busy_resources=ResourcePool(fire_units=0, ambulances=0, police=0),
            step=0,
            max_steps=self._MAX_STEPS,
            task_level=TaskLevel.EASY,
        )


# ---------------------------------------------------------------------------
# Task 2 — Medium
# ---------------------------------------------------------------------------

class MediumTask(Task):
    """Multi-Zone Weather Chaos (medium difficulty).

    Deterministic generation
    ------------------------
    The Suburbs fire severity and Downtown patient triage level are drawn from
    a fixed weighted pool locked to the seed.  This introduces meaningful
    scenario variety while remaining 100 % reproducible.
    """

    task_id = 2
    name = "Multi-Zone Weather Chaos"

    _IDLE_FIRE   = 5
    _IDLE_AMB    = 3
    _IDLE_POLICE = 2
    _MAX_STEPS   = 10

    # Scenario pools (weights must sum to 1.0 within each group).
    _FIRE_POOL  = [FireLevel.MEDIUM, FireLevel.HIGH]
    _FIRE_WTS   = [0.50, 0.50]
    _PAT_POOL   = [PatientLevel.MODERATE, PatientLevel.CRITICAL]
    _PAT_WTS    = [0.60, 0.40]

    def generate_initial_observation(self, seed: Optional[int] = None) -> Observation:
        """Generate the deterministic Task 2 starting state.

        Args:
            seed: Reproducibility seed.  ``None`` → non-deterministic.

        Returns:
            Observation with multi-zone incidents under STORM weather.
        """
        rng = random.Random(seed)

        # Draw fire level from weighted pool.
        suburbs_fire: FireLevel = rng.choices(
            self._FIRE_POOL, weights=self._FIRE_WTS, k=1
        )[0]

        # Draw Downtown patient severity from weighted pool.
        downtown_patient: PatientLevel = rng.choices(
            self._PAT_POOL, weights=self._PAT_WTS, k=1
        )[0]

        zones = {
            "Downtown": ZoneState(
                fire=FireLevel.NONE,
                patient=downtown_patient,
                traffic=TrafficLevel.HEAVY,
            ),
            "Suburbs": ZoneState(
                fire=suburbs_fire,
                patient=PatientLevel.NONE,
                traffic=TrafficLevel.LOW,
            ),
            "Industrial": ZoneState(
                fire=FireLevel.NONE,
                patient=PatientLevel.NONE,
                traffic=TrafficLevel.LOW,
            ),
        }

        return Observation(
            weather=WeatherCondition.STORM,
            zones=zones,
            idle_resources=ResourcePool(
                fire_units=self._IDLE_FIRE,
                ambulances=self._IDLE_AMB,
                police=self._IDLE_POLICE,
            ),
            busy_resources=ResourcePool(),
            step=0,
            max_steps=self._MAX_STEPS,
            task_level=TaskLevel.MEDIUM,
        )


# ---------------------------------------------------------------------------
# Task 3 — Hard
# ---------------------------------------------------------------------------

class HardTask(Task):
    """City-Wide Meta Triage (hard difficulty).

    Deterministic generation
    ------------------------
    All three zones are seeded with randomised severities drawn from
    pre-defined pools.  The Industrial fire is always CATASTROPHIC, but the
    other zones use the RNG pool to introduce variety.
    """

    task_id = 3
    name = "City-Wide Meta Triage"

    _IDLE_FIRE   = 8
    _IDLE_AMB    = 4
    _IDLE_POLICE = 2
    _MAX_STEPS   = 12

    # Downtown fire options (seed-determined).
    _DT_FIRE_POOL = [FireLevel.HIGH, FireLevel.CATASTROPHIC]
    _DT_FIRE_WTS  = [0.60, 0.40]

    # Suburbs patient severity options.
    _SUB_PAT_POOL = [PatientLevel.CRITICAL, PatientLevel.MODERATE]
    _SUB_PAT_WTS  = [0.70, 0.30]

    def generate_initial_observation(self, seed: Optional[int] = None) -> Observation:
        """Generate the deterministic Task 3 starting state.

        Args:
            seed: Reproducibility seed.  ``None`` → non-deterministic.

        Returns:
            Observation with city-wide multi-zone incidents under HURRICANE.
        """
        rng = random.Random(seed)

        downtown_fire: FireLevel = rng.choices(
            self._DT_FIRE_POOL, weights=self._DT_FIRE_WTS, k=1
        )[0]

        suburbs_patient: PatientLevel = rng.choices(
            self._SUB_PAT_POOL, weights=self._SUB_PAT_WTS, k=1
        )[0]

        zones = {
            "Downtown": ZoneState(
                fire=downtown_fire,
                patient=PatientLevel.NONE,
                traffic=TrafficLevel.GRIDLOCK,
            ),
            "Suburbs": ZoneState(
                fire=FireLevel.NONE,
                patient=suburbs_patient,
                traffic=TrafficLevel.GRIDLOCK,
            ),
            "Industrial": ZoneState(
                fire=FireLevel.CATASTROPHIC,  # Always catastrophic in Task 3.
                patient=PatientLevel.NONE,
                traffic=TrafficLevel.LOW,
            ),
        }

        return Observation(
            weather=WeatherCondition.HURRICANE,
            zones=zones,
            idle_resources=ResourcePool(
                fire_units=self._IDLE_FIRE,
                ambulances=self._IDLE_AMB,
                police=self._IDLE_POLICE,
            ),
            busy_resources=ResourcePool(),
            step=0,
            max_steps=self._MAX_STEPS,
            task_level=TaskLevel.HARD,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_task(task_id: int) -> Task:
    """Return the ``Task`` instance corresponding to ``task_id``.

    Args:
        task_id: 1 (Easy), 2 (Medium), or 3 (Hard).

    Returns:
        A ``Task`` subclass instance ready for ``generate_initial_observation``.

    Raises:
        ValueError: If ``task_id`` is not 1, 2, or 3.
    """
    _REGISTRY = {1: EasyTask, 2: MediumTask, 3: HardTask}
    if task_id not in _REGISTRY:
        raise ValueError(
            f"Invalid task ID {task_id!r}.  Valid IDs: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[task_id]()
