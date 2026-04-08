"""
Sepsis Early Warning & Intervention — Environment
==================================================
Implements the openenv Environment interface.

Each episode:
  - A ward of NUM_PATIENTS patients is initialised.
  - Each step, the agent picks ONE patient and applies ONE intervention.
  - After the action, all patients' disease states advance.
  - Episode ends when MAX_STEPS is reached.

Reward signals (fully self-contained, no API calls needed):
  +10   correct escalation within the 3-hour golden window
  +2    late-but-correct escalation (past window)
  -1    false positive (escalated a healthy patient)
  -0.5  duplicate escalation on already-caught patient
  -5    patient deteriorated to septic shock undetected
  -50   patient died from undetected/untreated sepsis
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from models import SepsisAction, SepsisObservation, SepsisState
from .patient_simulator import Ward

NUM_PATIENTS = 8
MAX_STEPS = 40          # 40 steps × 0.5 h/step = 20 simulated hours
HOURS_PER_STEP = 0.5
GOLDEN_WINDOW = 3.0     # hours


class SepsisEnvironment(Environment[SepsisAction, SepsisObservation, SepsisState]):
    """
    Full-ward sepsis monitoring environment.

    The agent sees noisy, partially-observable vitals for all patients and
    must decide which patient to act on and which intervention to apply each
    step.  Ground-truth sepsis stage is hidden — the agent must infer it
    from the vital signs pattern.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._ward = Ward(
            num_patients=NUM_PATIENTS,
            hours_per_step=HOURS_PER_STEP,
            golden_window=GOLDEN_WINDOW,
        )
        self._step_count = 0
        self._episode_id: Optional[str] = None
        self._cumulative_reward = 0.0

    # ------------------------------------------------------------------
    # openenv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SepsisObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._ward.reset(seed=seed)
        self._reset_rubric()

        return SepsisObservation(
            done=False,
            reward=0.0,
            patients=self._ward.get_agent_snapshots(),
            step_number=0,
            hours_elapsed=0.0,
            alert_fatigue_score=0.0,
            saved_patients=0,
            missed_patients=0,
            info=(
                f"Episode started. Ward has {NUM_PATIENTS} patients. "
                f"Golden window: {GOLDEN_WINDOW}h. Max steps: {MAX_STEPS}."
            ),
        )

    def step(
        self,
        action: SepsisAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SepsisObservation:
        self._step_count += 1

        # 1. Apply the agent's chosen intervention
        action_reward, action_info = self._ward.apply_intervention(
            action.patient_id, action.intervention
        )

        # 2. Advance all patients by one time step
        step_reward, step_info = self._ward.advance_all()

        total_reward = action_reward + step_reward
        self._cumulative_reward += total_reward

        done = (
            self._step_count >= MAX_STEPS
            or all(p.is_dead or p.is_caught for p in self._ward.patients)
        )

        combined_info = " | ".join(filter(None, [action_info, step_info]))

        return SepsisObservation(
            done=done,
            reward=round(total_reward, 2),
            patients=self._ward.get_agent_snapshots(),
            step_number=self._step_count,
            hours_elapsed=round(self._step_count * HOURS_PER_STEP, 1),
            alert_fatigue_score=round(self._ward.alert_fatigue, 1),
            saved_patients=self._ward.saved_count,
            missed_patients=self._ward.missed_count,
            info=combined_info,
        )

    @property
    def state(self) -> SepsisState:
        return SepsisState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            max_steps=MAX_STEPS,
            num_patients=NUM_PATIENTS,
            golden_window_hours=GOLDEN_WINDOW,
            saved_patients=self._ward.saved_count,
            missed_patients=self._ward.missed_count,
            alert_fatigue_score=round(self._ward.alert_fatigue, 1),
        )
