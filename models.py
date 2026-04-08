"""
Sepsis Early Warning & Intervention — Data Models
All actions, observations and state use Pydantic (required by openenv-core).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SepsisAction(Action):
    """
    One clinical decision the agent makes for a single patient.

    Attributes
    ----------
    patient_id : int
        Index of the patient to act on (0 to NUM_PATIENTS-1).
    intervention : str
        One of the five possible interventions:
        - "watch"           Do nothing; continue monitoring.
        - "order_cultures"  Draw blood cultures (first step in sepsis workup).
        - "start_antibiotics" Begin IV antibiotics.
        - "iv_fluids"       Administer a fluid bolus.
        - "icu_transfer"    Escalate to intensive care.
    """

    patient_id: int
    intervention: Literal[
        "watch",
        "order_cultures",
        "start_antibiotics",
        "iv_fluids",
        "icu_transfer",
    ]


# ---------------------------------------------------------------------------
# Per-patient snapshot (embedded in observation, NOT a top-level model)
# ---------------------------------------------------------------------------

class PatientSnapshot(Action):
    """
    Read-only vitals + status for one patient, returned inside the observation.
    Inherits Action only so Pydantic validates it — it is never sent by the agent.
    """

    model_config = {"extra": "forbid", "validate_assignment": True, "arbitrary_types_allowed": True}

    patient_id: int
    # Vitals — may be None when monitoring equipment fails (partial observability)
    heart_rate: Optional[float]        # bpm
    systolic_bp: Optional[float]       # mmHg
    temperature: Optional[float]       # °C
    respiratory_rate: Optional[float]  # breaths/min
    spo2: Optional[float]              # % oxygen saturation
    wbc: Optional[float]               # white blood cell count ×10³/µL (lab, delayed)
    lactate: Optional[float]           # mmol/L (lab, delayed)

    # Clinical status
    hours_in_ward: float               # time since admission
    sepsis_stage: Literal["none", "early", "sepsis", "septic_shock"]
    interventions_done: List[str]      # interventions already applied this episode
    alert_count: int                   # how many times agent has escalated this patient


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SepsisObservation(Observation):
    """
    Full ward snapshot returned after every step.

    done  and  reward  are inherited from openenv Observation.
    reward meaning:
      +10   caught sepsis before the 3-hour golden window closes
      -50   patient died from missed/late sepsis
      -1    unnecessary escalation (alert fatigue penalty)
      +0.5  correctly cleared a stable patient
      -5    patient deteriorated to septic shock while agent was watching
    """

    patients: List[Dict]   # serialised PatientSnapshot dicts (List[PatientSnapshot] hits Pydantic limits with nested models)
    step_number: int
    hours_elapsed: float   # simulated wall-clock hours since episode start
    alert_fatigue_score: float  # cumulative unnecessary escalation count
    saved_patients: int    # patients caught in time this episode
    missed_patients: int   # patients who deteriorated undetected
    info: str              # human-readable last-action result


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SepsisState(State):
    """Episode-level metadata."""

    max_steps: int
    num_patients: int
    golden_window_hours: float   # hours within which sepsis must be caught
    saved_patients: int
    missed_patients: int
    alert_fatigue_score: float
