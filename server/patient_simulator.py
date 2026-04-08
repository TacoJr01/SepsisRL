"""
Patient Simulator
=================
Generates a ward of patients, each following a Markov disease progression
chain.  Vitals are sampled from stage-specific distributions with configurable
noise and random missing-value dropout to model real monitoring gaps.

Sepsis stages
-------------
none  →  early  →  sepsis  →  septic_shock  →  (death / terminal)

Transition probabilities per simulated hour are set so that, without
intervention, a patient who enters "early" will reach "sepsis" in roughly
2–4 hours, matching clinical literature.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Transition matrix  (per simulated hour, no intervention)
# ---------------------------------------------------------------------------

# P(next_stage | current_stage)  — rows sum to 1
_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "none":         {"none": 0.97,  "early": 0.03,  "sepsis": 0.00, "septic_shock": 0.00},
    "early":        {"none": 0.05,  "early": 0.60,  "sepsis": 0.30, "septic_shock": 0.05},
    "sepsis":       {"none": 0.00,  "early": 0.00,  "sepsis": 0.55, "septic_shock": 0.40, "dead": 0.05},
    "septic_shock": {"none": 0.00,  "early": 0.00,  "sepsis": 0.00, "septic_shock": 0.50, "dead": 0.50},
    "dead":         {"dead": 1.00},
}

# Effect of correct interventions — multiply transition probability to worse stage by this factor
_INTERVENTION_MODIFIERS: Dict[str, float] = {
    "watch":            1.00,
    "order_cultures":   0.90,   # mild benefit — gets the workup started
    "start_antibiotics":0.40,   # strong benefit
    "iv_fluids":        0.70,   # moderate benefit (helps shock)
    "icu_transfer":     0.30,   # major benefit — specialist monitoring
}

# ---------------------------------------------------------------------------
# Vitals distributions  {stage: {vital: (mean, std)}}
# ---------------------------------------------------------------------------

_VITALS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "none": {
        "heart_rate":        (75,  8),
        "systolic_bp":       (120, 10),
        "temperature":       (37.0, 0.3),
        "respiratory_rate":  (16,  2),
        "spo2":              (98,  1),
        "wbc":               (7.5, 1.5),
        "lactate":           (1.0, 0.2),
    },
    "early": {
        "heart_rate":        (95,  10),
        "systolic_bp":       (112, 12),
        "temperature":       (38.4, 0.5),
        "respiratory_rate":  (20,  3),
        "spo2":              (96,  1.5),
        "wbc":               (13.0, 2.5),
        "lactate":           (1.8, 0.4),
    },
    "sepsis": {
        "heart_rate":        (115, 12),
        "systolic_bp":       (100, 15),
        "temperature":       (38.9, 0.6),
        "respiratory_rate":  (24,  4),
        "spo2":              (93,  2),
        "wbc":               (17.0, 3.0),
        "lactate":           (2.8, 0.6),
    },
    "septic_shock": {
        "heart_rate":        (135, 15),
        "systolic_bp":       (82,  18),
        "temperature":       (39.5, 0.8),
        "respiratory_rate":  (30,  5),
        "spo2":              (88,  3),
        "wbc":               (22.0, 4.0),
        "lactate":           (5.0, 1.2),
    },
    "dead": {
        "heart_rate":        (0, 0),
        "systolic_bp":       (0, 0),
        "temperature":       (35.0, 0),
        "respiratory_rate":  (0, 0),
        "spo2":              (0, 0),
        "wbc":               (0, 0),
        "lactate":           (0, 0),
    },
}

_VITAL_CLAMPS: Dict[str, Tuple[float, float]] = {
    "heart_rate": (20, 250),
    "systolic_bp": (40, 220),
    "temperature": (34.0, 42.0),
    "respiratory_rate": (4, 60),
    "spo2": (50, 100),
    "wbc": (0, 50),
    "lactate": (0, 20),
}

_WORSE_STAGES = ("early", "sepsis", "septic_shock", "dead")


def _compute_transition_weights(
    current: str,
    modifier: float,
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    trans = _TRANSITIONS.get(current, {"dead": 1.0})
    stages: List[str] = []
    weights: List[float] = []
    total = 0.0
    for stage, prob in trans.items():
        adjusted = prob
        if stage != current and stage in _WORSE_STAGES:
            adjusted *= modifier
        stages.append(stage)
        weights.append(adjusted)
        total += adjusted

    if total <= 0.0:
        return ("dead",), (1.0,)

    normalised = tuple(w / total for w in weights)
    return tuple(stages), normalised


_TRANSITION_TABLE: Dict[str, Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]]] = {}
for _stage in _TRANSITIONS.keys():
    per_intervention: Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]] = {}
    for _intervention, _modifier in _INTERVENTION_MODIFIERS.items():
        per_intervention[_intervention] = _compute_transition_weights(_stage, _modifier)
    _TRANSITION_TABLE[_stage] = per_intervention

# Lab results (wbc, lactate) are delayed — only available after this many steps
LAB_DELAY_STEPS = 2
# Probability a given vital reading is missing (equipment offline)
MISSING_PROB = 0.08


def _sample_vitals(stage: str, noise_multiplier: float = 1.0) -> Dict[str, Optional[float]]:
    """Sample noisy vitals for a given stage."""
    result: Dict[str, Optional[float]] = {}
    dists = _VITALS.get(stage, _VITALS["none"])
    rand = random.random
    gauss = random.gauss
    clamps = _VITAL_CLAMPS
    for vital, (mean, std) in dists.items():
        if rand() < MISSING_PROB:
            result[vital] = None  # monitoring dropout
            continue
        value = gauss(mean, std * noise_multiplier)
        lo, hi = clamps.get(vital, (-1e9, 1e9))
        result[vital] = round(max(lo, min(hi, value)), 1)
    return result


def _next_stage(current: str, intervention: Optional[str], rng: random.Random) -> str:
    """Sample next disease stage given current stage and intervention."""
    if current == "dead":
        return "dead"

    table = _TRANSITION_TABLE.get(current)
    if table is None:
        return "dead"

    key = intervention if intervention in _INTERVENTION_MODIFIERS else "watch"
    stages, weights = table[key]
    return rng.choices(stages, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Patient dataclass
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    patient_id: int
    sepsis_stage: str = "none"
    hours_in_ward: float = 0.0
    hours_since_onset: Optional[float] = None   # when early stage began
    interventions_done: List[str] = field(default_factory=list)
    alert_count: int = 0
    is_caught: bool = False        # agent escalated correctly in time
    is_dead: bool = False
    last_intervention: Optional[str] = None
    _step_vitals: Dict = field(default_factory=dict)
    _lab_ready_step: int = 0       # step from which lab results are visible

    def sample_vitals(self, current_step: int) -> Dict[str, Optional[float]]:
        vitals = _sample_vitals(self.sepsis_stage)
        # Hide lab results until delay has passed
        if current_step < self._lab_ready_step:
            vitals["wbc"] = None
            vitals["lactate"] = None
        self._step_vitals = vitals
        return vitals

    def advance(self, hours_per_step: float, rng: random.Random) -> None:
        """Advance disease state by one step."""
        if self.is_dead:
            return
        self.hours_in_ward += hours_per_step

        new_stage = _next_stage(self.sepsis_stage, self.last_intervention, rng)

        if self.sepsis_stage == "none" and new_stage == "early":
            self.hours_since_onset = self.hours_in_ward

        self.sepsis_stage = new_stage
        if new_stage == "dead":
            self.is_dead = True

        self.last_intervention = None  # reset after one step


# ---------------------------------------------------------------------------
# Ward — manages all patients
# ---------------------------------------------------------------------------

class Ward:
    """
    A simulated hospital ward with `num_patients` patients.

    Parameters
    ----------
    num_patients : int
    seed : int
    hours_per_step : float   Simulated time that elapses per env step
    golden_window : float    Hours from sepsis onset within which to intervene
    """

    def __init__(
        self,
        num_patients: int = 8,
        seed: int = 42,
        hours_per_step: float = 0.5,
        golden_window: float = 3.0,
    ):
        self.num_patients = num_patients
        self.hours_per_step = hours_per_step
        self.golden_window = golden_window
        self._rng = random.Random(seed)
        self.patients: List[Patient] = []
        self._step = 0
        self._reset_patients()

    # ------------------------------------------------------------------
    def _reset_patients(self) -> None:
        self.patients = []
        for i in range(self.num_patients):
            p = Patient(patient_id=i)
            # Seed a few patients already in "early" stage at episode start
            if self._rng.random() < 0.25:
                p.sepsis_stage = "early"
                p.hours_since_onset = self._rng.uniform(0.5, 1.5)
            p._lab_ready_step = self._step + LAB_DELAY_STEPS
            self.patients.append(p)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self._step = 0
        self._reset_patients()

    # ------------------------------------------------------------------
    def get_snapshots(self, include_stage: bool = True) -> List[Dict]:
        """Return serialisable vitals snapshots for all patients."""
        snapshots = []
        for p in self.patients:
            vitals = p.sample_vitals(self._step)
            snapshot = {
                "patient_id": p.patient_id,
                "heart_rate": vitals.get("heart_rate"),
                "systolic_bp": vitals.get("systolic_bp"),
                "temperature": vitals.get("temperature"),
                "respiratory_rate": vitals.get("respiratory_rate"),
                "spo2": vitals.get("spo2"),
                "wbc": vitals.get("wbc"),
                "lactate": vitals.get("lactate"),
                "hours_in_ward": round(p.hours_in_ward, 1),
                "interventions_done": list(p.interventions_done),
                "alert_count": p.alert_count,
            }
            if include_stage:
                snapshot["sepsis_stage"] = p.sepsis_stage
            snapshots.append(snapshot)
        return snapshots

    def get_agent_snapshots(self) -> List[Dict]:
        """
        Snapshots with sepsis_stage hidden — this is what the agent sees.
        Ground-truth stage is only used server-side for reward calculation.
        """
        return self.get_snapshots(include_stage=False)

    # ------------------------------------------------------------------
    def apply_intervention(self, patient_id: int, intervention: str) -> Tuple[float, str]:
        """
        Apply an intervention to a patient.
        Returns (reward, info_message).
        """
        if patient_id < 0 or patient_id >= self.num_patients:
            return -1.0, f"Invalid patient_id {patient_id}"

        p = self.patients[patient_id]

        if p.is_dead:
            return 0.0, f"Patient {patient_id} is already deceased."

        reward = 0.0
        info = ""

        if intervention == "watch":
            info = f"Patient {patient_id}: monitoring continued."
            return 0.0, info

        # Record intervention
        p.last_intervention = intervention
        if intervention not in p.interventions_done:
            p.interventions_done.append(intervention)

        is_sick = p.sepsis_stage in ("early", "sepsis", "septic_shock")

        if intervention in ("order_cultures", "start_antibiotics", "iv_fluids", "icu_transfer"):
            p.alert_count += 1

            if is_sick:
                hours_since = p.hours_in_ward - (p.hours_since_onset or p.hours_in_ward)
                in_golden_window = hours_since <= self.golden_window

                if not p.is_caught:
                    if in_golden_window:
                        reward = 10.0
                        p.is_caught = True
                        info = (
                            f"Patient {patient_id}: CORRECT escalation within golden window "
                            f"({hours_since:.1f}h after onset). +10"
                        )
                    else:
                        reward = 2.0  # late but better than nothing
                        p.is_caught = True
                        info = (
                            f"Patient {patient_id}: late escalation "
                            f"({hours_since:.1f}h — past {self.golden_window}h window). +2"
                        )
                else:
                    # Duplicate escalation — small alert fatigue penalty
                    reward = -0.5
                    info = f"Patient {patient_id}: already escalated; unnecessary repeat. -0.5"
            else:
                # False positive — healthy patient escalated
                reward = -1.0
                info = f"Patient {patient_id}: healthy patient escalated (false positive). -1.0"

        return reward, info

    # ------------------------------------------------------------------
    def advance_all(self) -> Tuple[float, str]:
        """
        Advance disease for all patients by one step.
        Returns (step_reward, info) — penalises any patient who deteriorates
        to septic_shock while not yet caught.
        """
        self._step += 1
        step_reward = 0.0
        infos = []

        for p in self.patients:
            prev_stage = p.sepsis_stage
            p.advance(self.hours_per_step, self._rng)

            if p.is_dead and not p.is_caught:
                step_reward -= 50.0
                infos.append(f"Patient {p.patient_id}: DIED — sepsis missed. -50")

            elif p.sepsis_stage == "septic_shock" and prev_stage != "septic_shock" and not p.is_caught:
                step_reward -= 5.0
                infos.append(
                    f"Patient {p.patient_id}: deteriorated to septic shock undetected. -5"
                )

        return step_reward, "; ".join(infos) if infos else "Ward stable."

    # ------------------------------------------------------------------
    @property
    def saved_count(self) -> int:
        return sum(1 for p in self.patients if p.is_caught)

    @property
    def missed_count(self) -> int:
        return sum(1 for p in self.patients if p.is_dead and not p.is_caught)

    @property
    def alert_fatigue(self) -> float:
        # Sum of unnecessary escalations
        total = sum(p.alert_count for p in self.patients)
        caught = sum(
            min(1, p.alert_count) for p in self.patients if p.is_caught
        )
        return float(max(0, total - caught))
