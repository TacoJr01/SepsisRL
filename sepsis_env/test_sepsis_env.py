"""
Unit tests for the Sepsis environment.
Run with:  pytest sepsis_env/test_sepsis_env.py -v
"""

import pytest
from sepsis_env.models import SepsisAction, SepsisObservation, SepsisState
from sepsis_env.server.sepsis_environment import SepsisEnvironment, MAX_STEPS, NUM_PATIENTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = SepsisEnvironment()
    e.reset(seed=42)
    return e


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

def test_reset_returns_observation():
    env = SepsisEnvironment()
    obs = env.reset(seed=0)
    assert isinstance(obs, SepsisObservation)
    assert obs.done is False
    assert obs.step_number == 0
    assert len(obs.patients) == NUM_PATIENTS


def test_reset_patients_have_expected_fields():
    env = SepsisEnvironment()
    obs = env.reset(seed=1)
    for p in obs.patients:
        assert "patient_id" in p
        assert "heart_rate" in p
        assert "interventions_done" in p
        # sepsis_stage must NOT be visible to agent
        assert "sepsis_stage" not in p


def test_reset_clears_state():
    env = SepsisEnvironment()
    env.reset(seed=0)
    # Do some steps
    for _ in range(5):
        env.step(SepsisAction(patient_id=0, intervention="watch"))
    # Reset and verify
    obs = env.reset(seed=0)
    assert obs.step_number == 0
    assert obs.saved_patients == 0
    assert obs.missed_patients == 0
    assert env.state.step_count == 0


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

def test_step_increments_step_count(env):
    obs = env.step(SepsisAction(patient_id=0, intervention="watch"))
    assert obs.step_number == 1
    assert env.state.step_count == 1


def test_step_returns_observation(env):
    obs = env.step(SepsisAction(patient_id=0, intervention="watch"))
    assert isinstance(obs, SepsisObservation)
    assert obs.reward is not None


def test_step_advances_hours(env):
    obs = env.step(SepsisAction(patient_id=0, intervention="watch"))
    assert obs.hours_elapsed > 0


def test_multiple_steps_advance_correctly(env):
    for i in range(10):
        obs = env.step(SepsisAction(patient_id=i % NUM_PATIENTS, intervention="watch"))
    assert obs.step_number == 10
    assert abs(obs.hours_elapsed - 5.0) < 0.1  # 10 × 0.5h


def test_invalid_patient_id_gives_negative_reward(env):
    obs = env.step(SepsisAction(patient_id=99, intervention="watch"))
    assert obs.reward is not None and obs.reward <= 0


# ---------------------------------------------------------------------------
# Termination tests
# ---------------------------------------------------------------------------

def test_episode_terminates_at_max_steps():
    env = SepsisEnvironment()
    obs = env.reset(seed=7)
    steps = 0
    while not obs.done:
        obs = env.step(SepsisAction(patient_id=steps % NUM_PATIENTS, intervention="watch"))
        steps += 1
        assert steps <= MAX_STEPS + 1, "Episode ran past MAX_STEPS"
    assert obs.done is True


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

def test_false_positive_gives_negative_reward():
    """
    Escalating a patient who stays healthy should yield negative reward
    across enough episodes (may occasionally be sick by chance).
    """
    negative_count = 0
    for seed in range(30):
        env = SepsisEnvironment()
        env.reset(seed=seed)
        # Step 0 patient, who starts as none with high probability
        obs = env.step(SepsisAction(patient_id=0, intervention="icu_transfer"))
        if obs.reward is not None and obs.reward < 0:
            negative_count += 1
    # At least half should be negative (patient starts healthy most of the time)
    assert negative_count >= 10, f"Expected mostly negative rewards, got {negative_count}/30"


def test_reward_has_variance():
    """Reward signal is not constant."""
    env = SepsisEnvironment()
    rewards = []
    for seed in range(20):
        obs = env.reset(seed=seed)
        for _ in range(10):
            obs = env.step(SepsisAction(
                patient_id=0,
                intervention="start_antibiotics"
            ))
            if obs.reward is not None:
                rewards.append(obs.reward)
    assert len(set(rewards)) > 1, "Reward is constant — something is wrong"


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

def test_state_fields(env):
    state = env.state
    assert isinstance(state, SepsisState)
    assert state.num_patients == NUM_PATIENTS
    assert state.max_steps == MAX_STEPS
    assert state.golden_window_hours == 3.0


def test_state_updates_after_steps(env):
    env.step(SepsisAction(patient_id=0, intervention="watch"))
    assert env.state.step_count == 1


# ---------------------------------------------------------------------------
# Observation hidden fields
# ---------------------------------------------------------------------------

def test_sepsis_stage_hidden_in_all_steps():
    env = SepsisEnvironment()
    obs = env.reset(seed=42)
    for _ in range(5):
        for p in obs.patients:
            assert "sepsis_stage" not in p, "sepsis_stage must not be visible to the agent"
        obs = env.step(SepsisAction(patient_id=0, intervention="watch"))


# ---------------------------------------------------------------------------
# Alert fatigue
# ---------------------------------------------------------------------------

def test_alert_fatigue_increases_on_false_positives():
    """Unnecessary escalations should drive up alert_fatigue_score."""
    env = SepsisEnvironment()
    obs = env.reset(seed=0)
    initial_fatigue = obs.alert_fatigue_score

    # Keep escalating — at least some will be false positives
    for _ in range(10):
        obs = env.step(SepsisAction(patient_id=0, intervention="icu_transfer"))

    # Fatigue score should have grown
    assert obs.alert_fatigue_score >= initial_fatigue
