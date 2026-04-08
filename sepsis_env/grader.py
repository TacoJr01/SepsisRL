"""
Sepsis Environment — Programmatic Grader
=========================================
Runs N episodes with random and heuristic policies and verifies that the
environment produces correct reward signals.  Used by Round 1 judges and
for your own local testing.

Run with:
    python -m sepsis_env.grader
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List

from .server.patient_simulator import Ward
from .server.sepsis_environment import (
    SepsisEnvironment,
    NUM_PATIENTS,
    MAX_STEPS,
    GOLDEN_WINDOW,
)
from .models import SepsisAction


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Picks a random patient and random intervention each step."""

    name = "random"

    def act(self, observation) -> SepsisAction:
        patient_id = random.randint(0, NUM_PATIENTS - 1)
        intervention = random.choice([
            "watch", "watch", "watch",  # weighted to mostly watch
            "order_cultures", "start_antibiotics", "iv_fluids", "icu_transfer",
        ])
        return SepsisAction(patient_id=patient_id, intervention=intervention)


class HeuristicPolicy:
    """
    Rule-based heuristic that escalates patients whose vitals cross
    clinical sepsis screening thresholds (qSOFA-inspired).

    qSOFA criteria (2 of 3 = flag):
      - Respiratory rate ≥ 22
      - Altered mentation  (not modelled — skip)
      - Systolic BP ≤ 100
    Additional: HR > 100 or Temp > 38.3
    """

    name = "heuristic"

    def act(self, observation) -> SepsisAction:
        patients = observation.patients
        best_pid = 0
        best_score = -1

        for p in patients:
            if p.get("interventions_done") and "icu_transfer" in p["interventions_done"]:
                continue  # already fully escalated

            score = 0
            if (p.get("respiratory_rate") or 0) >= 22:
                score += 1
            if (p.get("systolic_bp") or 999) <= 100:
                score += 1
            if (p.get("heart_rate") or 0) > 100:
                score += 1
            if (p.get("temperature") or 0) > 38.3:
                score += 1
            if (p.get("lactate") or 0) > 2.0:
                score += 2
            if (p.get("spo2") or 100) < 94:
                score += 1

            if score > best_score:
                best_score = score
                best_pid = p["patient_id"]

        if best_score >= 2:
            intervention = "start_antibiotics"
        elif best_score == 1:
            intervention = "order_cultures"
        else:
            intervention = "watch"

        return SepsisAction(patient_id=best_pid, intervention=intervention)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(policy, seed: int = 0) -> Dict:
    env = SepsisEnvironment()
    obs = env.reset(seed=seed)

    total_reward = 0.0
    for _ in range(MAX_STEPS):
        if obs.done:
            break
        action = policy.act(obs)
        obs = env.step(action)
        total_reward += obs.reward or 0.0

    final_state = env.state
    return {
        "total_reward": round(total_reward, 2),
        "saved": final_state.saved_patients,
        "missed": final_state.missed_patients,
        "alert_fatigue": final_state.alert_fatigue_score,
    }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(num_episodes: int = 20) -> None:
    print("=" * 60)
    print("  Sepsis Early Warning — Environment Grader")
    print("=" * 60)

    policies = [RandomPolicy(), HeuristicPolicy()]

    for policy in policies:
        results = [run_episode(policy, seed=i) for i in range(num_episodes)]

        rewards = [r["total_reward"] for r in results]
        saved   = [r["saved"] for r in results]
        missed  = [r["missed"] for r in results]
        fatigue = [r["alert_fatigue"] for r in results]

        print(f"\nPolicy: {policy.name}  ({num_episodes} episodes)")
        print(f"  Mean reward     : {statistics.mean(rewards):+.1f}  (±{statistics.stdev(rewards):.1f})")
        print(f"  Avg saved       : {statistics.mean(saved):.1f} / {NUM_PATIENTS}")
        print(f"  Avg missed      : {statistics.mean(missed):.1f}")
        print(f"  Alert fatigue   : {statistics.mean(fatigue):.1f}")

    print("\n" + "=" * 60)
    print("Sanity checks:")

    # Check 1: heuristic should outperform random
    rand_results  = [run_episode(RandomPolicy(),    seed=i) for i in range(num_episodes)]
    heur_results  = [run_episode(HeuristicPolicy(), seed=i) for i in range(num_episodes)]
    rand_mean  = statistics.mean(r["total_reward"] for r in rand_results)
    heur_mean  = statistics.mean(r["total_reward"] for r in heur_results)
    check1 = heur_mean > rand_mean
    print(f"  [{'PASS' if check1 else 'FAIL'}] Heuristic reward ({heur_mean:+.1f}) > Random ({rand_mean:+.1f})")

    # Check 2: reward signal is non-trivial (not constant)
    rewards = [r["total_reward"] for r in rand_results]
    check2 = statistics.stdev(rewards) > 1.0
    print(f"  [{'PASS' if check2 else 'FAIL'}] Reward has variance > 1.0 (actual: {statistics.stdev(rewards):.2f})")

    # Check 3: saved count is reasonable (> 0 for heuristic)
    heur_saved = statistics.mean(r["saved"] for r in heur_results)
    check3 = heur_saved > 0
    print(f"  [{'PASS' if check3 else 'FAIL'}] Heuristic saves > 0 patients on average ({heur_saved:.1f})")

    # Check 4: environment terminates correctly
    env = SepsisEnvironment()
    obs = env.reset(seed=999)
    steps = 0
    policy = RandomPolicy()
    while not obs.done:
        obs = env.step(policy.act(obs))
        steps += 1
        if steps > MAX_STEPS + 5:
            break
    check4 = obs.done and steps <= MAX_STEPS
    print(f"  [{'PASS' if check4 else 'FAIL'}] Episode terminates within MAX_STEPS ({steps} steps)")

    print("=" * 60)
    all_pass = check1 and check2 and check3 and check4
    print(f"\n  Overall: {'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
    print()


if __name__ == "__main__":
    grade()
