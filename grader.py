"""Task-based grader for the SepsisRL environment.

This grader defines 3 deterministic tasks (easy, medium, hard) and computes a
score in [0.0, 1.0] for each task from trajectory statistics across fixed seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, List

from models import SepsisAction
from server.sepsis_environment import MAX_STEPS, NUM_PATIENTS, SepsisEnvironment


@dataclass(frozen=True)
class TaskSpec:
    name: str
    difficulty: str
    objective: str
    seeds: List[int]
    reward_floor: float
    reward_ceiling: float
    max_alert_fatigue: float
    pass_threshold: float


TASKS: List[TaskSpec] = [
    TaskSpec(
        name="early_detection_easy",
        difficulty="easy",
        objective="Catch at least one septic patient while avoiding excessive false alarms.",
        seeds=[11, 12, 13, 14],
        reward_floor=-80.0,
        reward_ceiling=80.0,
        max_alert_fatigue=10.0,
        pass_threshold=0.55,
    ),
    TaskSpec(
        name="balanced_triage_medium",
        difficulty="medium",
        objective="Save multiple patients while balancing escalation quality and alert fatigue.",
        seeds=[21, 22, 23, 24, 25],
        reward_floor=-120.0,
        reward_ceiling=110.0,
        max_alert_fatigue=8.0,
        pass_threshold=0.62,
    ),
    TaskSpec(
        name="high_recall_hard",
        difficulty="hard",
        objective="Maintain high recall under noisy observations with low missed severe cases.",
        seeds=[31, 32, 33, 34, 35, 36],
        reward_floor=-180.0,
        reward_ceiling=130.0,
        max_alert_fatigue=6.5,
        pass_threshold=0.70,
    ),
]


@dataclass
class EpisodeMetrics:
    total_reward: float
    saved: int
    missed: int
    alert_fatigue: float
    steps: int


PolicyFn = Callable[[Dict], SepsisAction]


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def run_episode(seed: int, policy_fn: PolicyFn) -> EpisodeMetrics:
    env = SepsisEnvironment()
    obs = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    while not obs.done and steps < MAX_STEPS:
        action = policy_fn({"observation": obs, "step": steps + 1, "seed": seed})
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        steps += 1

    state = env.state
    return EpisodeMetrics(
        total_reward=round(total_reward, 4),
        saved=int(state.saved_patients),
        missed=int(state.missed_patients),
        alert_fatigue=float(state.alert_fatigue_score),
        steps=steps,
    )


def score_task(task: TaskSpec, episodes: List[EpisodeMetrics]) -> Dict:
    avg_reward = mean(e.total_reward for e in episodes)
    avg_saved = mean(e.saved for e in episodes)
    avg_missed = mean(e.missed for e in episodes)
    avg_fatigue = mean(e.alert_fatigue for e in episodes)
    avg_steps = mean(e.steps for e in episodes)

    reward_component = clip01((avg_reward - task.reward_floor) / (task.reward_ceiling - task.reward_floor))
    save_component = clip01(avg_saved / NUM_PATIENTS)
    missed_component = 1.0 - clip01(avg_missed / NUM_PATIENTS)
    fatigue_component = 1.0 - clip01(avg_fatigue / task.max_alert_fatigue)
    termination_component = 1.0 if avg_steps <= MAX_STEPS else 0.0

    # Dense trajectory-aware score: reward carries most weight, but quality and
    # safety signals (missed cases + alert fatigue) also matter.
    score = (
        0.45 * reward_component
        + 0.20 * save_component
        + 0.20 * missed_component
        + 0.10 * fatigue_component
        + 0.05 * termination_component
    )
    score = round(clip01(score), 4)

    return {
        "task": task.name,
        "difficulty": task.difficulty,
        "objective": task.objective,
        "score": score,
        "pass_threshold": task.pass_threshold,
        "passed": bool(score >= task.pass_threshold),
        "stats": {
            "avg_reward": round(avg_reward, 3),
            "avg_saved": round(avg_saved, 3),
            "avg_missed": round(avg_missed, 3),
            "avg_alert_fatigue": round(avg_fatigue, 3),
            "avg_steps": round(avg_steps, 3),
        },
        "components": {
            "reward_component": round(reward_component, 4),
            "save_component": round(save_component, 4),
            "missed_component": round(missed_component, 4),
            "fatigue_component": round(fatigue_component, 4),
            "termination_component": round(termination_component, 4),
        },
    }


def evaluate_policy(policy_fn: PolicyFn) -> Dict:
    task_results: List[Dict] = []
    for task in TASKS:
        episodes = [run_episode(seed=seed, policy_fn=policy_fn) for seed in task.seeds]
        task_results.append(score_task(task=task, episodes=episodes))

    overall_score = round(mean(item["score"] for item in task_results), 4)
    all_passed = all(item["passed"] for item in task_results)
    return {
        "overall_score": overall_score,
        "all_tasks_passed": all_passed,
        "tasks": task_results,
    }


def random_policy(context: Dict) -> SepsisAction:
    import random

    obs = context["observation"]
    patients = obs.patients or []
    patient_id = random.randint(0, max(len(patients) - 1, 0))
    intervention = random.choice(
        [
            "watch",
            "watch",
            "watch",
            "order_cultures",
            "start_antibiotics",
            "iv_fluids",
            "icu_transfer",
        ]
    )
    return SepsisAction(patient_id=patient_id, intervention=intervention)


def heuristic_policy(context: Dict) -> SepsisAction:
    obs = context["observation"]
    patients = obs.patients or []
    best_pid = 0
    best_score = -1

    for patient in patients:
        score = 0
        if float(patient.get("respiratory_rate") or 0.0) >= 22:
            score += 1
        if float(patient.get("systolic_bp") or 999.0) <= 100:
            score += 1
        if float(patient.get("heart_rate") or 0.0) > 100:
            score += 1
        if float(patient.get("temperature") or 0.0) > 38.3:
            score += 1
        if float(patient.get("lactate") or 0.0) > 2.0:
            score += 2
        if float(patient.get("spo2") or 100.0) < 94:
            score += 1

        if score > best_score:
            best_score = score
            best_pid = int(patient.get("patient_id") or 0)

    if best_score >= 2:
        intervention = "start_antibiotics"
    elif best_score == 1:
        intervention = "order_cultures"
    else:
        intervention = "watch"

    return SepsisAction(patient_id=best_pid, intervention=intervention)


def main() -> None:
    print("=" * 70)
    print("SepsisRL Task Grader (easy -> medium -> hard)")
    print("=" * 70)

    for name, policy in (("random", random_policy), ("heuristic", heuristic_policy)):
        result = evaluate_policy(policy)
        print(f"\nPolicy: {name}")
        print(f"overall_score={result['overall_score']:.4f} all_tasks_passed={result['all_tasks_passed']}")
        for task in result["tasks"]:
            print(
                f"  - {task['task']} ({task['difficulty']}): "
                f"score={task['score']:.4f} threshold={task['pass_threshold']:.2f} passed={task['passed']}"
            )


if __name__ == "__main__":
    main()
