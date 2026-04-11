"""Baseline model evaluation across deterministic SepsisRL tasks.

This script uses the OpenAI client and runs a model-driven policy over all
3 task definitions from grader.py, producing reproducible scores.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI

from grader import TASKS, evaluate_policy, evaluate_tasks
from inference import (
    MAX_TOKENS,
    MODEL_NAME,
    SYSTEM_PROMPT,
    TEMPERATURE,
    _parse_action,
)
from models import SepsisAction


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_patient(p: Dict) -> str:
    return (
        "patient_id={pid} hr={hr} sbp={sbp} temp={temp} rr={rr} "
        "spo2={spo2} wbc={wbc} lactate={lactate} alerts={alerts}"
    ).format(
        pid=p.get("patient_id"),
        hr=p.get("heart_rate"),
        sbp=p.get("systolic_bp"),
        temp=p.get("temperature"),
        rr=p.get("respiratory_rate"),
        spo2=p.get("spo2"),
        wbc=p.get("wbc"),
        lactate=p.get("lactate"),
        alerts=p.get("alert_count"),
    )


def _build_prompt(step: int, patients, last_reward: float, info: str) -> str:
    patient_lines = "\n".join(_format_patient(p) for p in patients)
    return (
        f"Step: {step}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Info: {info}\n\n"
        f"Patients:\n{patient_lines}"
    )


def _heuristic_fallback(patients):
    best_pid = 0
    best_score = -1

    for p in patients:
        score = 0
        if _safe_float(p.get("respiratory_rate")) >= 22:
            score += 1
        if _safe_float(p.get("systolic_bp"), 999.0) <= 100:
            score += 1
        if _safe_float(p.get("heart_rate")) > 100:
            score += 1
        if _safe_float(p.get("temperature")) > 38.3:
            score += 1
        if _safe_float(p.get("lactate")) > 2.0:
            score += 2
        if _safe_float(p.get("spo2"), 100.0) < 94:
            score += 1
        if score > best_score:
            best_score = score
            best_pid = int(p.get("patient_id") or 0)

    intervention = "start_antibiotics" if best_score >= 2 else "order_cultures" if best_score == 1 else "watch"
    return best_pid, intervention


def make_model_policy(client: OpenAI):
    def _policy(context: Dict) -> SepsisAction:
        obs = context["observation"]
        step = int(context.get("step", 1))
        patients = obs.patients or []
        last_reward = _safe_float(getattr(obs, "reward", 0.0), 0.0)
        info = str(getattr(obs, "info", ""))
        user_prompt = _build_prompt(step=step, patients=patients, last_reward=last_reward, info=info)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            parsed = _parse_action(text, len(patients))
            if parsed:
                patient_id, intervention = parsed
                return SepsisAction(patient_id=patient_id, intervention=intervention)
        except Exception:
            pass

        patient_id, intervention = _heuristic_fallback(patients)
        return SepsisAction(patient_id=patient_id, intervention=intervention)

    return _policy


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY)")

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("API_BASE_URL")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    policy = make_model_policy(client)
    result = evaluate_policy(policy)
    task_results = evaluate_tasks(policy)

    print("Baseline evaluation complete")
    print(f"overall_score={result['overall_score']:.4f}")
    for task in result["tasks"]:
        print(
            f"- {task['task']} ({task['difficulty']}): "
            f"score={task['score']:.4f} threshold={task['pass_threshold']:.2f} passed={task['passed']}"
        )

    print("\nTASK_SCORES_OPEN_INTERVAL")
    for task in task_results:
        print(f"- {task['name']}: score={task['score']:.4f} grader={task['grader']}")

    print("\nTASK_DEFINITIONS")
    print(json.dumps([
        {
            "name": task.name,
            "difficulty": task.difficulty,
            "objective": task.objective,
            "seeds": task.seeds,
        }
        for task in TASKS
    ], indent=2))


if __name__ == "__main__":
    main()
