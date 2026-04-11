import asyncio
import inspect
import json
import os
import re
import sys
import subprocess
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

from openai import OpenAI

try:
    from openenv.core import EnvClient
except ImportError:
    from openenv.core.env_client import EnvClient

try:
    from SepsisRL.models import SepsisAction, SepsisObservation
except ImportError:
    try:
        from .models import SepsisAction, SepsisObservation
    except ImportError:
        _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        if _CURRENT_DIR not in sys.path:
            sys.path.insert(0, _CURRENT_DIR)
        from models import SepsisAction, SepsisObservation

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.environ.get("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")
ENV_PORT = int(os.getenv("ENV_PORT") or "8000")

TASK_NAME = os.getenv("TASK_NAME") or "sepsis"
BENCHMARK = os.getenv("BENCHMARK") or "sepsis-env"
MAX_STEPS = int(os.getenv("MAX_STEPS") or "40")
TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "128")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD") or "0.1")

ALLOWED_INTERVENTIONS = (
    "watch",
    "order_cultures",
    "start_antibiotics",
    "iv_fluids",
    "icu_transfer",
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical triage assistant operating a sepsis monitoring system.
    Choose exactly one patient and one intervention per step.

    Return only:
    patient_id=<int> intervention=<one_of: watch, order_cultures, start_antibiotics, iv_fluids, icu_transfer>
    """
).strip()


def _log_debug(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_ws_url(url: str) -> str:
    if url.startswith("ws://") or url.startswith("wss://"):
        return url
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    return url


def _action_payload(patient_id: int, intervention: str) -> Dict[str, Any]:
    action = SepsisAction(patient_id=patient_id, intervention=intervention)
    if hasattr(action, "model_dump"):
        return action.model_dump()
    return action.dict()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_patient(p: Dict[str, Any]) -> str:
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


def _build_user_prompt(step: int, patients: List[Dict[str, Any]], last_reward: float, info: str) -> str:
    patient_lines = "\n".join(_format_patient(p) for p in patients)
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Info: {info}

        Patients:
        {patient_lines}
        """
    ).strip()


def _parse_action(text: str, num_patients: int) -> Optional[Tuple[int, str]]:
    text = (text or "").strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            pid = int(data.get("patient_id"))
            intervention = str(data.get("intervention"))
            if intervention in ALLOWED_INTERVENTIONS:
                return max(0, min(num_patients - 1, pid)), intervention
    except Exception:
        pass

    match = re.search(r"patient_id\s*=\s*(\d+)\s+intervention\s*=\s*([a-z_]+)", text)
    if not match:
        return None
    pid = int(match.group(1))
    intervention = match.group(2)
    if intervention not in ALLOWED_INTERVENTIONS:
        return None
    return max(0, min(num_patients - 1, pid)), intervention


def _heuristic_action(patients: List[Dict[str, Any]]) -> Tuple[int, str]:
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

    if best_score >= 2:
        intervention = "start_antibiotics"
    elif best_score == 1:
        intervention = "order_cultures"
    else:
        intervention = "watch"

    return best_pid, intervention


def _choose_action(
    client: OpenAI,
    step: int,
    patients: List[Dict[str, Any]],
    last_reward: float,
    info: str,
) -> Tuple[int, str]:
    user_prompt = _build_user_prompt(step, patients, last_reward, info)

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
            return parsed
    except Exception as exc:
        _log_debug(f"[DEBUG] LLM call failed: {exc}")

    return _heuristic_action(patients)


def _proxy_warmup_call(client: OpenAI) -> None:
    """Force at least one request through the injected LiteLLM proxy."""
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Reply with exactly: ok"},
                {"role": "user", "content": "ok"},
            ],
            temperature=0.0,
            max_tokens=1,
            stream=False,
        )
    except Exception as exc:
        _log_debug(f"[DEBUG] LLM warmup call failed: {exc}")


def _start_container(image_name: str, port: int) -> Optional[str]:
    name = f"sepsis-env-{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "-p",
        f"{port}:8000",
        "--name",
        name,
        image_name,
    ]

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            _log_debug(f"[DEBUG] Docker run failed: {result.stderr.strip()}")
            return None
        if not result.stdout.strip():
            return None
    except Exception as exc:
        _log_debug(f"[DEBUG] Docker run failed: {exc}")
        return None

    return name


def _stop_container(name: str) -> None:
    try:
        subprocess.run(["docker", "stop", name], check=False, capture_output=True, text=True)
    except Exception as exc:
        _log_debug(f"[DEBUG] Docker stop failed: {exc}")


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> bool:
    url = base_url.rstrip("/") + "/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _make_env_client(base_url: str) -> EnvClient:
    try:
        params = inspect.signature(EnvClient).parameters
        if "action_cls" in params and "observation_cls" in params:
            return EnvClient(
                base_url=base_url,
                action_cls=SepsisAction,
                observation_cls=SepsisObservation,
            )
    except Exception:
        pass
    return EnvClient(base_url=base_url)


async def main() -> None:
    try:
        api_base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
    except KeyError as exc:
        _log_debug(f"[DEBUG] Missing required environment variable: {exc}")
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    _proxy_warmup_call(client)

    container_name = None
    base_url = ENV_BASE_URL or f"http://localhost:{ENV_PORT}"

    if LOCAL_IMAGE_NAME and not ENV_BASE_URL:
        container_name = _start_container(LOCAL_IMAGE_NAME, ENV_PORT)
        base_url = f"http://localhost:{ENV_PORT}"
        _wait_for_health(base_url)

    ws_url = _normalize_ws_url(base_url)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with _make_env_client(ws_url) as env:
            result = await env.reset()
            obs = _get(result, "observation", result)
            last_reward = _safe_float(_get(obs, "reward", 0.0))
            info = _get(obs, "info", "")

            for step in range(1, MAX_STEPS + 1):
                done = bool(_get(obs, "done", False))
                if done:
                    break

                patients = _get(obs, "patients", []) or []
                patient_id, intervention = _choose_action(
                    client=client,
                    step=step,
                    patients=patients,
                    last_reward=last_reward,
                    info=info,
                )

                payload = _action_payload(patient_id, intervention)
                result = await env.step(payload)
                obs = _get(result, "observation", result)

                reward = _safe_float(_get(result, "reward", _get(obs, "reward", 0.0)))
                done = bool(_get(result, "done", _get(obs, "done", False)))
                info = _get(obs, "info", "")
                error = _get(result, "last_action_error", _get(obs, "last_action_error", None))

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                action_str = f"patient_id={patient_id} intervention={intervention}"
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

        max_total_reward = MAX_STEPS * 10.0
        if max_total_reward > 0:
            score = max(0.0, min(sum(rewards) / max_total_reward, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        _log_debug(f"[DEBUG] Inference error: {exc}")

    finally:
        if container_name:
            _stop_container(container_name)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
