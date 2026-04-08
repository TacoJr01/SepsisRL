---
title: Sepsis Early Warning & Intervention
emoji: 🏥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - healthcare
  - reinforcement-learning
---

# Sepsis Early Warning & Intervention

> An OpenEnv RL environment where an agent monitors a hospital ward and must
> identify septic patients before the 3-hour golden window closes.

## Why This Matters

Sepsis kills **11 million people per year** globally. Over 80% of deaths are
preventable if caught and treated within the first 3 hours — yet clinicians
working under cognitive load routinely miss early warning signs.  
This environment trains agents to monitor multiple patients simultaneously,
infer disease progression from noisy partial observations, and act decisively
without triggering alert fatigue.

---

## Environment Overview

| Property | Value |
|---|---|
| Patients per episode | 8 |
| Episode length | 40 steps (≈ 20 simulated hours) |
| Time per step | 0.5 simulated hours |
| Golden window | 3 hours from sepsis onset |
| Observability | Partial — noisy vitals, delayed labs, random dropouts |

### What the agent sees

Each step the agent receives a snapshot of all 8 patients' vitals:

| Vital | Notes |
|---|---|
| Heart rate (bpm) | May be missing (8% dropout) |
| Systolic BP (mmHg) | May be missing |
| Temperature (°C) | May be missing |
| Respiratory rate | May be missing |
| SpO2 (%) | May be missing |
| WBC (×10³/µL) | Lab result — delayed 2 steps after ordering |
| Lactate (mmol/L) | Lab result — delayed 2 steps after ordering |

**Sepsis stage is hidden.** The agent must infer it from vitals patterns.

### What the agent can do

Each step the agent picks **one patient** and applies **one intervention**:

| Intervention | Effect |
|---|---|
| `watch` | Continue monitoring; no escalation |
| `order_cultures` | Start blood culture workup; mild disease modifier |
| `start_antibiotics` | Begin IV antibiotics; strong disease modifier |
| `iv_fluids` | Fluid bolus; helps shock patients |
| `icu_transfer` | Escalate to ICU; strongest modifier |

### Reward signal

| Event | Reward |
|---|---|
| Correct escalation within 3-hour window | **+10** |
| Late-but-correct escalation (past window) | +2 |
| False positive (healthy patient escalated) | −1 |
| Duplicate escalation (already caught) | −0.5 |
| Patient deteriorates to septic shock undetected | −5 |
| Patient dies from missed sepsis | **−50** |

---

## Disease Model

Patients follow a Markov progression chain:

```
none → early → sepsis → septic_shock → (death)
```

Transition probabilities are calibrated so that, without intervention, a
patient who enters "early" reaches "sepsis" in approximately 2–4 simulated
hours — matching clinical literature.  Interventions reduce the probability
of worsening by a stage-specific modifier.

---

## Quick Start

### Run locally (no Docker)

```bash
pip install openenv-core fastapi uvicorn pydantic

# From the parent directory of sepsis_env/
uvicorn sepsis_env.server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
# Build from the sepsis_env/ directory
docker build -t sepsis-env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 sepsis-env:latest
```

### Test the environment directly (no server)

```bash
pytest sepsis_env/test_sepsis_env.py -v
```

### Run the grader

```bash
python -m sepsis_env.grader
```

Expected output:
```
============================================================
  Sepsis Early Warning — Environment Grader
============================================================

Policy: random  (20 episodes)
  Mean reward     : -42.3  (±18.7)
  Avg saved       :  1.2 / 8
  Avg missed      :  2.8
  Alert fatigue   :  6.1

Policy: heuristic  (20 episodes)
  Mean reward     : +28.4  (±12.1)
  Avg saved       :  3.9 / 8
  Avg missed      :  0.9
  Alert fatigue   :  2.4

Sanity checks:
  [PASS] Heuristic reward > Random
  [PASS] Reward has variance > 1.0
  [PASS] Heuristic saves > 0 patients on average
  [PASS] Episode terminates within MAX_STEPS
```

---

## Using the Environment in a Training Loop

```python
import asyncio
from openenv.core import EnvClient, StepResult

# Connect to a running server
async def train():
    async with EnvClient(base_url="ws://localhost:8000") as client:
        obs = await client.reset()

        for step in range(40):
            if obs.done:
                break

            # Your agent logic here
            # obs.patients contains vitals for all 8 patients
            action = {"patient_id": 0, "intervention": "watch"}
            obs = await client.step(action)
            print(f"Step {step}: reward={obs.reward:.1f}, saved={obs.saved_patients}")

asyncio.run(train())
```

### With TRL / GRPO

```python
from trl import GRPOTrainer, GRPOConfig

# Point TRL at the running environment
config = GRPOConfig(
    environment_url="ws://localhost:8000",
    ...
)
trainer = GRPOTrainer(model=model, args=config)
trainer.train()
```

---

## Project Structure

```
sepsis_env/
├── __init__.py                  # Package exports
├── models.py                    # SepsisAction, SepsisObservation, SepsisState
├── grader.py                    # Programmatic grader (run for evaluation)
├── test_sepsis_env.py           # Pytest unit tests
├── openenv.yaml                 # Environment manifest
├── pyproject.toml               # Package config and dependencies
├── README.md                    # This file
└── server/
    ├── __init__.py
    ├── sepsis_environment.py    # Core Environment subclass
    ├── patient_simulator.py     # Markov disease model + vitals generator
    ├── app.py                   # FastAPI app (create_fastapi_app)
    ├── requirements.txt         # Docker dependencies
    └── Dockerfile               # Container definition
```

---

## Real-World Relevance

This environment is directly inspired by clinical decision support research:

- **qSOFA scoring** (quick Sequential Organ Failure Assessment) — the
  heuristic policy implements a simplified qSOFA screen.
- **Surviving Sepsis Campaign** guidelines recommend antibiotics within 1
  hour of sepsis recognition — modelled by the golden window reward.
- **Alert fatigue** is a well-documented phenomenon in ICUs where too many
  alarms cause clinicians to become desensitised — modelled as an
  accumulating reward penalty.

---

## License

BSD 3-Clause
