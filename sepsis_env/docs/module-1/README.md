# Module 1: Why OpenEnv? From Cartpole to Production RL

## The RL Loop in 60 Seconds

Reinforcement Learning is a loop:

```python
while not done:
    observation = environment.observe()
    action = policy.choose(observation)
    reward = environment.step(action)
    policy.learn(reward)
```

Observe → Act → Reward → Repeat. That's it.

The agent interacts with an environment, gets feedback, and improves. Every RL system — from game-playing bots to LLM fine-tuning with GRPO — follows this pattern.

## Why Gym/Gymnasium Falls Short for LLM Training

OpenAI Gym (now Gymnasium) is the standard for RL research. It works great for Cartpole. But when you try to use it for production LLM training, problems appear:

| Challenge | Gymnasium | What you actually need |
|-----------|-----------|----------------------|
| **Type Safety** | `obs[0][3]` — what is this? | `obs.info_state` — IDE knows |
| **Isolation** | Same process (can crash training) | Docker containers (fully isolated) |
| **Deployment** | "Works on my machine" | Same container everywhere |
| **Scaling** | Hard to distribute | Deploy to Kubernetes |
| **Language** | Python only | Any language via HTTP |
| **Debugging** | Cryptic numpy errors | Clear type errors |

The core issue: Gymnasium assumes your environment runs in the same process as your training code. That's fine for research. It's a disaster for production.

## The OpenEnv Philosophy

**RL environments should be microservices.**

You don't run your database in the same process as your web server. Same principle applies to RL environments:

- **Isolated** — Run in containers. Security + stability.
- **Standard** — HTTP/WebSocket API. Works from any language.
- **Versioned** — Docker images. Reproducible everywhere.
- **Scalable** — Deploy to cloud with one command.
- **Type-safe** — Catch bugs before they happen.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE                                        │
│                                                            │
│  env = EchoEnv(base_url="https://...")                    │
│  result = env.reset()           ← Type-safe!              │
│  result = env.step(action)      ← Type-safe!              │
│                                                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  │  WebSocket / HTTP  (Language-Agnostic)
                  │
┌─────────────────▼──────────────────────────────────────────┐
│  DOCKER CONTAINER (HF Space, local, cloud)                 │
│                                                            │
│  ┌──────────────────────────────────────────────┐         │
│  │  FastAPI Server                              │         │
│  │  └─ Environment (reset, step, state)         │         │
│  │     └─ Your Game/Simulation Logic            │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  Isolated • Reproducible • Secure                          │
└────────────────────────────────────────────────────────────┘
```

The client uses the `/ws` WebSocket endpoint by default. You never see the HTTP details — just clean Python methods:

```python
env.reset()    # Under the hood: WebSocket message
env.step(...)  # Under the hood: WebSocket message
env.state()    # Under the hood: WebSocket message
```

## The 3-Method Interface

Every OpenEnv environment exposes exactly three methods:

| Method | What it does | Returns |
|--------|-------------|---------|
| `reset()` | Start a new episode | `StepResult` (observation, reward, done) |
| `step(action)` | Take an action | `StepResult` (observation, reward, done) |
| `state()` | Get episode metadata | `State` (episode_id, step_count, etc.) |

This is the same whether you're playing Catch, Wordle, Tic-Tac-Toe, or a custom environment you built yourself.

## The 3-Component Pattern

Every OpenEnv environment has three components:

```
my_env/
├── models.py              ← Type-safe contracts (Action, Observation, State)
├── client.py              ← What you import in training code
└── server/
    ├── environment.py     ← Game/simulation logic
    ├── app.py             ← FastAPI server
    └── Dockerfile         ← Container definition
```

**Server side** (runs in Docker):
```python
class Environment(ABC):
    def reset(self) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    @property
    def state(self) -> State: ...
```

**Client side** (your training code):
```python
class EnvClient(ABC):
    async def reset(self, **kwargs) -> StepResult: ...
    async def step(self, action) -> StepResult: ...
    async def state(self) -> State: ...
    def sync(self) -> SyncEnvClient: ...  # Sync wrapper for notebooks/scripts
```

Same interface on both sides. Communication via WebSocket. You focus on RL.

For simple MCP-based environments (like the Echo environment), the interface is
tool-based instead: `env.list_tools()` and `env.call_tool(name, **kwargs)`.

## What's Next

In the [notebook](notebook.ipynb), you'll connect to three real hosted environments — Echo, OpenSpiel Catch, and TextArena Wordle — and interact with each using the same pattern.

**Key takeaway:** Every OpenEnv environment has the same 3-method interface. Once you know one, you know them all.
