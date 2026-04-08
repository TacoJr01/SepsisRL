# Module 4: Building Your Own Environment

## The 3-Component Pattern

Every OpenEnv environment has the same structure:

```
my_env/
├── models.py              ← Types: Action, Observation, State
├── client.py              ← HTTP/WebSocket client (what users import)
├── server/
│   ├── environment.py     ← Game logic (reset, step, state)
│   ├── app.py             ← FastAPI server
│   └── Dockerfile         ← Container definition
├── openenv.yaml           ← Manifest
└── pyproject.toml         ← Package metadata
```

You'll build all of these for a word-guessing game. ~100 lines of meaningful code.

## Step 1: Define Your Types (`models.py`)

Start with the data contracts. What does an action look like? What does an observation contain?

```python
from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

# Action, Observation, State are Pydantic BaseModel subclasses —
# no @dataclass decorator needed; define fields directly as class attributes.

class WordGameAction(Action):
    guess: str  # The player's guessed letter

class WordGameObservation(Observation):
    # done: bool and reward: Optional[float] are already in Observation base
    masked_word: str           # e.g., "h_ll_"
    guessed_letters: List[str] # Letters tried so far
    attempts_remaining: int
    message: str               # Feedback message

class WordGameState(State):
    # episode_id: Optional[str] and step_count: int are already in State base
    target_word: str = ""
    max_attempts: int = 10
```

These Pydantic models do three things:
1. **Document the API** — anyone reading `models.py` knows the interface
2. **Enable IDE autocomplete** — `obs.masked_word` instead of `obs["masked_word"]`
3. **Catch bugs at type-check time** — misspell a field name and your linter tells you

## Step 2: Implement the Environment (`server/environment.py`)

The environment implements `reset()`, `step()`, and `state`. This is where your game logic lives.

```python
import random
import uuid
from openenv.core.env_server import Environment
from .models import WordGameAction, WordGameObservation, WordGameState

WORDS = ["python", "neural", "tensor", "matrix", "vector",
         "kernel", "lambda", "signal", "binary", "cipher"]

class WordGameEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # Allow multiple simultaneous clients

    MAX_ATTEMPTS = 10

    def __init__(self):
        self._state = WordGameState()
        self._target = ""
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS

    def reset(self, seed=None, episode_id=None, **kwargs) -> WordGameObservation:
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS
        self._state = WordGameState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=self.MAX_ATTEMPTS,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    def step(self, action: WordGameAction, timeout_s=None, **kwargs) -> WordGameObservation:
        letter = action.guess.lower().strip()
        self._state.step_count += 1
        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{letter}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{letter}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{self._target}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{self._target}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    @property
    def state(self) -> WordGameState:
        return self._state

    def _mask(self) -> str:
        return "".join(c if c in self._guessed else "_" for c in self._target)
```

## Step 3: Create the Client (`client.py`)

The client translates between your typed models and the WebSocket wire format. Three abstract methods to implement:

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import WordGameAction, WordGameObservation, WordGameState

class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):
    def _step_payload(self, action: WordGameAction) -> dict:
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=WordGameObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                masked_word=obs_data.get("masked_word", ""),
                guessed_letters=obs_data.get("guessed_letters", []),
                attempts_remaining=obs_data.get("attempts_remaining", 0),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WordGameState:
        return WordGameState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target_word=payload.get("target_word", ""),
            max_attempts=payload.get("max_attempts", 6),
        )
```

That's it. The `EnvClient` base class handles all WebSocket communication.

## Step 4: Wire Up FastAPI (`server/app.py`)

One line of meaningful code:

```python
from openenv.core.env_server import create_fastapi_app
from environment import WordGameEnvironment

app = create_fastapi_app(WordGameEnvironment)
```

`create_fastapi_app()` creates all the endpoints: `/ws`, `/reset`, `/step`, `/state`, `/health`, `/web`, `/docs`.

## Step 5: Dockerize (`server/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## The Fast Path: `openenv init`

Don't want to write all this by hand? Scaffold it:

```bash
openenv init word_game
cd word_game
```

This creates the full directory structure with placeholder code. You just fill in:
1. Your types in `models.py`
2. Your game logic in `server/environment.py`
3. Your client parsing in `client.py`

Then test and deploy:
```bash
uv run server                    # Test locally
openenv push --repo-id user/word-game  # Deploy
```

## What's Next

In the [notebook](notebook.ipynb), you'll scaffold a word game with `openenv init`, implement the game logic, test it locally, and deploy it.

**Key takeaway:** The pattern is always the same — types, server logic, client, container. ~100 lines of meaningful code for a custom environment.
