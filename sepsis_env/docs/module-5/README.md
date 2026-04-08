# Module 5: Training with OpenEnv + TRL

## What is GRPO?

**Group Relative Policy Optimization** is a reinforcement learning algorithm for fine-tuning LLMs. The intuition:

1. Generate a group of completions for the same prompt
2. Score each completion with reward functions
3. Use the relative ranking within the group to update the policy

No value model needed (unlike PPO). The group itself provides the baseline.

GRPO works well for tasks where you can define reward functions — games, code generation, reasoning, structured output.

## The TRL + OpenEnv Integration

[TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl) provides `GRPOTrainer` with native OpenEnv support. The key abstraction is the **rollout function** — it defines how the model interacts with the environment during training.

The loop:
1. `GRPOTrainer` calls your rollout function with prompts
2. Your function generates completions using the model
3. Each completion is sent as an action to the environment
4. The environment returns observations + rewards
5. TRL uses the rewards to update the model

```python
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct, reward_greens, reward_yellows],
    rollout_func=rollout_func,   # Your environment interaction
    train_dataset=dataset,
    args=grpo_config,
)
trainer.train()
```

## The Wordle Training Pipeline

We'll train Qwen3-1.7B to play Wordle using the TextArena environment.

### Environment Setup

```python
from envs.textarena_env import TextArenaEnv

env = TextArenaEnv(base_url="https://burtenshaw-textarena.hf.space")
```

The TextArena Wordle environment:
- Accepts guesses as `[word]` (5-letter words in brackets)
- Returns feedback: G (green), Y (yellow), X (gray) for each letter
- 6 attempts per game
- Reward: 1.0 for correct guess, 0.0 otherwise

### System Prompt

The system prompt guides the model's strategy:

```python
system_prompt = """
You are an expert Wordle solver.

RULES:
- Guess a 5-letter English word
- Feedback: GREEN (correct position), YELLOW (wrong position), GRAY (not in word)
- 6 attempts maximum

RESPONSE FORMAT:
Only respond with your guess in square brackets, e.g., [crane]

STRATEGY:
- Start with vowel-rich words: CRANE, SLATE, STARE
- Use GREEN letters in their positions
- Move YELLOW letters to new positions
- Eliminate GRAY letters
"""
```

### Reward Functions

Multiple reward signals give the model richer gradient information:

| Reward | What it measures | Range |
|--------|-----------------|-------|
| `reward_correct` | Did the model solve it? | 0.0 or 1.0 |
| `reward_greens` | How many green letters? | 0.0 to 1.0 |
| `reward_yellows` | How many yellow letters? | 0.0 to 1.0 |
| `reward_repetition` | Penalize repeated guesses | 0.0 to 1.0 |

Greens and yellows provide shaping signal even when the model doesn't win. Repetition penalty discourages the model from guessing the same word twice.

### The Rollout Function

The rollout function plays one full Wordle game:

```python
def rollout_once(trainer, env, tokenizer, prompt, system_prompt, max_turns):
    result = env.reset()
    observation = result.observation

    for turn in range(max_turns):
        if result.done:
            break

        # Build prompt from game state
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_game_state(observation)},
        ]

        # Generate with the model
        rollout = generate_rollout_completions(trainer, [messages])

        # Parse guess and send to environment
        guess = extract_guess(rollout["text"])
        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

    return {
        "prompt_ids": ..., "completion_ids": ..., "logprobs": ...,
        "correct_reward": ..., "green_reward": ...,
    }
```

### GRPO Configuration

```python
grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=64,
    per_device_train_batch_size=1,
    num_generations=2,
    max_completion_length=8,
    max_prompt_length=1400,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.1,
    gradient_checkpointing=True,
    report_to="trackio",
)
```

Key settings:
- **vLLM colocate mode** — generation and training share one GPU
- **gradient_accumulation_steps=64** — effective batch size without OOM
- **max_completion_length=8** — Wordle guesses are short

### Hardware

- **GPU:** A100 40GB (Colab Pro or similar)
- **Training time:** ~90 minutes
- **Peak memory:** ~37GB

## What the Model Learns

After training:
- Opens with strong words (CRANE, SLATE)
- Uses feedback to narrow down candidates
- Places confirmed letters in correct positions
- Still struggles with repeated guesses (common RL challenge)

This is a starting point. Improvements:
- Longer training runs
- Stronger repetition penalties
- Larger models (Qwen3-8B, etc.)
- Custom environments (swap Wordle for anything)

## The Key Insight

OpenEnv makes the environment a plug-in. The training pipeline stays the same — swap Wordle for your Module 4 word game, a coding environment, a math problem, or anything else. The `rollout_func` interface is the same.

## What's Next

In the [notebook](notebook.ipynb), you'll run the full Wordle GRPO training pipeline on an A100.

**Key takeaway:** OpenEnv + TRL gives you a standard way to train LLMs with environment feedback. Build the environment (Modules 1-4), plug it into GRPO, train.
