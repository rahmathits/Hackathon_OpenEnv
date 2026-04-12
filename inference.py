"""
inference.py
===================================
MANDATORY environment variables:
    API_BASE_URL     The API endpoint for the LLM
    MODEL_NAME       The model identifier
    HF_TOKEN         Your Hugging Face API key

STDOUT FORMAT (single lines, flush=True):
    [START] task=<task> env=eda_openenv model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import re
import json
import argparse
import pandas as pd
from typing import List, Optional
from openai import OpenAI

from server.EDA_OpenEnv_environment import EdaOpenenvEnvironment as EDAEnv, TASKS, TASK_ACTION_MAP
from models import EdaOpenenvAction as Action, Reward
from pipeline import validate_action, apply_order_bonus, PIPELINE, get_completed_actions

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "eda_openenv"

MAX_STEPS              = 10
TEMPERATURE            = 0.0
MAX_TOKENS             = 100
SUCCESS_SCORE_THRESHOLD = 0.1

VALID_ACTIONS = [
    "clean_data", "eda", "feature_engineering", "train_model",
    "missing", "correlation", "insight",
]

SYSTEM_PROMPT = """You are an expert data science agent working inside an EDA environment.
Select the single best action to take next.

Pipeline order (must follow in order):
1. clean_data
2. eda
3. feature_engineering
4. train_model

Task-specific actions (run after pipeline):
- detect_missing   → missing
- find_correlation → correlation
- generate_insight → insight

Respond ONLY with JSON: {"action": "<action_name>", "reason": "<one sentence>"}"""


# ─────────────────────────────────────────
# Logging — single-line format, flush=True
# ─────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _safe_score(value) -> float:
    """Convert any value to float strictly between 0.02 and 0.98."""
    try:
        f = float(value) if value is not None else 0.5
    except (TypeError, ValueError):
        f = 0.5
    return round(max(0.02, min(0.98, f)), 4)


# ─────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────
class LLMAgent:

    def __init__(self):
        if not API_KEY:
            print("  [warn] HF_TOKEN not set — API calls may fail.", flush=True)
        if not MODEL_NAME:
            print("  [warn] MODEL_NAME not set — using default.", flush=True)
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
        self.model  = MODEL_NAME or "Qwen/Qwen2.5-72B-Instruct"

    def select_action(self, obs, history: list) -> tuple[str, str]:
        completed          = get_completed_actions(history)
        next_pipeline_step = next((s for s in PIPELINE if s not in completed), "pipeline complete")

        user_message = f"""Task     : {obs.task}
Columns  : {obs.columns}
History  : {obs.history}
Completed: {completed}
Next step: {next_pipeline_step}

Dataset (first 5 rows):
{json.dumps(obs.dataset_head, indent=2)}

What is the single best action to take next?"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
            # Fallback: follow pipeline then task-specific action
            if next_pipeline_step != "pipeline complete":
                return next_pipeline_step, "fallback"
            return TASK_ACTION_MAP.get(obs.task, "missing"), "fallback — task action"

        try:
            clean  = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(clean)
            action = parsed.get("action", "").strip()
            reason = parsed.get("reason", "")
            if action not in VALID_ACTIONS:
                action = next_pipeline_step if next_pipeline_step != "pipeline complete" else TASK_ACTION_MAP.get(obs.task, "missing")
                reason = "fallback — invalid action"
        except json.JSONDecodeError:
            action = next_pipeline_step if next_pipeline_step != "pipeline complete" else TASK_ACTION_MAP.get(obs.task, "missing")
            reason = "fallback — parse error"

        return action, reason


# ─────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────
def run_episode(env: EDAEnv, agent: LLMAgent, task_override: dict = None) -> dict:
    obs = env.reset()
    if task_override:
        env._task = task_override.copy()
        obs = env._get_obs()

    task_name = obs.task
    history   = []
    rewards   = []
    step      = 0
    done      = False
    success   = False
    score     = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            action_type, reason = agent.select_action(obs, history)

            # Pipeline validation
            penalty = validate_action(action_type, history)
            if penalty:
                raw_reward = _safe_score(penalty.score)
                error      = penalty.feedback
            else:
                action     = Action(action_type=action_type)
                obs        = env.step(action)
                done       = obs.done
                r          = apply_order_bonus(
                    action_type, history,
                    Reward(score=_safe_score(obs.reward), feedback="", is_penalty=False)
                )
                raw_reward = _safe_score(r.score)
                error      = None

            rewards.append(raw_reward)
            step = step_num

            log_step(step=step_num, action=action_type, reward=raw_reward, done=done, error=error)

            history.append({
                "action":     action_type,
                "reward":     raw_reward,
                "is_penalty": penalty is not None,
                "done":       done,
            })

        # Calculate final score — sum of rewards normalised to [0,1]
        # Use average reward across steps so score is naturally in (0,1)
        if rewards:
            score = sum(rewards) / len(rewards)
        else:
            score = 0.5

        # Clamp strictly between 0.02 and 0.98
        score   = _safe_score(score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=step, score=score, rewards=rewards)

    return {
        "task":       task_name,
        "difficulty": env._task["difficulty"],
        "steps":      step,
        "score":      score,
        "success":    success,
        "rewards":    rewards,
        "penalties":  sum(1 for h in history if h["is_penalty"]),
    }


# ─────────────────────────────────────────
# Main — runs all 3 tasks
# ─────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="EDA OpenEnv — Inference Script")
    parser.add_argument("--csv",      required=False, default=None, help="Path to CSV (optional)")
    parser.add_argument("--steps",    type=int, default=10,         help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1,          help="Runs per task")
    args = parser.parse_args()

    # Load dataset
    if args.csv:
        df           = pd.read_csv(args.csv)
        dataset_name = args.csv
    else:
        df = pd.DataFrame({
            "age":        [25, 30, None, 45, 22, 35, 28, None, 50, 33],
            "salary":     [50000, 60000, 70000, None, 45000, 80000, 55000, 62000, None, 72000],
            "score":      [88, 92, 75, 85, 90, 78, 95, 82, 88, 91],
            "experience": [2, 5, 8, 15, 1, 10, 3, 6, 20, 9],
            "department": ["HR", "IT", "IT", "Finance", "HR", "IT", "Finance", "HR", "IT", "Finance"],
        })
        dataset_name = "built-in sample dataset"

    env   = EDAEnv(df, max_steps=args.steps)
    agent = LLMAgent()

    all_results = []
    for task in TASKS:
        for ep in range(args.episodes):
            result = run_episode(env, agent, task_override=task)
            all_results.append(result)

    # Save results
    baseline_score = sum(r["score"] for r in all_results) / len(all_results)
    output = {
        "model":          MODEL_NAME,
        "dataset":        dataset_name,
        "task_results":   all_results,
        "baseline_score": round(baseline_score, 4),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()