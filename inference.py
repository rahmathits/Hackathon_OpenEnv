"""
inference.py

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face API key

Usage:
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    set HF_TOKEN=hf_...
    python inference.py --csv data/sample.csv
"""

import os
import re
import json
import argparse
import pandas as pd
from openai import OpenAI

from server.EDA_OpenEnv_environment import EdaOpenenvEnvironment as EDAEnv, TASKS, TASK_ACTION_MAP
from models import EdaOpenenvAction as Action
from pipeline import validate_action, apply_order_bonus, PIPELINE, get_completed_actions

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS   = 10
TEMPERATURE = 0.0
MAX_TOKENS  = 100

VALID_ACTIONS = [
    "clean_data", "eda", "feature_engineering", "train_model",
    "missing", "correlation", "insight",
]

SYSTEM_PROMPT = """You are an expert data science agent working inside an EDA environment.
Select the single best action to take next.

Pipeline order (must follow):
1. clean_data
2. eda
3. feature_engineering
4. train_model

Task-specific actions:
- detect_missing   → missing
- find_correlation → correlation
- generate_insight → insight

Respond ONLY with JSON: {"action": "<action_name>", "reason": "<one sentence>"}"""


class LLMAgent:

    def __init__(self):
        if not API_KEY:
            print("  [warn] HF_TOKEN not set — API calls may fail. Set HF_TOKEN=hf_...")
        if not MODEL_NAME:
            print("  [warn] MODEL_NAME not set — using default Qwen/Qwen2.5-72B-Instruct")

        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
        self.model  = MODEL_NAME or "Qwen/Qwen2.5-72B-Instruct"
        print(f"  API Base : {API_BASE_URL}")
        print(f"  Model    : {self.model}")

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
            print(f"  [warn] Model request failed: {exc}. Using pipeline fallback.")
            # Fallback: just follow the pipeline in order
            return next_pipeline_step if next_pipeline_step != "pipeline complete" else "missing", "fallback — API unavailable"

        try:
            clean  = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(clean)
            action = parsed.get("action", "").strip()
            reason = parsed.get("reason", "")
            if action not in VALID_ACTIONS:
                action = next_pipeline_step if next_pipeline_step != "pipeline complete" else "eda"
                reason = "fallback — invalid action"
        except json.JSONDecodeError:
            action = next_pipeline_step if next_pipeline_step != "pipeline complete" else "eda"
            reason = "fallback — JSON parse error"

        return action, reason


def run_episode(env, agent, task_override=None, verbose=True) -> dict:
    obs = env.reset()
    if task_override:
        env._task = task_override.copy()
        obs = env._get_obs()

    from models import Reward

    history      = []
    total_reward = 0.0
    step         = 0

    # Print [START] block — required by validator
    print(f"[START]")
    print(f"task={obs.task}")
    print(f"objective={obs.objective}")
    print(f"difficulty={obs.difficulty}")
    print(f"[/START]")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Task      : {obs.task}")
        print(f"  Objective : {obs.objective}")
        print(f"  Difficulty: {obs.difficulty}")
        print(f"{'─'*60}")

    while True:
        action_type, reason = agent.select_action(obs, history)

        penalty = validate_action(action_type, history)
        if penalty:
            reward = penalty
            done   = False
            if verbose:
                print(f"  Step {step+1:02d} | {action_type:<22} | score={reward.score:.4f} | ⚠️  {reward.feedback}")
        else:
            action = Action(action_type=action_type)
            obs    = env.step(action)
            done   = obs.done
            reward = apply_order_bonus(
                action_type, history,
                Reward(score=obs.reward or 0.0, feedback="", is_penalty=False)
            )
            if verbose:
                status = "✅ DONE" if done else "▶️ "
                print(f"  Step {step+1:02d} | {action_type:<22} | score={reward.score:.4f} | {status}")
                print(f"         reason  : {reason}")
                print(f"         feedback: {reward.feedback}")

        # Print [STEP] block — required by validator
        print(f"[STEP]")
        print(f"action={action_type}")
        print(f"reward={reward.score:.4f}")
        print(f"is_penalty={penalty is not None}")
        print(f"done={done}")
        print(f"[/STEP]")

        history.append({
            "action":     action_type,
            "reward":     reward.score,
            "feedback":   reward.feedback,
            "is_penalty": penalty is not None,
            "done":       done,
        })
        total_reward = round(max(0.02, min(0.98, total_reward + reward.score)), 4)
        step         += 1

        if done or step >= env.max_steps:
            break

    # total_reward is already clamped to [0.02, 0.98] during accumulation
    normalised_score = total_reward

    # Print [END] block — required by validator
    print(f"[END]")
    print(f"task={env._task['name']}")
    print(f"total_reward={normalised_score:.4f}")
    print(f"steps={step}")
    print(f"[/END]")

    if verbose:
        print(f"{'─'*60}")
        print(f"  Finished | steps={step} | total_reward={total_reward:.4f}")

    return {
        "task":             env._task["name"],
        "difficulty":       env._task["difficulty"],
        "steps":            step,
        "total_reward":     round(total_reward, 4),
        "normalised_score": normalised_score,
        "history":          [h["action"] for h in history],
        "penalties":    sum(1 for h in history if h["is_penalty"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA OpenEnv — Inference Script")
    parser.add_argument("--csv",      required=False, default=None, help="Path to CSV dataset (optional — uses built-in sample if not provided)")
    parser.add_argument("--steps",    type=int, default=10, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1,  help="Runs per task")
    parser.add_argument("--quiet",    action="store_true",  help="Suppress per-step output")
    args = parser.parse_args()

    # Use provided CSV or fall back to built-in sample dataset
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

    print(f"\n{'═'*60}")
    print(f"  EDA OpenEnv — Inference Script")
    print(f"{'═'*60}")
    print(f"  Dataset       : {dataset_name} ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  Episodes/task : {args.episodes}")

    env   = EDAEnv(df, max_steps=args.steps)
    agent = LLMAgent()

    all_results = []
    for task in TASKS:
        task_results = []
        print(f"\n{'═'*60}")
        print(f"  Task: {task['name']}  (difficulty: {task['difficulty']})")
        print(f"{'═'*60}")
        for ep in range(args.episodes):
            if args.episodes > 1:
                print(f"\n  Episode {ep+1}/{args.episodes}")
            result = run_episode(env, agent, task_override=task, verbose=not args.quiet)
            task_results.append(result)
        avg = sum(r["normalised_score"] for r in task_results) / len(task_results)
        avg = round(max(0.0001, min(0.9999, avg)), 4)
        all_results.append({**task_results[-1], "avg_reward": avg})

    print(f"\n{'═'*60}")
    print("  BASELINE SCORE SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Task':<25} {'Diff':<10} {'Steps':<8} {'Penalties':<10} {'Avg Reward'}")
    print(f"  {'─'*57}")
    total = 0.0
    for r in all_results:
        print(f"  {r['task']:<25} {r['difficulty']:<10} {r['steps']:<8} {r['penalties']:<10} {r['avg_reward']:.4f}")
        total += r["avg_reward"]
    baseline = round(total / len(all_results), 4)
    print(f"  {'─'*57}")
    print(f"  {'BASELINE SCORE':<48} {baseline:.4f}")
    print(f"{'═'*60}\n")

    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "dataset": dataset_name, "task_results": all_results, "baseline_score": baseline}, f, indent=2)
    print("Results saved → baseline_results.json\n")


if __name__ == "__main__":
    main()