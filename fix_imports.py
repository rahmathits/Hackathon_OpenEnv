"""
fix_imports.py

Run this from your EDA_OpenEnv root directory to fix all import issues.
    python fix_imports.py
"""

import os

server_init = '''import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
'''

server_app = '''import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

from models import EdaOpenenvAction, EdaOpenenvObservation
from server.EDA_OpenEnv_environment import EdaOpenenvEnvironment

app = create_app(
    EdaOpenenvEnvironment,
    EdaOpenenvAction,
    EdaOpenenvObservation,
    env_name="EDA_OpenEnv",
    max_concurrent_envs=1,
    base_path="/web",
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
'''

server_env = '''import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pandas as pd
from typing import Tuple, Dict, Any

from models import EdaOpenenvAction, EdaOpenenvObservation, Reward
from pipeline import validate_action, apply_order_bonus, PIPELINE
from grader import grade_task

TASK_ACTION_MAP = {
    "detect_missing":   "missing",
    "find_correlation": "correlation",
    "generate_insight": "insight",
}

TASKS = [
    {"name": "detect_missing",   "difficulty": "easy",   "objective": "Identify all columns with missing values in the dataset."},
    {"name": "find_correlation", "difficulty": "medium", "objective": "Find the strongest correlation between any two numeric columns."},
    {"name": "generate_insight", "difficulty": "hard",   "objective": "Generate a meaningful insight referencing actual data values and column names."},
]

_DEFAULT_DF = pd.DataFrame({
    "age":    [25, 30, None, 45, 22],
    "salary": [50000, 60000, 70000, None, 45000],
    "score":  [88, 92, 75, 85, 90],
})


class EdaOpenenvEnvironment:

    def __init__(self, df: pd.DataFrame = None, max_steps: int = 10):
        self.df = df if df is not None else _DEFAULT_DF.copy()
        self.max_steps = max_steps
        self.history = []
        self.step_history = []
        self.steps = 0
        self.done = False
        self.task = None
        self.cumulative_reward = 0.0

    def reset(self) -> EdaOpenenvObservation:
        self.history = []
        self.step_history = []
        self.steps = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.task = random.choice(TASKS).copy()
        return self._get_obs()

    def _get_obs(self) -> EdaOpenenvObservation:
        return EdaOpenenvObservation(
            dataset_head=self.df.head().to_dict(orient="records"),
            columns=list(self.df.columns),
            stats=self.df.describe().to_dict(),
            history=self.history.copy(),
            task=self.task["name"],
            objective=self.task["objective"],
            difficulty=self.task["difficulty"],
        )

    def step(self, action: EdaOpenenvAction) -> Tuple[EdaOpenenvObservation, float, bool, dict]:
        if self.done:
            return self._get_obs(), 0.0, True, {"info": "Episode done. Call reset()."}

        penalty = validate_action(action.action_type, self.step_history)
        if penalty:
            self.step_history.append({"action": action.action_type, "reward": penalty.score, "is_penalty": True, "done": False})
            self.steps += 1
            return self._get_obs(), penalty.score, False, {"feedback": penalty.feedback, "is_penalty": True}

        reward_obj = self._compute_reward(action)
        reward_obj = apply_order_bonus(action.action_type, self.step_history, reward_obj)

        self.history.append(action.action_type)
        self.step_history.append({"action": action.action_type, "reward": reward_obj.score, "is_penalty": False, "done": self.done})
        self.steps += 1
        self.cumulative_reward = round(self.cumulative_reward + reward_obj.score, 4)

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward_obj.score, self.done, {
            "task": self.task["name"],
            "feedback": reward_obj.feedback,
            "cumulative_reward": self.cumulative_reward,
        }

    def _compute_reward(self, action: EdaOpenenvAction) -> Reward:
        if action.action_type in self.history:
            return Reward(score=0.1, feedback=f"Repeated action.", is_penalty=True)

        expected = TASK_ACTION_MAP.get(self.task["name"])
        if action.action_type == expected:
            grade, feedback = grade_task(
                task_name=self.task["name"],
                df=self.df,
                history=self.history + [action.action_type],
                result=None,
            )
            if grade >= 1.0:
                self.done = True
                return Reward(score=1.0, feedback=f"Task complete! {feedback}", is_penalty=False)
            return Reward(score=round(grade, 4), feedback=feedback, is_penalty=False)

        return Reward(score=0.2, feedback=f"Valid but not relevant to task \'{self.task[\'name\']}\'.", is_penalty=False)

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task["name"] if self.task else None,
            "objective": self.task["objective"] if self.task else None,
            "difficulty": self.task["difficulty"] if self.task else None,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "history": self.history.copy(),
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
        }
'''

# Write all files
os.makedirs("server", exist_ok=True)

files = {
    "server/__init__.py":                  server_init,
    "server/app.py":                       server_app,
    "server/EDA_OpenEnv_environment.py":   server_env,
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Written: {path}")

print("\nAll files updated. Now run:")
print("  docker build -t eda-openenv:latest . --no-cache --progress=plain")