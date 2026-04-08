import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pandas as pd
from typing import Optional, Dict, Any

from openenv.core import Environment

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


class EdaOpenenvEnvironment(Environment[EdaOpenenvAction, EdaOpenenvObservation, Dict[str, Any]]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, df: pd.DataFrame = None, max_steps: int = 10):
        super().__init__()
        self.df                = df if df is not None else _DEFAULT_DF.copy()
        self.max_steps         = max_steps
        self.history           = []
        self.step_history      = []
        self._steps            = 0
        self._done             = False
        self._task             = random.choice(TASKS).copy()  # auto-init so step() works before reset()
        self.cumulative_reward = 0.0

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> EdaOpenenvObservation:
        # Handle dict input from web UI
        if isinstance(seed, dict):
            data       = seed
            seed       = data.get("seed", None)
            episode_id = data.get("episode_id", None)
        elif isinstance(seed, str):
            seed = None

        self._reset_rubric()
        self.history           = []
        self.step_history      = []
        self._steps            = 0
        self._done             = False
        self.cumulative_reward = 0.0

        # Deterministic task selection:
        # - If seed provided → always picks same task for same seed
        # - If task_name in kwargs → use that specific task
        # - Otherwise → random
        task_name = kwargs.get("task_name", None)
        if task_name:
            matched = next((t for t in TASKS if t["name"] == task_name), None)
            self._task = matched.copy() if matched else random.choice(TASKS).copy()
        elif seed is not None:
            random.seed(seed)
            self._task = random.choice(TASKS).copy()
            random.seed(None)  # reset seed so other randomness is unaffected
        else:
            self._task = random.choice(TASKS).copy()

        return self._get_obs(reward=None, done=False)

    # ─────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────
    def step(
        self,
        action: EdaOpenenvAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> EdaOpenenvObservation:
        # Handle dict input from web UI — extract action_type from any dict shape
        if isinstance(action, dict):
            action_type = (
                action.get("action_type")
                or action.get("action")
                or action.get("message")
                or "clean_data"
            )
            action = EdaOpenenvAction(action_type=action_type)
        if self._done:
            return self._get_obs(reward=0.0, done=True)

        # Pipeline ordering check
        penalty = validate_action(action.action_type, self.step_history)
        if penalty:
            self.step_history.append({
                "action":     action.action_type,
                "reward":     penalty.score,
                "is_penalty": True,
                "done":       False,
            })
            self._steps += 1
            return self._get_obs(reward=penalty.score, done=False)

        # Compute reward
        reward_obj = self._compute_reward(action)
        reward_obj = apply_order_bonus(action.action_type, self.step_history, reward_obj)

        self.history.append(action.action_type)
        self.step_history.append({
            "action":     action.action_type,
            "reward":     reward_obj.score,
            "is_penalty": False,
            "done":       self._done,
        })
        self._steps            += 1
        self.cumulative_reward  = round(self.cumulative_reward + reward_obj.score, 4)

        if self._steps >= self.max_steps:
            self._done = True

        obs = self._get_obs(reward=reward_obj.score, done=self._done)
        obs.reward = self._apply_rubric(action, obs) or reward_obj.score
        return obs

    # ─────────────────────────────────────────
    # STATE — must be a property
    # ─────────────────────────────────────────
    @property
    def state(self) -> Dict[str, Any]:
        from openenv.core.env_server.http_server import State

        class EdaState(State):
            task:              str | None = None
            objective:         str | None = None
            difficulty:        str | None = None
            max_steps:         int        = 10
            history:           list       = []
            cumulative_reward: float      = 0.0
            done:              bool       = False

        return EdaState(
            episode_id=None,
            step_count=self._steps,
            task=self._task["name"] if self._task else None,
            objective=self._task["objective"] if self._task else None,
            difficulty=self._task["difficulty"] if self._task else None,
            max_steps=self.max_steps,
            history=self.history.copy(),
            cumulative_reward=self.cumulative_reward,
            done=self._done,
        )

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────
    def _get_obs(self, reward: float = None, done: bool = False) -> EdaOpenenvObservation:
        # Clamp reward strictly between 0 and 1 exclusive if provided
        if reward is not None:
            reward = round(max(0.0001, min(0.9999, reward)), 4)
        return EdaOpenenvObservation(
            done=done,
            reward=reward,
            dataset_head=self.df.head().to_dict(orient="records"),
            columns=list(self.df.columns),
            stats=self.df.describe().to_dict(),
            history=self.history.copy(),
            task=self._task["name"] if self._task else "",
            objective=self._task["objective"] if self._task else "",
            difficulty=self._task["difficulty"] if self._task else "",
        )

    def _compute_reward(self, action: EdaOpenenvAction) -> Reward:
        if action.action_type in self.history:
            return Reward(score=0.0500, feedback="Repeated action.", is_penalty=True)

        expected = TASK_ACTION_MAP.get(self._task["name"])
        if action.action_type == expected:
            grade, feedback = grade_task(
                task_name=self._task["name"],
                df=self.df,
                history=self.history + [action.action_type],
                result=None,
            )
            if grade >= 0.9999:
                self._done = True
                return Reward(score=0.9999, feedback=f"Task complete! {feedback}", is_penalty=False)
            return Reward(score=round(max(0.0001, min(0.9999, grade)), 4), feedback=feedback, is_penalty=False)

        return Reward(
            score=0.1500,
            feedback=f"'{action.action_type}' valid but not relevant to task '{self._task['name']}'.",
            is_penalty=False,
        )