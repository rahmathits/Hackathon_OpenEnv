"""
client.py

Client SDK for EDA OpenEnv.
Uses the openenv-core EnvClient pattern for remote environment access.

Usage:
    from EDA_OpenEnv import EdaOpenenvAction, EdaOpenenvEnv

    EDA_OPENENV_URL = "https://rahmath1-openenv.hf.space"

    with EdaOpenenvEnv(base_url=EDA_OPENENV_URL) as env:
        result = env.reset()
        print("Task:", result.observation.task)
        print("Objective:", result.observation.objective)

        actions = ["clean_data", "eda", "feature_engineering", "train_model"]
        for action in actions:
            if result.done:
                break
            result = env.step(EdaOpenenvAction(action_type=action))
            print(f"Action  : {action}")
            print(f"Reward  : {result.reward:.4f}")
            print(f"Done    : {result.done}")
"""

import requests
from dataclasses import dataclass, field
from typing import Dict, Any

from models import EdaOpenenvAction, EdaOpenenvObservation


@dataclass
class EnvResult:
    observation: EdaOpenenvObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class EdaOpenenvEnv:
    """
    HTTP client for the EDA OpenEnv API.
    Compatible with the openenv-core EnvClient interface.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to EDA OpenEnv at {self.base_url}\n"
                f"Make sure the HF Space is running.\nError: {e}"
            )

    def reset(self) -> EnvResult:
        r = requests.post(f"{self.base_url}/reset", json={}, timeout=self.timeout)
        self._raise(r)
        return self._parse(r.json())

    def step(self, action: EdaOpenenvAction) -> EnvResult:
        r = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        self._raise(r)
        return self._parse(r.json())

    def state(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        self._raise(r)
        return r.json()

    def close(self):
        pass  # stateless HTTP — nothing to close

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _raise(response):
        if not response.ok:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

    @staticmethod
    def _parse(data: dict) -> EnvResult:
        return EnvResult(
            observation=EdaOpenenvObservation(**data["observation"]),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )