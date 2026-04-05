from typing import List, Dict, Any
from pydantic import BaseModel


class EdaOpenenvObservation(BaseModel):
    dataset_head: List[Dict]
    columns: List[str]
    stats: Dict[str, Any]
    history: List[str]
    task: str
    objective: str = ""
    difficulty: str = ""


class EdaOpenenvAction(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


# Aliases for backward compatibility
Observation = EdaOpenenvObservation
Action = EdaOpenenvAction


class Reward(BaseModel):
    score: float
    feedback: str
    is_penalty: bool = False