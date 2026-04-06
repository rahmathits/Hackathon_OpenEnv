from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

from openenv.core import Action as BaseAction, Observation as BaseObservation


# ─────────────────────────────────────────
# Action — inherits from openenv-core BaseAction
# Web UI only shows fields defined HERE
# ─────────────────────────────────────────
class EdaOpenenvAction(BaseAction):
    action_type: str = "clean_data"


# ─────────────────────────────────────────
# Observation — inherits from openenv-core BaseObservation
# All our env data goes into metadata dict
# ─────────────────────────────────────────
class EdaOpenenvObservation(BaseObservation):
    # Core openenv fields (from base): done, reward, metadata
    # Our EDA-specific fields:
    dataset_head: List[Dict]   = []
    columns:      List[str]    = []
    stats:        Dict[str, Any] = {}
    history:      List[str]    = []
    task:         str          = ""
    objective:    str          = ""
    difficulty:   str          = ""


# Aliases
Observation = EdaOpenenvObservation
Action      = EdaOpenenvAction


# ─────────────────────────────────────────
# Reward — internal use only (not sent to web UI)
# ─────────────────────────────────────────
class Reward(BaseModel):
    score:      float
    feedback:   str
    is_penalty: bool = False