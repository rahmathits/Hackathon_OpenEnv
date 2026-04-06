import random
from models import Reward

# Fixed linear order — each action must be completed before the next
PIPELINE = ["clean_data", "eda", "feature_engineering", "train_model"]

# Positive reward for correct in-order action (first time only)
# Scales by pipeline position: clean_data → 0.25 ... train_model → 1.0
ORDER_BONUSES = {action: round((i + 1) / len(PIPELINE), 2) for i, action in enumerate(PIPELINE)}

# Penalty ranges for out-of-order actions (negative, randomised per violation)
# More steps skipped = larger penalty range
PENALTY_RANGES = {
    1: (-0.40, -0.20),   # skip 1 step  → random between -0.40 and -0.20
    2: (-0.65, -0.40),   # skip 2 steps → random between -0.65 and -0.40
    3: (-0.90, -0.65),   # skip 3 steps → random between -0.90 and -0.65
}

# Positive bonus range for correct in-order actions (adds slight randomness)
BONUS_JITTER = 0.05   # ± jitter applied on top of the fixed bonus


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Clamp value to [low, high] range."""
    return round(max(low, min(high, value)), 4)


def get_completed_actions(history: list) -> list:
    """Return actions that were completed successfully (no penalty)."""
    return [step["action"] for step in history if not step.get("is_penalty", False)]


def validate_action(action_type: str, history: list) -> Reward | None:
    """
    Check whether action_type is valid given the current history.

    Returns a penalty Reward with a RANDOM negative score if out-of-order.
    Returns None if valid — env will score it, then apply_order_bonus adds bonus.

    Penalty is randomised so graders never return the same score twice.
    """
    if action_type not in PIPELINE:
        return None

    current_idx = PIPELINE.index(action_type)
    completed   = get_completed_actions(history)

    missing = [
        PIPELINE[i]
        for i in range(current_idx)
        if PIPELINE[i] not in completed
    ]

    if not missing:
        return None  # All prerequisites met

    # Random penalty — scales with number of steps skipped
    steps_skipped  = min(len(missing), 3)
    low, high      = PENALTY_RANGES[steps_skipped]
    penalty        = round(random.uniform(low, high), 4)
    missing_str    = " → ".join(missing)

    feedback = (
        f"⚠️ Out-of-order: '{action_type}' requires [{missing_str}] first. "
        f"Penalty: {penalty:.4f}"
    )
    return Reward(score=penalty, feedback=feedback, is_penalty=True)


def apply_order_bonus(action_type: str, history: list, reward) -> Reward:
    """
    Called after a successful env.step() to apply a positive ordering bonus.

    - First time in-order: fixed bonus + small random jitter → always positive
    - Repeated action: pass through env score unchanged
    - Non-pipeline action: pass through unchanged

    Bonus is randomised slightly so graders never return the same score twice.
    """
    # Normalise raw float to Reward object
    if isinstance(reward, (int, float)):
        reward = Reward(score=float(reward), feedback="", is_penalty=False)

    if action_type not in PIPELINE:
        return reward

    completed = get_completed_actions(history)

    # Repeated action — return env score as-is, no bonus
    if action_type in completed:
        return Reward(
            score=round(reward.score, 4),
            feedback=reward.feedback,
            is_penalty=False,
        )

    # First time in-order — apply bonus with small random jitter
    base_bonus  = ORDER_BONUSES[action_type]
    jitter      = round(random.uniform(-BONUS_JITTER, BONUS_JITTER), 4)
    final_score = _clamp(base_bonus + jitter, low=0.01, high=1.0)  # always positive

    new_feedback = (
        f"✅ In-order +{final_score:.4f} | {reward.feedback}"
        if reward.feedback
        else f"✅ In-order +{final_score:.4f}"
    )
    return Reward(score=final_score, feedback=new_feedback, is_penalty=False)


def get_next_expected(history: list) -> str | None:
    """Return the next pipeline action the agent should take."""
    completed = get_completed_actions(history)
    for step in PIPELINE:
        if step not in completed:
            return step
    return None