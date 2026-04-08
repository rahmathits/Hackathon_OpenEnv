from models import Reward

# Fixed linear order — each action must be completed before the next
PIPELINE = ["clean_data", "eda", "feature_engineering", "train_model"]

# Pipeline ordering bonuses — strictly between 0 and 1
ORDER_BONUSES = {
    "clean_data":          0.2500,
    "eda":                 0.5000,
    "feature_engineering": 0.7500,
    "train_model":         0.9500,   # was 1.0 — must be strictly less than 1
}

# Out-of-order penalties — fixed, deterministic, scale by steps skipped
# These penalise wrong order during training (not used in final grading)
SKIP_PENALTIES = {
    1: -0.25,
    2: -0.50,
    3: -0.75,
}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Clamp value to (low, high) range — strictly exclusive of boundaries."""
    return round(max(low + 0.0001, min(high - 0.0001, value)), 4)


def get_completed_actions(history: list) -> list:
    """Return actions that were completed successfully (no penalty)."""
    return [step["action"] for step in history if not step.get("is_penalty", False)]


def validate_action(action_type: str, history: list) -> Reward | None:
    """
    Check whether action_type is valid given the current history.

    Returns a deterministic penalty Reward if out-of-order.
    Returns None if valid — env scores it, then apply_order_bonus adds bonus.

    Penalties are fixed and deterministic — same input always gives same output.
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

    steps_skipped = min(len(missing), 3)
    penalty       = SKIP_PENALTIES[steps_skipped]
    missing_str   = " → ".join(missing)

    feedback = (
        f"⚠️ Out-of-order: '{action_type}' requires [{missing_str}] first. "
        f"Penalty: {penalty:.2f}"
    )
    return Reward(score=penalty, feedback=feedback, is_penalty=True)


def apply_order_bonus(action_type: str, history: list, reward) -> Reward:
    """
    Called after a successful env.step() to apply a positive ordering bonus.
    Bonuses are fixed and deterministic — same action always gives same bonus.

    - First time in-order: fixed bonus from ORDER_BONUSES
    - Repeated action: pass through env score unchanged
    - Non-pipeline action: pass through unchanged
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

    # First time in-order — apply fixed deterministic bonus
    final_score  = _clamp(ORDER_BONUSES[action_type], low=0.0, high=1.0)
    new_feedback = (
        f"✅ In-order +{final_score:.2f} | {reward.feedback}"
        if reward.feedback
        else f"✅ In-order +{final_score:.2f}"
    )
    return Reward(score=final_score, feedback=new_feedback, is_penalty=False)


def get_next_expected(history: list) -> str | None:
    """Return the next pipeline action the agent should take."""
    completed = get_completed_actions(history)
    for step in PIPELINE:
        if step not in completed:
            return step
    return None