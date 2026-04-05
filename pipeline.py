from models import Reward

# Fixed linear order — each action must be completed before the next
PIPELINE = ["clean_data", "eda", "feature_engineering", "train_model"]

# Reward for correct in-order action (first time only), scales by pipeline position:
# clean_data → 0.25, eda → 0.50, feature_engineering → 0.75, train_model → 1.0
ORDER_BONUSES = {action: round((i + 1) / len(PIPELINE), 2) for i, action in enumerate(PIPELINE)}

# Penalty for out-of-order action, scales by number of steps skipped
SKIP_PENALTY = 0.25


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def get_completed_actions(history: list) -> list:
    return [step["action"] for step in history if not step.get("is_penalty", False)]


def validate_action(action_type: str, history: list) -> Reward | None:
    if action_type not in PIPELINE:
        return None

    current_idx = PIPELINE.index(action_type)
    completed = get_completed_actions(history)

    missing = [
        PIPELINE[i]
        for i in range(current_idx)
        if PIPELINE[i] not in completed
    ]

    if not missing:
        return None

    penalty = _clamp(len(missing) * SKIP_PENALTY)
    missing_str = " → ".join(missing)
    feedback = (
        f"⚠️ Out-of-order: '{action_type}' requires [{missing_str}] first. "
        f"Penalty: -{penalty:.2f}"
    )
    return Reward(score=penalty, feedback=feedback, is_penalty=True)


def apply_order_bonus(action_type: str, history: list, reward) -> Reward:
    """
    Accept either a Reward object or a raw float for robustness.
    Applies a positive ordering bonus for first-time in-order actions.
    """
    # Normalise raw float to Reward object
    if isinstance(reward, (int, float)):
        reward = Reward(score=float(reward), feedback="", is_penalty=False)

    if action_type not in PIPELINE:
        return reward

    completed = get_completed_actions(history)

    # Repeated action — clamp env score, no bonus
    if action_type in completed:
        return Reward(
            score=_clamp(reward.score),
            feedback=reward.feedback,
            is_penalty=False
        )

    bonus = ORDER_BONUSES[action_type]
    final_score = _clamp(bonus)
    new_feedback = (
        f"✅ In-order +{final_score:.2f} | {reward.feedback}"
        if reward.feedback
        else f"✅ In-order +{final_score:.2f}"
    )
    return Reward(score=final_score, feedback=new_feedback, is_penalty=False)


def get_next_expected(history: list) -> str | None:
    completed = get_completed_actions(history)
    for step in PIPELINE:
        if step not in completed:
            return step
    return None