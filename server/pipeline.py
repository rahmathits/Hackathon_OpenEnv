from env.models import Reward

# Fixed linear order — each action must be completed before the next
PIPELINE = ["clean_data", "eda", "feature_engineering", "train_model"]

# Reward for correct in-order action (first time only), scales by pipeline position:
# clean_data → 0.25, eda → 0.50, feature_engineering → 0.75, train_model → 1.0
ORDER_BONUSES = {action: round((i + 1) / len(PIPELINE), 2) for i, action in enumerate(PIPELINE)}

# Penalty for out-of-order action, scales by number of steps skipped:
# 1 step skipped → 0.25, 2 → 0.50, 3 → 0.75 (all clamped to [0, 1])
SKIP_PENALTY = 0.25


def _clamp(value: float) -> float:
    """Clamp a reward score to the [0, 1] range."""
    return round(max(0.0, min(1.0, value)), 4)


def get_completed_actions(history: list) -> list:
    """Return the action types that have been run successfully (no penalty)."""
    return [step["action"] for step in history if not step.get("is_penalty", False)]


def validate_action(action_type: str, history: list) -> Reward | None:
    """
    Check whether action_type is valid given what has already been completed.

    - Returns a penalty Reward (score in [0, 1]) if prerequisites are missing.
    - Returns None if valid — env will score it, then apply_order_bonus adds bonus.
    """
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
        return None  # Prerequisites met — let env run

    penalty = _clamp(len(missing) * SKIP_PENALTY)
    missing_str = " → ".join(missing)
    feedback = (
        f"⚠️ Out-of-order: '{action_type}' requires [{missing_str}] first. "
        f"Penalty: -{penalty:.2f}"
    )
    return Reward(score=penalty, feedback=feedback, is_penalty=True)


def apply_order_bonus(action_type: str, history: list, reward: Reward) -> Reward:
    """
    Called after a successful env.step() to apply a positive ordering bonus.
    Bonus is fixed per pipeline stage and the final score is clamped to [0, 1].

    clean_data          → 0.25
    eda                 → 0.50
    feature_engineering → 0.75
    train_model         → 1.00
    """
    if action_type not in PIPELINE:
        return reward

    completed = get_completed_actions(history)

    # Only bonus the first time this action is completed in order
    if action_type in completed:
        # Repeated action — just clamp whatever env returned
        return Reward(
            score=_clamp(reward.score),
            feedback=reward.feedback,
            is_penalty=False
        )

    bonus = ORDER_BONUSES[action_type]
    final_score = _clamp(bonus)
    new_feedback = f"✅ In-order +{final_score:.2f} | {reward.feedback}" if reward.feedback else f"✅ In-order +{final_score:.2f}"
    return Reward(score=final_score, feedback=new_feedback, is_penalty=False)


def get_next_expected(history: list) -> str | None:
    """Return the next action the pipeline expects, or None if pipeline is complete."""
    completed = get_completed_actions(history)
    for step in PIPELINE:
        if step not in completed:
            return step
    return None