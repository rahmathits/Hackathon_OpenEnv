"""
env/grader.py

Dynamic graders for each task. Each grader returns a score in [0, 1]
based on the actual quality of the result — not a fixed value.
"""

import pandas as pd
from typing import Any


def grade_task(task_name: str, df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    """
    Route to the correct grader based on task name.
    Returns (score: float, feedback: str), score in [0, 1].
    """
    graders = {
        "detect_missing":   _grade_detect_missing,
        "find_correlation": _grade_find_correlation,
        "generate_insight": _grade_generate_insight,
    }
    grader = graders.get(task_name)
    if grader is None:
        return 0.0, f"Unknown task: {task_name}"
    return grader(df, history, result)


# ─────────────────────────────────────────
# Task 1 — detect_missing (easy)
# Grades on: did the agent find missing values,
# and how accurately does the result reflect reality?
# ─────────────────────────────────────────
def _grade_detect_missing(df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    actual_missing = df.isnull().sum().sum()

    if actual_missing == 0:
        # No missing values exist — agent gets full score if it ran the right action
        if "missing" in history:
            return 1.0, "No missing values exist; agent correctly investigated."
        return 0.3, "No missing values exist and agent skipped investigation."

    # Missing values exist — did agent use the right action?
    if "missing" not in history:
        return 0.0, f"Dataset has {actual_missing} missing values but agent never ran 'missing' action."

    # Partial credit based on how many columns were identified
    total_cols = len(df.columns)
    cols_with_missing = (df.isnull().sum() > 0).sum()
    coverage = cols_with_missing / total_cols if total_cols > 0 else 0

    score = round(0.5 + 0.5 * coverage, 4)
    return score, f"Found missing in {cols_with_missing}/{total_cols} columns. Score: {score:.2f}"


# ─────────────────────────────────────────
# Task 2 — find_correlation (medium)
# Grades on: strength of the highest correlation found
# and whether the agent used the correlation action.
# ─────────────────────────────────────────
def _grade_find_correlation(df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    if "correlation" not in history:
        return 0.0, "Agent never ran 'correlation' action."

    try:
        corr_matrix = df.corr(numeric_only=True).abs()
        # Exclude self-correlations (diagonal = 1.0)
        corr_matrix.values[[range(len(corr_matrix))] * 2] = 0
        max_corr = corr_matrix.max().max()
    except Exception as e:
        return 0.1, f"Correlation computation failed: {e}"

    if max_corr >= 0.9:
        score, note = 1.0, "very strong correlation found"
    elif max_corr >= 0.7:
        score, note = 0.8, "strong correlation found"
    elif max_corr >= 0.5:
        score, note = 0.6, "moderate correlation found"
    elif max_corr >= 0.3:
        score, note = 0.4, "weak correlation found"
    else:
        score, note = 0.2, "no meaningful correlation found"

    return score, f"Max correlation: {max_corr:.3f} — {note}."


# ─────────────────────────────────────────
# Task 3 — generate_insight (hard)
# Grades on: length, specificity, and numeric content
# of the generated insight string.
# ─────────────────────────────────────────
def _grade_generate_insight(df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    if "insight" not in history:
        return 0.0, "Agent never ran 'insight' action."

    if not isinstance(result, str) or len(result.strip()) == 0:
        return 0.0, "Insight result is empty or not a string."

    text = result.strip()
    score = 0.0
    notes = []

    # Length — minimum viable insight is 30 chars, full credit at 200+
    length_score = min(len(text) / 200, 1.0) * 0.4
    score += length_score
    notes.append(f"length={len(text)} chars (+{length_score:.2f})")

    # Numeric content — insights should reference actual data values
    import re
    numeric_count = len(re.findall(r"\d+\.?\d*", text))
    numeric_score = min(numeric_count / 5, 1.0) * 0.3
    score += numeric_score
    notes.append(f"numeric refs={numeric_count} (+{numeric_score:.2f})")

    # Column name mentions — insight should reference actual dataset columns
    col_mentions = sum(1 for col in df.columns if col.lower() in text.lower())
    col_score = min(col_mentions / max(len(df.columns), 1), 1.0) * 0.3
    score += col_score
    notes.append(f"column refs={col_mentions} (+{col_score:.2f})")

    score = round(min(score, 1.0), 4)
    return score, f"Insight graded: {'; '.join(notes)}. Total: {score:.2f}"