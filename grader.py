"""
grader.py

Dynamic, deterministic graders for each task.

Rules:
- Same task + same data + same actions = same score (deterministic)
- Score always in [0.0, 1.0]
- No randomness — graders measure actual data quality
"""

import pandas as pd
from typing import Any


def _strict(value: float) -> float:
    """Ensure score is strictly between 0.02 and 0.98 to avoid float precision issues."""
    return round(max(0.02, min(0.98, value)), 4)


def grade_task(task_name: str, df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    graders = {
        "detect_missing":   _grade_detect_missing,
        "find_correlation": _grade_find_correlation,
        "generate_insight": _grade_generate_insight,
    }
    grader = graders.get(task_name)
    if grader is None:
        return 0.02, f"Unknown task: '{task_name}'"
    return grader(df, history, result)


def _grade_detect_missing(df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    if "missing" not in history:
        return 0.02, "Agent never ran 'missing' action."

    actual_missing    = df.isnull().sum().sum()
    total_cols        = len(df.columns)
    cols_with_missing = int((df.isnull().sum() > 0).sum())

    if actual_missing == 0:
        return 0.9999, "No missing values exist. Agent correctly investigated."

    coverage = round(cols_with_missing / total_cols, 4) if total_cols > 0 else 0.0
    score    = _strict(0.5 + 0.5 * coverage)

    return score, (
        f"Found {cols_with_missing}/{total_cols} columns with missing values. "
        f"Total missing cells: {actual_missing}. Score: {score:.4f}"
    )


def _grade_find_correlation(df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    if "correlation" not in history:
        return 0.02, "Agent never ran 'correlation' action."

    try:
        numeric_df  = df.select_dtypes(include="number")
        if len(numeric_df.columns) < 2:
            return 0.10, "Not enough numeric columns for correlation."

        corr_matrix = numeric_df.corr(numeric_only=True).abs()
        for col in corr_matrix.columns:
            corr_matrix.loc[col, col] = 0.0
        max_corr = float(corr_matrix.max().max())

    except Exception as e:
        return 0.10, f"Correlation computation failed: {e}"

    if max_corr >= 0.9:
        score, note = 0.9500, "very strong correlation found"
    elif max_corr >= 0.7:
        score, note = 0.7500, "strong correlation found"
    elif max_corr >= 0.5:
        score, note = 0.5500, "moderate correlation found"
    elif max_corr >= 0.3:
        score, note = 0.3500, "weak correlation found"
    else:
        score, note = 0.1500, "no meaningful correlation found"

    return _strict(score), f"Max correlation: {max_corr:.4f} — {note}. Score: {score:.4f}"


def _grade_generate_insight(df: pd.DataFrame, history: list, result: Any) -> tuple[float, str]:
    if "insight" not in history:
        return 0.02, "Agent never ran 'insight' action."

    if not isinstance(result, str) or len(result.strip()) == 0:
        return 0.02, "Insight result is empty or not a string."

    text  = result.strip()
    score = 0.0
    notes = []

    length_score = round(min(len(text) / 200, 1.0) * 0.4, 4)
    score       += length_score
    notes.append(f"length={len(text)} chars (+{length_score:.4f})")

    import re
    numeric_count = len(re.findall(r"\d+\.?\d*", text))
    numeric_score = round(min(numeric_count / 5, 1.0) * 0.3, 4)
    score        += numeric_score
    notes.append(f"numeric_refs={numeric_count} (+{numeric_score:.4f})")

    col_mentions = sum(1 for col in df.columns if col.lower() in text.lower())
    col_score    = round(min(col_mentions / max(len(df.columns), 1), 1.0) * 0.3, 4)
    score       += col_score
    notes.append(f"col_refs={col_mentions}/{len(df.columns)} (+{col_score:.4f})")

    return _strict(score), f"Insight graded: {'; '.join(notes)}. Total: {_strict(score):.4f}"