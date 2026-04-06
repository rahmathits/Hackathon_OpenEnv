"""
tools/eda_tools.py

Action executors for the EDA OpenEnv environment.
Each function takes a DataFrame and returns a result that the grader scores.
All results are dynamic — based on actual data — so graders never return
the same score twice on different datasets.
"""

import pandas as pd
import numpy as np


def execute_action(df: pd.DataFrame, action_type: str):
    """
    Execute the given action on the DataFrame and return a result.

    Actions:
        clean_data          — summary of missing values and dtypes
        eda                 — descriptive statistics
        feature_engineering — correlation matrix + new feature suggestions
        train_model         — basic model readiness report
        missing             — per-column missing value counts and percentages
        correlation         — full correlation matrix
        outliers            — IQR-based outlier detection per column
        insight             — dynamically generated insight string from real data
    """

    if action_type == "clean_data":
        missing      = df.isnull().sum()
        missing_pct  = (missing / len(df) * 100).round(2)
        dtypes       = df.dtypes.astype(str)
        return {
            "shape":            list(df.shape),
            "missing_counts":   missing.to_dict(),
            "missing_pct":      missing_pct.to_dict(),
            "dtypes":           dtypes.to_dict(),
            "duplicate_rows":   int(df.duplicated().sum()),
        }

    if action_type == "eda":
        desc = df.describe(include="all")
        return {
            "describe":         desc.to_dict(),
            "numeric_cols":     list(df.select_dtypes(include="number").columns),
            "categorical_cols": list(df.select_dtypes(include="object").columns),
            "row_count":        len(df),
            "col_count":        len(df.columns),
        }

    if action_type == "feature_engineering":
        numeric_df   = df.select_dtypes(include="number")
        corr         = numeric_df.corr().round(4).to_dict() if len(numeric_df.columns) > 1 else {}
        suggestions  = []
        for col in numeric_df.columns:
            skew = numeric_df[col].skew()
            if abs(skew) > 1:
                suggestions.append(f"log_transform({col}) — skew={skew:.2f}")
        return {
            "correlation_matrix": corr,
            "feature_suggestions": suggestions,
            "numeric_feature_count": len(numeric_df.columns),
        }

    if action_type == "train_model":
        numeric_df   = df.select_dtypes(include="number")
        missing_cols = df.columns[df.isnull().any()].tolist()
        return {
            "ready_for_training":   len(missing_cols) == 0,
            "columns_with_missing": missing_cols,
            "numeric_features":     list(numeric_df.columns),
            "sample_size":          len(df),
            "feature_count":        len(df.columns),
        }

    if action_type == "missing":
        missing     = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        cols_with_missing = missing[missing > 0]
        return {
            "missing_counts":      missing.to_dict(),
            "missing_pct":         missing_pct.to_dict(),
            "cols_with_missing":   cols_with_missing.to_dict(),
            "total_missing_cells": int(missing.sum()),
            "cols_affected":       int((missing > 0).sum()),
        }

    if action_type == "correlation":
        numeric_df = df.select_dtypes(include="number")
        if len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}
        corr   = numeric_df.corr().round(4)
        # Find strongest correlation pair
        mask   = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper  = corr.where(mask)
        max_corr_val  = upper.abs().max().max()
        max_corr_pair = upper.abs().stack().idxmax() if not upper.abs().stack().empty else None
        return {
            "correlation_matrix": corr.to_dict(),
            "max_correlation":    round(float(max_corr_val), 4),
            "max_pair":           list(max_corr_pair) if max_corr_pair else [],
            "numeric_columns":    list(numeric_df.columns),
        }

    if action_type == "outliers":
        numeric_df = df.select_dtypes(include="number")
        outlier_report = {}
        for col in numeric_df.columns:
            Q1  = numeric_df[col].quantile(0.25)
            Q3  = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            n_outliers = int(((numeric_df[col] < Q1 - 1.5 * IQR) |
                               (numeric_df[col] > Q3 + 1.5 * IQR)).sum())
            outlier_report[col] = {
                "outlier_count": n_outliers,
                "Q1": round(float(Q1), 4),
                "Q3": round(float(Q3), 4),
                "IQR": round(float(IQR), 4),
            }
        return outlier_report

    if action_type == "insight":
        return _generate_insight(df)

    return {}


def _generate_insight(df: pd.DataFrame) -> str:
    """
    Generate a dynamic insight string from the actual dataset.
    References real column names, values, and statistics so the grader
    can score it based on specificity and numeric content.
    """
    lines = []
    numeric_df = df.select_dtypes(include="number")

    # Dataset shape
    lines.append(
        f"The dataset contains {len(df)} rows and {len(df.columns)} columns "
        f"({len(numeric_df.columns)} numeric, "
        f"{len(df.select_dtypes(include='object').columns)} categorical)."
    )

    # Missing values
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        missing_info = ", ".join(
            [f"{col} ({cnt} missing)" for col, cnt in cols_with_missing.items()]
        )
        lines.append(f"Missing values found in: {missing_info}.")
    else:
        lines.append("No missing values detected — dataset is complete.")

    # Numeric stats per column
    for col in numeric_df.columns[:3]:  # limit to first 3 to keep insight concise
        mean_val = round(numeric_df[col].mean(), 2)
        min_val  = round(numeric_df[col].min(), 2)
        max_val  = round(numeric_df[col].max(), 2)
        std_val  = round(numeric_df[col].std(), 2)
        lines.append(
            f"Column '{col}': mean={mean_val}, min={min_val}, "
            f"max={max_val}, std={std_val}."
        )

    # Correlation insight
    if len(numeric_df.columns) >= 2:
        corr   = numeric_df.corr().abs()
        mask   = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper  = corr.where(mask)
        if not upper.stack().empty:
            max_val  = upper.stack().max()
            max_pair = upper.stack().idxmax()
            lines.append(
                f"Strongest correlation: '{max_pair[0]}' and '{max_pair[1]}' "
                f"with correlation={round(float(max_val), 4)}."
            )

    # Skewness insight
    skewed = []
    for col in numeric_df.columns:
        skew = numeric_df[col].skew()
        if abs(skew) > 1:
            skewed.append(f"{col} (skew={round(skew, 2)})")
    if skewed:
        lines.append(f"Highly skewed columns that may need transformation: {', '.join(skewed)}.")

    return " ".join(lines)