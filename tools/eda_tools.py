import pandas as pd

def execute_action(df, action_type):

    if action_type == "describe":
        return df.describe().to_dict()

    if action_type == "missing":
        return df.isnull().sum().to_dict()

    if action_type == "correlation":
        return df.corr(numeric_only=True).to_dict()

    if action_type == "outliers":
        return "outliers detected"

    if action_type == "insight":
        return "dataset has trends and patterns"

    return {}