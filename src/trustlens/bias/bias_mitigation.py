from __future__ import annotations

import numpy as np
import pandas as pd
from trustlens.models import ModelRegistry

def calculate_reweighing_weights(y_train: pd.Series, sens_train: pd.Series) -> pd.Series:
    """
    Calculate sample weights for Reweighing (Pre-processing).
    Formula: W(a, y) = P(A=a) * P(Y=y) / P(A=a, Y=y)
    """
    y_arr = np.asarray(y_train)
    s_arr = np.asarray(sens_train)
    n = len(y_arr)
    
    weights = np.ones(n)
    
    # Calculate group probabilities
    p_a = {
        0: np.mean(s_arr == 0),
        1: np.mean(s_arr == 1)
    }
    p_y = {
        0: np.mean(y_arr == 0),
        1: np.mean(y_arr == 1)
    }
    
    # Calculate joint probabilities P(A=a, Y=y)
    p_ay = {}
    for a in [0, 1]:
        for y in [0, 1]:
            joint_count = np.sum((s_arr == a) & (y_arr == y))
            p_ay[(a, y)] = joint_count / n if joint_count > 0 else 0.0

    # Assign weights
    for idx in range(n):
        a = s_arr[idx]
        y = y_arr[idx]
        joint_p = p_ay.get((a, y), 0.0)
        
        if joint_p > 0:
            weights[idx] = (p_a[a] * p_y[y]) / joint_p
        else:
            weights[idx] = 1.0
            
    return pd.Series(weights, index=y_train.index)


def train_fairlearn_reduction(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sens_train: pd.Series,
    constraint_type: str
):
    """
    Train a model using Fairlearn Exponentiated Gradient (In-processing).
    Constraints supported: Demographic Parity, Equal Opportunity, Equalized Odds.
    """
    from fairlearn.reductions import ExponentiatedGradient
    from fairlearn.reductions import DemographicParity, EqualizedOdds, TruePositiveRateParity
    
    base_estimator = ModelRegistry.get_estimator(model_type)
    
    # Map constraints
    if constraint_type == "Demographic Parity":
        constraint = DemographicParity()
    elif constraint_type == "Equal Opportunity":
        # In Fairlearn, Equal Opportunity is TruePositiveRateParity
        constraint = TruePositiveRateParity()
    elif constraint_type == "Equalized Odds":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"Unsupported Fairlearn constraint: {constraint_type}")
        
    mitigator = ExponentiatedGradient(base_estimator, constraints=constraint)
    
    # ExponentiatedGradient needs numerical labels, ensure y_train is binary numeric
    y_numeric = y_train.astype(int)
    s_numeric = sens_train.astype(int)
    
    mitigator.fit(X_train, y_numeric, sensitive_features=s_numeric)
    return mitigator


def train_fairlearn_postprocessing(
    fitted_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sens_train: pd.Series,
    constraint_type: str
):
    """
    Apply Fairlearn Threshold Optimizer (Post-processing) on a pre-fit baseline model.
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    
    # Map constraints
    if constraint_type == "Demographic Parity":
        constraint = "demographic_parity"
    elif constraint_type == "Equal Opportunity":
        constraint = "equal_opportunity"
    elif constraint_type == "Equalized Odds":
        constraint = "equalized_odds"
    else:
        raise ValueError(f"Unsupported constraint for postprocessing: {constraint_type}")
        
    y_numeric = y_train.astype(int)
    s_numeric = sens_train.astype(int)
    
    mitigator = ThresholdOptimizer(
        estimator=fitted_model,
        constraints=constraint,
        predict_method="predict_proba" if hasattr(fitted_model, "predict_proba") else "auto",
        prefit=True
    )
    
    mitigator.fit(X_train, y_numeric, sensitive_features=s_numeric)
    return mitigator
