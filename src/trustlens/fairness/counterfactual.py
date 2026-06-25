from __future__ import annotations

import pandas as pd
import numpy as np

class CounterfactualEvaluator:
    def __init__(self):
        pass

    def evaluate_candidate(
        self,
        model,
        loader,
        df_raw: pd.DataFrame,
        candidate_dict: dict,
        protected_attribute: str,
        feature_names: list[str]
    ) -> list[dict]:
        """
        Evaluate a single candidate against their counterfactual counterparts.
        Swaps the protected_attribute and records model output changes.
        """
        # Determine possible values for the protected attribute
        priv_dict = loader.protected_attributes[protected_attribute]
        priv_val = priv_dict["privileged"]
        unpriv_val = priv_dict["unprivileged"]
        
        orig_val = candidate_dict.get(protected_attribute, "")
        
        # Find potential swap values
        if protected_attribute.lower() in ("sex", "gender"):
            swap_val = "Female" if orig_val == "Male" else "Male"
            swap_values = [swap_val]
        elif protected_attribute == "race":
            swap_values = ["Black", "White", "Asian-Pac-Islander"]
            swap_values = [v for v in swap_values if v != orig_val]
        elif protected_attribute == "MaritalStatus":
            swap_values = ["Single", "Married", "Divorced"]
            swap_values = [v for v in swap_values if v != orig_val]
        elif protected_attribute == "age_group":
            swap_val = "Young" if orig_val == "Older" else "Older"
            swap_values = [swap_val]
        else:
            swap_values = [unpriv_val if orig_val == priv_val else priv_val]
            
        # Create versions
        variants = [{"label": "Original", "data": candidate_dict.copy()}]
        for val in swap_values:
            variant_dict = candidate_dict.copy()
            variant_dict[protected_attribute] = val
            
            # If swapping age group, also update the numerical age to be realistic
            if protected_attribute == "age_group":
                age_col = next((c for c in candidate_dict if c.lower() == "age"), None)
                if age_col:
                    if val == "Young":
                        variant_dict[age_col] = 28
                    else:
                        variant_dict[age_col] = 50
                        
            variants.append({
                "label": f"Counterfactual ({val})",
                "data": variant_dict
            })
            
        # Preprocess all variants deterministically using df_raw statistics
        results = []
        for var in variants:
            cand_row = var["data"]
            cand_processed = {}
            
            # 1. Scale numericals
            for col in loader.numerical_features:
                if col in cand_row:
                    train_col = df_raw[col]
                    mean = train_col.mean()
                    std = train_col.std() + 1e-9
                    cand_processed[col] = (float(cand_row[col]) - mean) / std
                else:
                    cand_processed[col] = 0.0
                    
            # 2. One-hot encode categoricals
            for col in loader.categorical_features:
                if col in cand_row:
                    cats = sorted(df_raw[col].dropna().unique())
                    cand_val = cand_row[col]
                    for cat in cats:
                        dummy_name = f"{col}_{cat}"
                        cand_processed[dummy_name] = 1.0 if cand_val == cat else 0.0
                        
            # Build 1-row DataFrame aligned with training feature names
            cand_df = pd.DataFrame(index=[0])
            for feat in feature_names:
                cand_df[feat] = cand_processed.get(feat, 0.0)
                
            # Run model prediction
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(cand_df)[0, 1])
            else:
                prob = float(model.predict(cand_df)[0])
                
            pred = int(prob >= 0.5)
            
            results.append({
                "label": var["label"],
                "data": cand_row,
                "prediction": pred,
                "probability": prob
            })
            
        return results
