"""
Inference module for applying CMR rules to test data
"""

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


def apply_rules(X_test, exclusive_rules, model=None, use_majority=False):
    """
    Apply exclusive rules to test data and compute metrics.
    
    Args:
        X_test: Test DataFrame with 'label' column
        exclusive_rules: DataFrame with extracted rules
        model: Optional model for predicting uncovered instances
        use_majority: If True and model is None, use majority class for uncovered instances
        
    Returns:
        Dictionary with predictions and metrics including:
        - y_pred: Predictions for all instances
        - coverage: Percentage of instances covered by rules
        - f1_coverage: F1-score on covered instances only
        - f1_full: F1-score on all instances (with model or majority class)
        - conflict_rate: Percentage of instances with conflicting rules
        - skipped_conflicts: Number of instances skipped due to conflicts
        - skipped_no_rule: Number of instances with no matching rule
    """
    # Initialize predictions with 0, then change skipped ones to None
    y_pred = np.full(len(X_test), 0, dtype=object)  # Start with 0
    conflict_mask = np.zeros(len(X_test), dtype=bool)  # Tracks conflicts
    no_rule_mask = np.ones(len(X_test), dtype=bool)  # Tracks uncovered cases (start with all True)

    for _, rule in tqdm(exclusive_rules.iterrows(), total=len(exclusive_rules), desc="Applying rules"):
        feature, value, label, pct = rule['feature'], rule['value'], rule['predict'], rule['pct']

        # Apply rule
        mask = X_test[feature] == value
        matching_indices = np.where(mask)[0]

        for idx in matching_indices:
            no_rule_mask[idx] = False  # Rule covered this instance

            if y_pred[idx] == 0:  # If still default, assign the first rule
                y_pred[idx] = (label, pct)  
            else:
                prev_label, prev_pct = y_pred[idx] if isinstance(y_pred[idx], tuple) else (y_pred[idx], 0)

                if prev_label == label:
                    y_pred[idx] = (prev_label, prev_pct + pct)  # Sum percentages for same prediction
                else:
                    conflict_mask[idx] = True  # Mark as conflict
                    # Compare summed percentages before deciding
                    if pct > prev_pct:
                        y_pred[idx] = (label, pct)  # Assign label with higher total pct

    # Extract final predictions, replacing default 0 with None where necessary
    y_pred = [p[0] if isinstance(p, tuple) else (None if p == 0 else p) for p in y_pred]

    # Track reasons for skipping
    skipped_mask = [p is None for p in y_pred]
    skipped_total = sum(skipped_mask)

    # Correct count of skipped due to conflicts and no rule
    skipped_conflicts = sum(conflict_mask & skipped_mask)
    skipped_no_rule = sum(no_rule_mask & skipped_mask)

    # Convert to NumPy array for filtering
    y_test = X_test['label'].values
    valid_mask = ~np.array(skipped_mask)
    y_pred_valid = np.array(y_pred)[valid_mask].astype(np.int64)  # Convert to numeric
    y_test_valid = np.array(y_test)[valid_mask].astype(np.int64)

    # Compute F1-score only for valid predictions
    f1_valid = None
    coverage = len(y_pred_valid) / len(y_test) * 100
    if len(y_pred_valid) > 0:
        f1_valid = f1_score(y_test_valid, y_pred_valid, average='weighted')
        print(f"Test set coverage: {coverage:.2f}%")
        print(f"Test coverage F1-score: {f1_valid:.4f}")
    else:
        print("No valid predictions available for F1-score calculation.")

    # Compute F1-score using ALL instances (assign default label to skipped cases)
    if model:
        X_test_features = X_test.drop('label', axis=1).values  
        missing_idx = np.where(np.array(y_pred) == None)[0]  

        if len(missing_idx) > 0:  
            missing_preds = model.predict(X_test_features[missing_idx])  
            y_pred_full = np.array(y_pred)  
            y_pred_full[missing_idx] = missing_preds  
            y_pred_full = y_pred_full.astype(np.int64)  
        else:
            y_pred_full = np.array(y_pred, dtype=np.int64)  
    elif use_majority:
        majority_label = np.bincount(y_test).argmax()
        y_pred_full = np.array([majority_label if p is None else p for p in y_pred], dtype=np.int64)
    else:
        # Default to 0 for uncovered instances
        y_pred_full = np.array([0 if p is None else p for p in y_pred], dtype=np.int64)

    f1_full = f1_score(y_test, y_pred_full, average='weighted')

    # Print statistics
    total_instances = len(X_test)
    conflict_rate = round(skipped_conflicts / total_instances * 100, 2)
    
    print('--------------------------------------')
    print(f"Total Skipped: {skipped_total} ({round(skipped_total / total_instances * 100, 2)}%)")
    print(f" - Due to Conflicts: {skipped_conflicts} ({conflict_rate}%)")
    print(f" - Due to No Rule Coverage: {skipped_no_rule} ({round(skipped_no_rule / total_instances * 100, 2)}%)")
    print('--------------------------------------')
    
    if model:
        print('Using model to predict non-covered cases.')
    elif use_majority:
        print('Using majority class to predict non-covered cases.')
    else:
        print('Using default class (0) to predict non-covered cases.')
        
    print(f"F1-score (full dataset): {f1_full:.4f}")
    
    return {
        'y_pred': y_pred_full,
        'coverage': coverage,
        'f1_coverage': f1_valid,
        'f1_full': f1_full,
        'conflict_rate': conflict_rate,
        'skipped_conflicts': skipped_conflicts,
        'skipped_no_rule': skipped_no_rule,
        'total_skipped': skipped_total
    }
