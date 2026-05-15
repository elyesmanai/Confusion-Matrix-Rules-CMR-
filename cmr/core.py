"""
Core CMR functions for rule extraction from confusion matrix categories
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


def make_cmc(df, y_train, y_pred_train):
    """
    Create Confusion Matrix Categories (CMC) from predictions and true labels.
    
    Args:
        df: DataFrame with features
        y_train: True labels
        y_pred_train: Predicted labels
        
    Returns:
        DataFrame with features and CMC categories (TP, TN, FP, FN)
    """
    train = df.copy()
    
    # Convert target variables to NumPy arrays to avoid broadcasting issues
    y_train = np.array(y_train).flatten()
    y_pred_train = np.array(y_pred_train).flatten()

    # Ensure they have the same length as df
    assert len(y_train) == len(df), "y_train length does not match dataframe rows"
    assert len(y_pred_train) == len(df), "y_pred_train length does not match dataframe rows"

    # Assign labels to the DataFrame
    train['label'] = y_train
    train['predicted'] = y_pred_train

    # Compute CMC categories efficiently using NumPy
    train['CMC'] = np.where(y_pred_train == y_train, 
                            np.where(y_pred_train == 1, 'TP', 'TN'), 
                            np.where(y_pred_train == 1, 'FP', 'FN'))

    return train


def get_exclusive_rules(cmc, analysis_features):
    """
    Extract exclusive rules where a feature value uniquely maps to one CMC category.
    
    Args:
        cmc: DataFrame containing feature values and CMC labels
        analysis_features: List of features to exclude (e.g., 'label', 'predicted', 'CMC')
        
    Returns:
        DataFrame containing exclusive rules with columns:
        [feature, is, value, predict, support, pct, ids]
    """
    len_rows = cmc.shape[0]
    exclusivity_rules_list = []

    # Iterate over features while dropping analysis features
    for feature in tqdm(cmc.columns.drop(analysis_features), desc="Extracting exclusive rules"):
        # Get unique CMC groups for each feature value
        value_to_cmc = cmc.groupby(feature)['CMC'].nunique()

        # Filter values that belong to only one CMC group
        exclusive_values = value_to_cmc[value_to_cmc == 1].index

        # If no exclusive values, skip
        if exclusive_values.empty:
            continue

        # Get mapping of each exclusive value to its corresponding CMC category
        cmc_map = cmc.groupby(feature)['CMC'].first()

        # Get IDs for all exclusive values in one step
        id_map = cmc[cmc[feature].isin(exclusive_values)].groupby(feature).apply(lambda x: x.index.tolist())

        # Construct rules efficiently
        for val in exclusive_values:
            group = cmc_map[val]
            ids = id_map[val]

            if group in ['TP', 'TN']:
                pred = 1 if group == 'TP' else 0
                support = len(ids)
                pct = round(support / len_rows * 100, 2)

                if support > 2:  # Filter out low-support rules
                    exclusivity_rules_list.append([feature, '==', val, pred, support, pct, ids])

    # Convert list to DataFrame at the end for efficiency
    return pd.DataFrame(exclusivity_rules_list, columns=['feature', 'is', 'value', 'predict', 'support', 'pct', 'ids'])


def get_support(cmc, feature, threshold):
    """
    Helper function to compute support for a threshold-based rule.
    
    Args:
        cmc: DataFrame with CMC categories
        feature: Feature name
        threshold: Threshold value
        
    Returns:
        Tuple of (support, percentage, ids) or None if rule is invalid
    """
    len_rows = cmc.shape[0]
    results = cmc[cmc[feature] > threshold + 0.0001]['label'].value_counts()
    
    if results.shape[0] != 1:
        return None
    
    support = results.values[0]
    pct = round(support / len_rows * 100, 2)
    return support, pct, cmc[cmc[feature] > threshold + 0.0001].index.tolist()


def get_minmax_rules(cmc, analysis_features):
    """
    Extract min-max threshold rules from continuous features.
    
    Args:
        cmc: DataFrame containing feature values and CMC labels
        analysis_features: List of features to exclude
        
    Returns:
        DataFrame containing min-max rules with columns:
        [feature, is, value, predict, support, pct, ids]
    """
    minmax_rules = pd.DataFrame(columns=['feature', 'is', 'value', 'predict', 'support', 'pct', 'ids'])
    
    for feature in tqdm(cmc.columns.drop(analysis_features), desc="Extracting min-max rules"):
        try:
            # Get the max values per CMC category safely
            max_values = cmc.groupby('CMC')[feature].max().to_dict()

            # Extract values safely, defaulting to -inf (or another placeholder)
            mFN = max_values.get('FN', float('-inf'))
            mFP = max_values.get('FP', float('-inf'))
            mTN = max_values.get('TN', float('-inf'))
            mTP = max_values.get('TP', float('-inf'))
            
            # Compute min/max with safety checks
            min_FT, max_FT = min(mFN, mTN), max(mFP, mTP)
            min_PT, max_PT = min(mFP, mTP), max(mFN, mTN)

        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
            continue

        threshold, label = None, None
        
        if min_FT > max_FT:
            threshold, label = min_FT, 0
        elif min_PT > max_PT:
            threshold, label = min_PT, 1
        elif min_PT == max_PT or min_FT == max_FT:
            continue
        else:
            continue
        
        result = get_support(cmc, feature, threshold)
        if result:
            support, pct, ids = result
            minmax_rules = pd.concat([minmax_rules, pd.DataFrame({
                'feature': [feature],
                'is': ['>'],
                'value': [threshold],
                'predict': [label],
                'support': [support],
                'pct': [pct],
                'ids': [ids]
            })], ignore_index=True)
    
    return minmax_rules
