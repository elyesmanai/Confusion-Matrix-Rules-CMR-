"""
Metrics for evaluating CMR rules - coverage, conflict rate, F1 scores
"""

import numpy as np
import pandas as pd


def compute_rule_coverage(rules_df, len_rows):
    """
    Compute coverage statistics for extracted rules.
    
    Args:
        rules_df: DataFrame containing rule-based results with 'support' and 'ids' columns
        len_rows: Total number of rows in the dataset
        
    Returns:
        Tuple of (sorted DataFrame, unique IDs set)
    """
    # Compute total support
    total_support = rules_df['support'].sum()
    support_pct = round(total_support / len_rows * 100, 2)
    
    # Remove duplicate IDs
    unique_ids = set(item for sublist in rules_df['ids'].values for item in sublist)
    unique_ids_pct = round(len(unique_ids) / len_rows * 100, 2)
    
    # Print results
    print(f"Total Support: {support_pct}% ({total_support})")
    print("After removing duplicates:")
    print(f"Unique Coverage: {unique_ids_pct}% ({len(unique_ids)})")
    
    # Return sorted DataFrame
    return rules_df.sort_values(by='pct', ascending=False), unique_ids


def compute_rule_overlap(uids_minmax, uids_ex, len_rows):
    """
    Compute overlap and unique coverage between two rule sets.
    
    Args:
        uids_minmax: Set of unique IDs from min-max rules
        uids_ex: Set of unique IDs from exclusive rules
        len_rows: Total number of rows in the dataset
        
    Returns:
        Dictionary with overlap statistics
    """
    # Compute intersection (common IDs in both rule sets)
    intersection_ids = uids_minmax.intersection(uids_ex)
    intersection_count = len(intersection_ids)
    intersection_pct = round(intersection_count / len_rows * 100, 2)
    
    # Compute total unique IDs without double counting overlap
    total_unique_ids = len(uids_minmax) + len(uids_ex) - intersection_count
    total_unique_pct = round(total_unique_ids / len_rows * 100, 2)
    
    # Print results
    print("Overlap")
    print(f"Count: {intersection_count}")
    print(f"Percentage: {intersection_pct}%\n")
    
    print("Without Overlap")
    print(f"Total Unique Count: {total_unique_ids}")
    print(f"Percentage: {total_unique_pct}%")
    
    return {
        "overlap_count": intersection_count,
        "overlap_pct": intersection_pct,
        "unique_count": total_unique_ids,
        "unique_pct": total_unique_pct
    }


def compute_conflict_rate(conflict_mask, total_instances):
    """
    Compute the conflict rate - percentage of instances with conflicting rules.
    
    Args:
        conflict_mask: Boolean array indicating conflicts
        total_instances: Total number of instances
        
    Returns:
        Float representing conflict rate percentage
    """
    conflicts = sum(conflict_mask)
    conflict_rate = round(conflicts / total_instances * 100, 2)
    return conflict_rate
