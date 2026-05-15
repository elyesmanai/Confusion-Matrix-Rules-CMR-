"""
Evaluation script for UNSW-NB15 dataset - Apply CMR rules and compute metrics
"""

import os
import sys
import pandas as pd
import joblib

# Add parent directory to path to import cmr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmr import apply_rules


def main():
    """Evaluate CMR rules on UNSW-NB15 test set"""
    
    print("=" * 60)
    print("Evaluating CMR on UNSW-NB15 Dataset")
    print("=" * 60)
    
    # Load test data
    print("\n1. Loading test data and rules...")
    test = pd.read_csv('../Datasets/UNSW-NB15/processed/le_test.csv')
    X_test = test.drop(['id', 'attack_cat'], axis=1)
    
    # Load exclusive rules
    exclusive_rules = pd.read_csv('../Rules/UNSWNB15_exclusive_rules.csv')
    print(f"   Number of rules: {len(exclusive_rules)}")
    print(f"   Test set size: {len(X_test)}")
    
    # Load model
    model = joblib.load('../Models/UNSW-NB15.pkl')
    print(f"   Model loaded from ../Models/UNSW-NB15.pkl")
    
    # Evaluate with majority class
    print("\n2. Evaluating with majority class for uncovered instances...")
    results_majority = apply_rules(X_test, exclusive_rules, model=None, use_majority=True)
    
    # Evaluate with model
    print("\n3. Evaluating with model predictions for uncovered instances...")
    results_model = apply_rules(X_test, exclusive_rules, model=model, use_majority=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - UNSW-NB15 Results")
    print("=" * 60)
    print(f"Coverage: {results_model['coverage']:.2f}%")
    print(f"Conflict Rate: {results_model['conflict_rate']:.2f}%")
    print(f"F1-score (covered only): {results_model['f1_coverage']:.4f}")
    print(f"F1-score (majority): {results_majority['f1_full']:.4f}")
    print(f"F1-score (hybrid/model): {results_model['f1_full']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
