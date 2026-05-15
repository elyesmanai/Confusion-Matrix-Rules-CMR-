"""
Evaluation script for DoH20 dataset - Apply CMR rules and compute metrics
"""

import os
import sys
import pandas as pd
import joblib

# Add parent directory to path to import cmr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmr import apply_rules


def main():
    """Evaluate CMR rules on DoH20 test set"""
    
    print("=" * 60)
    print("Evaluating CMR on DoH20 (DOHBRW-20) Dataset")
    print("=" * 60)
    
    # Load test data
    print("\n1. Loading test data and rules...")
    X_test = pd.read_csv('../Datasets/DoH20/X_test.csv')
    y_test = pd.read_csv('../Datasets/DoH20/y_test.csv').values.ravel()
    X_test['label'] = y_test
    
    # Load exclusive rules
    exclusive_rules = pd.read_csv('../Rules/DOH20_exclusive_rules.csv')
    print(f"   Number of rules: {len(exclusive_rules)}")
    print(f"   Test set size: {len(X_test)}")
    
    # Load model
    model = joblib.load('../Models/DoH20.pkl')
    print(f"   Model loaded from ../Models/DoH20.pkl")
    
    # Evaluate with majority class
    print("\n2. Evaluating with majority class for uncovered instances...")
    results_majority = apply_rules(X_test, exclusive_rules, model=None, use_majority=True)
    
    # Evaluate with model
    print("\n3. Evaluating with model predictions for uncovered instances...")
    results_model = apply_rules(X_test, exclusive_rules, model=model, use_majority=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - DoH20 Results")
    print("=" * 60)
    print(f"Coverage: {results_model['coverage']:.2f}%")
    print(f"Conflict Rate: {results_model['conflict_rate']:.2f}%")
    print(f"F1-score (covered only): {results_model['f1_coverage']:.4f}")
    print(f"F1-score (majority): {results_majority['f1_full']:.4f}")
    print(f"F1-score (hybrid/model): {results_model['f1_full']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
