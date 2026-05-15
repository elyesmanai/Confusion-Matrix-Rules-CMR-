"""
Training script for BIG15 dataset with CMR rule extraction
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
import joblib

# Add parent directory to path to import cmr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmr import make_cmc, get_exclusive_rules, get_minmax_rules, compute_rule_coverage


def main():
    """Train model and extract CMR rules for BIG15 dataset"""
    
    print("=" * 60)
    print("Training CMR on BIG15 Dataset")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading BIG15 dataset...")
    X_train = pd.read_csv('../Datasets/BIG15/X_train.csv')
    X_test = pd.read_csv('../Datasets/BIG15/X_test.csv')
    y_train = pd.read_csv('../Datasets/BIG15/y_train.csv').values.ravel()
    y_test = pd.read_csv('../Datasets/BIG15/y_test.csv').values.ravel()
    print(f"   Train size: {X_train.shape}")
    print(f"   Test size: {X_test.shape}")
    
    # Train XGBoost model
    print("\n2. Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    print(f"   Train F1-score: {train_f1:.4f}")
    print(f"   Test F1-score: {test_f1:.4f}")
    
    # Save model
    os.makedirs('../Models', exist_ok=True)
    model_path = '../Models/BIG15.pkl'
    joblib.dump(model, model_path)
    print(f"   Model saved to {model_path}")
    
    # Create Confusion Matrix Categories (CMC)
    print("\n3. Creating Confusion Matrix Categories...")
    train_cmc = make_cmc(X_train, y_train, y_pred_train)
    print(f"   CMC Distribution:")
    print(train_cmc['CMC'].value_counts())
    
    # Save CMC
    os.makedirs('../Datasets/BIG15', exist_ok=True)
    train_cmc.to_csv('../Datasets/BIG15/train_cmc.csv', index=False)
    print(f"   CMC saved to ../Datasets/BIG15/train_cmc.csv")
    
    # Extract exclusive rules
    print("\n4. Extracting exclusive rules...")
    analysis_features = ['label', 'predicted', 'CMC']
    exclusive_rules = get_exclusive_rules(train_cmc, analysis_features)
    print(f"   Number of exclusive rules: {len(exclusive_rules)}")
    
    # Compute coverage
    print("\n5. Computing rule coverage...")
    exclusive_rules_sorted, unique_ids = compute_rule_coverage(exclusive_rules, len(X_train))
    
    # Save rules
    os.makedirs('../Rules', exist_ok=True)
    exclusive_rules_sorted.to_csv('../Rules/BIG15_exclusive_rules.csv', index=False)
    print(f"   Rules saved to ../Rules/BIG15_exclusive_rules.csv")
    
    # Extract min-max rules (optional)
    print("\n6. Extracting min-max rules...")
    minmax_rules = get_minmax_rules(train_cmc, analysis_features)
    print(f"   Number of min-max rules: {len(minmax_rules)}")
    if len(minmax_rules) > 0:
        minmax_rules.to_csv('../Rules/BIG15_minmax_rules.csv', index=False)
        print(f"   Min-max rules saved to ../Rules/BIG15_minmax_rules.csv")
    
    print("\n" + "=" * 60)
    print("BIG15 Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
