"""
Run all CMR experiments - train models and evaluate on all datasets
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment(dataset_name, train_script, eval_script):
    """Run training and evaluation for a single dataset"""
    print("\n" + "=" * 70)
    print(f"RUNNING {dataset_name} EXPERIMENT")
    print("=" * 70)
    
    # Import and run training
    print(f"\n### TRAINING {dataset_name} ###")
    try:
        if train_script == 'train_kdd99':
            from train_kdd99 import main as train_main
        elif train_script == 'train_unsw_nb15':
            from train_unsw_nb15 import main as train_main
        elif train_script == 'train_doh20':
            from train_doh20 import main as train_main
        elif train_script == 'train_big15':
            from train_big15 import main as train_main
        
        train_main()
        print(f"\n✓ {dataset_name} training completed successfully")
    except Exception as e:
        print(f"\n✗ {dataset_name} training failed: {str(e)}")
        return False
    
    # Import and run evaluation
    print(f"\n### EVALUATING {dataset_name} ###")
    try:
        if eval_script == 'evaluate_kdd99':
            from evaluate_kdd99 import main as eval_main
        elif eval_script == 'evaluate_unsw_nb15':
            from evaluate_unsw_nb15 import main as eval_main
        elif eval_script == 'evaluate_doh20':
            from evaluate_doh20 import main as eval_main
        elif eval_script == 'evaluate_big15':
            from evaluate_big15 import main as eval_main
        
        eval_main()
        print(f"\n✓ {dataset_name} evaluation completed successfully")
    except Exception as e:
        print(f"\n✗ {dataset_name} evaluation failed: {str(e)}")
        return False
    
    return True


def main():
    """Run all experiments"""
    print("=" * 70)
    print("CMR - RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    print("\nThis will train models and extract CMR rules for all datasets:")
    print("  1. KDD99")
    print("  2. UNSW-NB15")
    print("  3. DoH20 (DOHBRW-20)")
    print("  4. BIG15")
    print("\nNote: Make sure all datasets are available in the ../Datasets/ directory")
    print("=" * 70)
    
    experiments = [
        ("KDD99", "train_kdd99", "evaluate_kdd99"),
        ("UNSW-NB15", "train_unsw_nb15", "evaluate_unsw_nb15"),
        ("DoH20", "train_doh20", "evaluate_doh20"),
        ("BIG15", "train_big15", "evaluate_big15"),
    ]
    
    results = {}
    
    for dataset_name, train_script, eval_script in experiments:
        success = run_experiment(dataset_name, train_script, eval_script)
        results[dataset_name] = success
    
    # Summary
    print("\n\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for dataset_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{dataset_name:20s} {status}")
    print("=" * 70)
    
    # Overall status
    all_success = all(results.values())
    if all_success:
        print("\n🎉 All experiments completed successfully!")
    else:
        print("\n⚠️  Some experiments failed. Check the logs above for details.")
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
