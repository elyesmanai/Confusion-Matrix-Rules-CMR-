# Quick Start Guide - CMR

This guide will help you get started with the CMR (Confusion Matrix Rules) implementation.

## Prerequisites

- Python 3.7 or higher
- Datasets organized in the `Datasets/` directory

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Option 1: Run Individual Experiments

Train and evaluate on a specific dataset:

```bash
# Train model and extract rules
python scripts/train_kdd99.py

# Evaluate on test set
python scripts/evaluate_kdd99.py
```

Available scripts:
- `train_kdd99.py` / `evaluate_kdd99.py` - KDD99 dataset
- `train_unsw_nb15.py` / `evaluate_unsw_nb15.py` - UNSW-NB15 dataset
- `train_doh20.py` / `evaluate_doh20.py` - DoH20 (DOHBRW-20) dataset
- `train_big15.py` / `evaluate_big15.py` - BIG15 dataset

### Option 2: Run All Experiments

Run training and evaluation for all datasets at once:

```bash
python scripts/run_all_experiments.py
```

## Understanding the Output

### Training Output

When you run a training script, you'll see:

1. **Model Training**: XGBoost model training with F1-scores
2. **CMC Creation**: Distribution of confusion matrix categories
3. **Rule Extraction**: Number of exclusive and min-max rules extracted
4. **Coverage**: Percentage of training data covered by rules

### Evaluation Output

When you run an evaluation script, you'll see:

1. **Coverage**: Percentage of test instances covered by rules
2. **Conflict Rate**: Percentage of instances with conflicting rules
3. **F1-score (coverage)**: Performance on covered instances only
4. **F1-score (majority)**: Performance with majority class for uncovered instances
5. **F1-score (hybrid)**: Performance with model predictions for uncovered instances

## Output Files

After running experiments, you'll find:

```
CMR/
├── Models/
│   ├── KDD99.pkl              # Trained XGBoost model
│   ├── UNSW-NB15.pkl
│   ├── DoH20.pkl
│   └── BIG15.pkl
├── Rules/
│   ├── KDD99_exclusive_rules.csv    # Extracted exclusive rules
│   ├── KDD99_minmax_rules.csv       # Extracted min-max rules
│   └── ...
└── Datasets/
    └── [dataset]/
        └── train_cmc.csv            # Confusion Matrix Categories
```

## Using CMR in Your Code

```python
from cmr import make_cmc, get_exclusive_rules, apply_rules
import pandas as pd
import joblib

# 1. Train your model
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)

# 2. Create CMC
train_cmc = make_cmc(X_train, y_train, y_pred_train)

# 3. Extract rules
analysis_features = ['label', 'predicted', 'CMC']
rules = get_exclusive_rules(train_cmc, analysis_features)

# 4. Apply rules to test data
X_test['label'] = y_test
results = apply_rules(X_test, rules, model=model)

print(f"Coverage: {results['coverage']:.2f}%")
print(f"F1-score: {results['f1_full']:.4f}")
```

## Troubleshooting

### Dataset Not Found

Make sure your datasets are organized as:
```
Datasets/
├── KDD99/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
└── ...
```

### ImportError

Make sure you're running scripts from the repository root or the scripts directory:
```bash
cd CMR/
python scripts/train_kdd99.py
```

Or from the scripts directory:
```bash
cd CMR/scripts/
python train_kdd99.py
```

## Next Steps

- Check the full README.md for detailed documentation
- Examine the extracted rules in the `Rules/` directory
- Modify hyperparameters in the training scripts
- Implement custom rule extraction strategies in `cmr/core.py`

## Support

For issues or questions, please open an issue on GitHub.
