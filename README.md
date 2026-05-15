# CMR - Confusion Matrix Rules

A rule-based explainable AI method for binary classification using Confusion Matrix Categories.

## Overview

CMR (Confusion Matrix Rules) is a novel approach that extracts interpretable rules from trained machine learning models by analyzing confusion matrix categories (True Positives, True Negatives, False Positives, False Negatives). This repository contains the implementation and evaluation code for the CMR method on four benchmark intrusion detection datasets.

## Features

- **Explainable AI**: Extract human-readable rules from black-box models
- **High Coverage**: Rules cover a significant portion of the test instances
- **Hybrid Predictions**: Combine rule-based and model-based predictions
- **Multiple Metrics**: F1-score (coverage, majority, hybrid), coverage rate, conflict rate

## Repository Structure

```
CMR/
├── cmr/                          # Core CMR package
│   ├── __init__.py              # Package initialization
│   ├── core.py                  # Core CMR functions (make_cmc, rule extraction)
│   ├── inference.py             # Rule application and inference
│   └── metrics.py               # Evaluation metrics
├── scripts/                      # Training and evaluation scripts
│   ├── train_kdd99.py           # KDD99 training script
│   ├── train_unsw_nb15.py       # UNSW-NB15 training script
│   ├── train_doh20.py           # DoH20 training script
│   ├── train_big15.py           # BIG15 training script
│   ├── evaluate_kdd99.py        # KDD99 evaluation script
│   ├── evaluate_unsw_nb15.py    # UNSW-NB15 evaluation script
│   ├── evaluate_doh20.py        # DoH20 evaluation script
│   └── evaluate_big15.py        # BIG15 evaluation script
├── Datasets/                     # Dataset directory (not included)
├── Models/                       # Trained models directory
├── Rules/                        # Extracted rules directory
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wiseresearch/Confusion-Matrix-Rules-CMR-.git
cd Confusion-Matrix-Rules-CMR-
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

This repository supports four intrusion detection datasets:

1. **KDD99**: NSL-KDD dataset for network intrusion detection
2. **UNSW-NB15**: A modern network intrusion detection dataset
3. **DoH20 (DOHBRW-20)**: DNS over HTTPS traffic classification
4. **BIG15**: Large-scale network traffic dataset

### Dataset Structure

Datasets should be organized as follows:

```
Datasets/
├── KDD99/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── UNSW-NB15/
│   └── processed/
│       ├── le_train.csv
│       └── le_test.csv
├── DoH20/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
└── BIG15/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    └── y_test.csv
```

**Note**: Datasets are not included in this repository due to size constraints. Please download them from their respective sources and organize them as shown above.

## Usage

### Training

To train a model and extract CMR rules for a dataset:

```bash
# KDD99
python scripts/train_kdd99.py

# UNSW-NB15
python scripts/train_unsw_nb15.py

# DoH20
python scripts/train_doh20.py

# BIG15
python scripts/train_big15.py
```

This will:
1. Train an XGBoost classifier on the training data
2. Create Confusion Matrix Categories (CMC)
3. Extract exclusive and min-max rules
4. Save the model, CMC, and rules

### Evaluation

To evaluate the extracted rules on test data:

```bash
# KDD99
python scripts/evaluate_kdd99.py

# UNSW-NB15
python scripts/evaluate_unsw_nb15.py

# DoH20
python scripts/evaluate_doh20.py

# BIG15
python scripts/evaluate_big15.py
```

This will:
1. Load the test data and extracted rules
2. Apply rules to test instances
3. Compute metrics (coverage, conflict rate, F1-scores)
4. Display results with majority class and hybrid (model) predictions

### Using the CMR Package

You can also use the CMR package directly in your Python code:

```python
from cmr import make_cmc, get_exclusive_rules, apply_rules
import pandas as pd
import joblib

# Load your data and model
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
model = joblib.load('model.pkl')

# Get predictions
y_pred_train = model.predict(X_train)

# Create Confusion Matrix Categories
train_cmc = make_cmc(X_train, y_train, y_pred_train)

# Extract rules
analysis_features = ['label', 'predicted', 'CMC']
exclusive_rules = get_exclusive_rules(train_cmc, analysis_features)

# Apply rules to test data
X_test = pd.read_csv('X_test.csv')
X_test['label'] = pd.read_csv('y_test.csv').values.ravel()
results = apply_rules(X_test, exclusive_rules, model=model)

print(f"Coverage: {results['coverage']:.2f}%")
print(f"F1-score: {results['f1_full']:.4f}")
```

## Metrics

CMR provides several evaluation metrics:

- **Coverage**: Percentage of test instances covered by at least one rule
- **Conflict Rate**: Percentage of instances with conflicting rule predictions
- **F1-score (Coverage)**: F1-score computed only on covered instances
- **F1-score (Majority)**: F1-score with majority class for uncovered instances
- **F1-score (Hybrid)**: F1-score with model predictions for uncovered instances

## Method Overview

### 1. Confusion Matrix Categories (CMC)

Given a trained model's predictions, we categorize each training instance into:
- **TP (True Positive)**: Correctly predicted as positive
- **TN (True Negative)**: Correctly predicted as negative
- **FP (False Positive)**: Incorrectly predicted as positive
- **FN (False Negative)**: Incorrectly predicted as negative

### 2. Rule Extraction

**Exclusive Rules**: For each feature value, if all instances with that value belong to the same CMC category (TP or TN), we create a rule:
- `feature == value → predict class`

**Min-Max Rules**: For continuous features, we identify threshold values that separate prediction classes.

### 3. Rule Application

When applying rules to test instances:
1. Find all matching rules for each instance
2. If rules agree, use the rule prediction
3. If rules conflict, use the rule with higher support
4. If no rules match, use the trained model or majority class

## License

[Specify License]

## Contact

For questions or issues, please open an issue on GitHub or contact [contact information].

## Acknowledgments

This work was conducted as part of research on explainable AI for intrusion detection systems.
