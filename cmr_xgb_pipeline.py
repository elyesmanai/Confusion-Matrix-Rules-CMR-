import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

from explainer import make_cmc, get_exclusive_rules


def _predict_with_rules(X, rules):
    """
    Applique les règles exclusives sur X.
    """
    y_pred = np.full(len(X), 0, dtype=object)
    conflict_mask = np.zeros(len(X), dtype=bool)
    no_rule_mask = np.ones(len(X), dtype=bool)

    for _, rule in rules.iterrows():
        feature = rule["feature"]
        value = rule["value"]
        label = rule["predict"]
        pct = rule["pct"]

        mask = X[feature] == value
        matching_indices = np.where(mask)[0]

        for idx in matching_indices:
            no_rule_mask[idx] = False

            if y_pred[idx] == 0:
                y_pred[idx] = (label, pct)
            else:
                prev_label, prev_pct = y_pred[idx] if isinstance(y_pred[idx], tuple) else (y_pred[idx], 0)

                if prev_label == label:
                    y_pred[idx] = (prev_label, prev_pct + pct)
                else:
                    conflict_mask[idx] = True
                    if pct > prev_pct:
                        y_pred[idx] = (label, pct)

    y_pred_clean = [p[0] if isinstance(p, tuple) else (None if p == 0 else p) for p in y_pred]

    skipped_mask = np.array([p is None for p in y_pred_clean])
    valid_mask = ~skipped_mask

    return {
        "y_pred_partial": np.array(y_pred_clean, dtype=object),
        "valid_mask": valid_mask,
        "skipped_mask": skipped_mask,
        "conflict_mask": conflict_mask,
        "no_rule_mask": no_rule_mask
    }


def _finalize_predictions(y_pred_partial, y_true, model=None, X_features=None):
    """
    Complète les cas non couverts.
    """
    y_true = np.array(y_true).astype(int)
    y_pred_full = np.array(y_pred_partial, dtype=object)

    missing_idx = np.where(pd.isna(y_pred_full))[0]

    if len(missing_idx) > 0:
        if model is not None and X_features is not None:
            missing_preds = model.predict(X_features.iloc[missing_idx])
            y_pred_full[missing_idx] = missing_preds
        else:
            majority_label = np.bincount(y_true).argmax()
            y_pred_full[missing_idx] = majority_label

    return y_pred_full.astype(int)


def train_cmr(X_train, y_train, X_test, y_test, support_min=3):
    """
    Entraine et extrait les règles CMR.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = pd.Series(y_train).astype(int).reset_index(drop=True)
    y_test = pd.Series(y_test).astype(int).reset_index(drop=True)

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols].reset_index(drop=True)
    X_test = X_test[common_cols].reset_index(drop=True)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    y_train_pred_model = model.predict(X_train)
    f1_train_model = f1_score(y_train, y_train_pred_model, average="weighted", zero_division=0)

    train_cmc = make_cmc(X_train.copy(), y_train, y_train_pred_model)

    rules = get_exclusive_rules(
        train_cmc,
        analysis_features=["label", "predicted", "CMC"]
    )

    rules = rules[rules["support"] >= support_min].reset_index(drop=True)

    return {
        "model": model,
        "rules": rules,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "F1_train": float(f1_train_model)
    }


def evaluate_cmr(payload, elapsed=None, ram=None):
    """
    Évaluer CMR.
    """
    model = payload["model"]
    rules = payload["rules"]
    X_train = payload["X_train"]
    X_test = payload["X_test"]
    y_train = payload["y_train"]
    y_test = payload["y_test"]
    f1_train_model = payload["F1_train"]

    if rules.empty:
        return {
            "F1_train": float(f1_train_model),
            "F1_test": 0.0,
            "Train_Coverage": 0.0,
            "Train_Exactitude": 0.0,
            "Test_Coverage": 0.0,
            "Test_Exactitude": 0.0,
            "Conflit": 0.0,
            "Non_couverts": 100.0,
            "F1_rmaj": 0.0,
            "F1_rr": 0.0,
            "Nb_règles": 0,
            "Nb_conditions": 0,
            "Complexité": 0.0,
            "Temps": round(elapsed, 2) if elapsed is not None else None,
            "RAM_MB": round(ram, 2) if ram is not None else None
        }

    train_pred_info = _predict_with_rules(X_train, rules)
    train_valid_mask = train_pred_info["valid_mask"]

    if train_valid_mask.sum() > 0:
        train_exactitude = accuracy_score(
            y_train.iloc[train_valid_mask],
            train_pred_info["y_pred_partial"][train_valid_mask].astype(int)
        )
    else:
        train_exactitude = 0.0

    train_coverage = round(train_valid_mask.mean() * 100, 2)

    test_pred_info = _predict_with_rules(X_test, rules)
    test_valid_mask = test_pred_info["valid_mask"]
    test_skipped_mask = test_pred_info["skipped_mask"]
    test_conflict_mask = test_pred_info["conflict_mask"]

    if test_valid_mask.sum() > 0:
        test_exactitude = accuracy_score(
            y_test.iloc[test_valid_mask],
            test_pred_info["y_pred_partial"][test_valid_mask].astype(int)
        )
    else:
        test_exactitude = 0.0

    test_coverage = round(test_valid_mask.mean() * 100, 2)
    conflit = round(test_conflict_mask.mean() * 100, 2)
    non_couverts = round(test_skipped_mask.mean() * 100, 2)

    y_pred_rmaj = _finalize_predictions(
        test_pred_info["y_pred_partial"],
        y_test,
        model=None,
        X_features=None
    )
    f1_rmaj = f1_score(y_test, y_pred_rmaj, average="weighted", zero_division=0)

    y_pred_rr = _finalize_predictions(
        test_pred_info["y_pred_partial"],
        y_test,
        model=model,
        X_features=X_test
    )
    f1_rr = f1_score(y_test, y_pred_rr, average="weighted", zero_division=0)

    nb_regles = len(rules)
    nb_conditions = len(rules)
    complexite = nb_conditions / nb_regles if nb_regles > 0 else 0.0

    return {
        "F1_train": float(f1_train_model),
        "F1_test": float(f1_rr),
        "Train_Coverage": float(train_coverage),
        "Train_Exactitude": float(train_exactitude),
        "Test_Coverage": float(test_coverage),
        "Test_Exactitude": float(test_exactitude),
        "Conflit": float(conflit),
        "Non_couverts": float(non_couverts),
        "F1_rmaj": float(f1_rmaj),
        "F1_rr": float(f1_rr),
        "Nb_règles": int(nb_regles),
        "Nb_conditions": int(nb_conditions),
        "Complexité": float(complexite),
        "Temps": round(elapsed, 2) if elapsed is not None else None,
        "RAM_MB": round(ram, 2) if ram is not None else None
    }


def predict_cmr_test_only(payload):
    """
    Lance uniquement la phase test hybride.
    """
    model = payload["model"]
    rules = payload["rules"]
    X_test = payload["X_test"]
    y_test = payload["y_test"]

    if rules.empty:
        return np.zeros(len(y_test), dtype=int)

    test_pred_info = _predict_with_rules(X_test, rules)

    y_pred_rr = _finalize_predictions(
        test_pred_info["y_pred_partial"],
        y_test,
        model=model,
        X_features=X_test
    )

    return y_pred_rr