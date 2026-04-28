import re
import time
import numpy as np
import pandas as pd

from aix360.algorithms.rbm import FeatureBinarizer, BooleanRuleCG
from sklearn.metrics import accuracy_score, f1_score


def train_brcg(x_train, y_train, x_test, y_test, lambda0=0.001, lambda1=0.001):
    """
    Entraîne l'algorithme BRCG.
    """
    fb = FeatureBinarizer(negations=True)

    X_train_bin = fb.fit_transform(x_train)

    if isinstance(X_train_bin, pd.DataFrame):
        columns = list(X_train_bin.columns)
        X_train_bin = X_train_bin.fillna(0).astype(int)
    else:
        X_train_bin = np.nan_to_num(X_train_bin)
        columns = [f"f{i}" for i in range(X_train_bin.shape[1])]
        X_train_bin = pd.DataFrame(X_train_bin, columns=columns).astype(int)

    model = BooleanRuleCG(lambda0=lambda0, lambda1=lambda1, verbose=False)

    start_time = time.time()
    model.fit(X_train_bin, y_train) # entraînement brcg avec des donness binarisées
    train_time = time.time() - start_time

    metrics = {
        "train_time": train_time
    }

    try:
        rules = model.explain()
        if not isinstance(rules, dict) or len(rules.get("rules", [])) == 0:
            rules = None
    except Exception:
        rules = None

    return model, fb, columns, train_time, metrics, rules


def compute_rule_based_complexity(rules):
    """
    Calcule la complexité moyenne des règles.
    """
    if rules is None:
        return 0.0, 0, 0

    nb_regles = 0
    total_conditions = 0

    if isinstance(rules, dict):
        rule_list = rules.get("rules", [])

        for rule in rule_list:
            if not isinstance(rule, str):
                continue

            rule = rule.strip()
            if not rule:
                continue

            nb_regles += 1
            conditions = [c.strip() for c in re.split(r"\s+AND\s+", rule) if c.strip()]
            total_conditions += len(conditions)

    elif isinstance(rules, (list, tuple, set)):
        for rule in rules:
            rule = str(rule).strip()
            if not rule:
                continue

            nb_regles += 1
            conditions = [c.strip() for c in re.split(r"\s+AND\s+", rule) if c.strip()]
            total_conditions += len(conditions)

    elif isinstance(rules, str):
        text = rules.strip()
        if text:
            raw_rules = [r.strip() for r in re.split(r"\s+OR\s+|\|", text) if r.strip()]

            for rule in raw_rules:
                nb_regles += 1
                conditions = [c.strip() for c in re.split(r"\s+AND\s+", rule) if c.strip()]
                total_conditions += len(conditions)

    elif isinstance(rules, pd.DataFrame):
        nb_regles = len(rules)
        total_conditions = len(rules)

    if nb_regles == 0:
        return 0.0, 0, 0

    complexite_moyenne = total_conditions / nb_regles
    return float(complexite_moyenne), int(nb_regles), int(total_conditions)


def evaluate_brcg(model, fb, X_train, y_train, X_test, y_test, elapsed=None, ram=None, columns=None, rules=None):
    """
    Évaluer BRCG.
    """
    X_train_bin = fb.transform(X_train)
    X_test_bin = fb.transform(X_test)

    if isinstance(X_train_bin, pd.DataFrame):
        X_train_bin = X_train_bin.fillna(0).astype(int)
    else:
        X_train_bin = pd.DataFrame(np.nan_to_num(X_train_bin), columns=columns).astype(int)

    if isinstance(X_test_bin, pd.DataFrame):
        X_test_bin = X_test_bin.fillna(0).astype(int)
    else:
        X_test_bin = pd.DataFrame(np.nan_to_num(X_test_bin), columns=columns).astype(int)

    y_train_pred = model.predict(X_train_bin)
    y_test_pred = model.predict(X_test_bin)

    f1_train = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
    f1_test = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    exact_train = accuracy_score(y_train, y_train_pred)
    exact_test = accuracy_score(y_test, y_test_pred)

    try:
        if rules is None:
            rules = model.explain()
        complexity, nb_regles, total_conditions = compute_rule_based_complexity(rules)
    except Exception:
        complexity, nb_regles, total_conditions = 0.0, 0, 0

    return {
        "F1_train": float(f1_train),
        "F1_test": float(f1_test),
        "Train_Coverage": np.nan,
        "Train_Exactitude": float(exact_train),
        "Test_Coverage": np.nan,
        "Test_Exactitude": float(exact_test),
        "Conflit": np.nan,
        "Non_couverts": np.nan,
        "F1_rmaj": np.nan,
        "F1_rr": np.nan,
        "Complexité": float(complexity),
        "Nb_règles": int(nb_regles),
        "Nb_conditions": int(total_conditions),
        "Temps": round(elapsed, 2) if elapsed is not None else None,
        "RAM_MB": round(ram, 2) if ram is not None else None
    }