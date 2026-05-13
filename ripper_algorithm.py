import re
import time
import numpy as np
import pandas as pd
import wittgenstein as lw

from sklearn.metrics import accuracy_score, f1_score

def train_ripper_te(x_train, y_train, prune_size=0.33, k=2, max_rules=50):
    """
    TE RIPPER : apprentissage du modèle.
    """
    train_data = x_train.copy()
    train_data["label"] = pd.Series(y_train).values

    model = lw.RIPPER(
        prune_size=prune_size,
        k=k,
        max_rules=max_rules
    )

    model.fit(
        train_data,
        class_feat="label",
        pos_class=1,
        neg_class=0
    )

    return model


def generate_ripper_rules_tg(model):
    """
    TG RIPPER : récupération des règles.
    """
    return getattr(model, "ruleset_", None)


def train_ripper(x_train, y_train, x_test, y_test, prune_size=0.33, k=2, max_rules=50):
    """
    Entraîne l'algorithme RIPPER.
    """
    train_data = x_train.copy()
    train_data["label"] = pd.Series(y_train).values

    model = lw.RIPPER(prune_size=prune_size, k=k, max_rules=max_rules)

    start_time = time.time()
    model.fit(train_data, class_feat="label", pos_class=1, neg_class=0)
    train_time = time.time() - start_time

    metrics = {
        "train_time": train_time
    }

    rules = getattr(model, "ruleset_", None)

    return model, metrics, rules


def compute_ripper_complexity(rules):
    """
    Calcule la complexité moyenne des règles.
    """
    if rules is None:
        return 0.0, 0, 0

    # Initialisation des compteurs
    nb_regles = 0
    total_conditions = 0
    # CAS1 : les regles peuvent être sous différentes formes : une chaîne de caractères(string).
    if isinstance(rules, str):
        text = rules.strip()
        if text:
            raw_rules = [r.strip() for r in re.split(r"\s+V\s+|\s+OR\s+|\|", text) if r.strip()]

            # Pour chaque règle, on compte le nombre de conditions
            for rule in raw_rules:
                rule = rule.strip("[] ").strip() # Nettoyage des crochets et espaces
                if not rule:
                    continue

                nb_regles += 1 # Incrémentation du nombre de règles
                conditions = [c.strip() for c in rule.split("^") if c.strip()] # Séparation des conditions par le symbole ^ et nettoyage
                total_conditions += len(conditions) # Incrémentation du nombre total de conditions
    # CAS2 : les règles peuvent être sous forme de liste, tuple ou set de chaînes de caractères.
    elif isinstance(rules, (list, tuple, set)):
        for rule in rules:
            rule_text = str(rule).strip()
            if not rule_text:
                continue

            nb_regles += 1
            conditions = [c.strip() for c in re.split(r"\^", rule_text) if c.strip()]
            total_conditions += len(conditions)
    # CAS3 : les règles peuvent être sous forme de dataframe pandas, où chaque ligne représente une règle et les conditions sont séparées par des colonnes.
    elif isinstance(rules, pd.DataFrame):
        nb_regles = len(rules)
        total_conditions = len(rules)
    # CAS4 : autres formats possibles, on essaie de les traiter comme une chaîne de caractères brute, en comptant les règles et les conditions à partir du texte.
    else:
        try:
            rule_text = str(rules).strip()
            if rule_text:
                raw_rules = [r.strip() for r in re.split(r"\s+V\s+|\s+OR\s+|\|", rule_text) if r.strip()]
                for rule in raw_rules:
                    rule = rule.strip("[] ").strip()
                    if not rule:
                        continue
                    nb_regles += 1
                    conditions = [c.strip() for c in rule.split("^") if c.strip()]
                    total_conditions += len(conditions)
        except Exception:
            return 0.0, 0, 0 # Si le format est totalement inattendu, on retourne une complexité de 0

    if nb_regles == 0: # Pour éviter la division par zéro, si aucune règle n'est trouvée, on considère que la complexité est de 0
        return 0.0, 0, 0

    complexite_moyenne = total_conditions / nb_regles
    return float(complexite_moyenne), int(nb_regles), int(total_conditions)


def evaluate_ripper(model, X_train, y_train, X_test, y_test, elapsed=None, ram=None):
    """
    Évaluer RIPPER.
    """
    train_data = X_train.copy()
    train_data["label"] = pd.Series(y_train).values

    test_data = X_test.copy()
    test_data["label"] = pd.Series(y_test).values

    y_train_pred = model.predict(train_data)
    y_test_pred = model.predict(test_data)

    f1_train = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
    f1_test = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    exact_train = accuracy_score(y_train, y_train_pred)
    exact_test = accuracy_score(y_test, y_test_pred)

    rules = getattr(model, "ruleset_", None)
    complexity, nb_regles, total_conditions = compute_ripper_complexity(rules)

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