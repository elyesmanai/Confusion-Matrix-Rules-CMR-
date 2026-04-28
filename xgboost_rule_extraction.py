#!/usr/bin/env python
# coding: utf-8

# 1. Import des bibliothèches principales
import time
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


# 1.1 Chargement des bibliothèques XGBoost et métriques
import import_ipynb
from adultdata_preparation_evaluat import prepare_adult_dataset

x_train, x_test, y_train, y_test, mapping = prepare_adult_dataset()


# 2. Définition et entrainement de XGBoost
def train_xgboost(x_train, y_train, x_test, y_test):
    """
    Train an XGBoost model and return predictions and metrics.

    Parameters
    ----------
    x_train : DataFrame
    y_train : Series
    x_test : DataFrame
    y_test : Series

    Returns
    -------
    model : trained XGBoost model
    y_pred : predictions on test set
    metrics : dictionary with evaluation metrics
    """

    start_time = time.time()

    # modèle
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # entrainement
    model.fit(x_train, y_train)

    # prédictions
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    training_time = time.time() - start_time

    # métriques test
    metrics_test = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    # métriques train
    metrics_train = {
        "accuracy_train": accuracy_score(y_train, y_pred_train),
        "f1_train": f1_score(y_train, y_pred_train)
    }

    # fusion métriques
    metrics = {**metrics_test, **metrics_train}
    metrics["training_time"] = training_time

    return model, y_pred, metrics


# 2.1 Nettoyage des données - suppression de la colonne label
x_train = x_train.drop(columns=['label'], errors='ignore')
x_test  = x_test.drop(columns=['label'], errors='ignore')

# 2.2 Entrainement du modèle XGBoost
model, y_pred, metrics = train_xgboost(
    x_train, y_train, x_test, y_test
)


# 2.3 Rechargement du module explainer
import importlib
import explainer
importlib.reload(explainer)


# 2.4 Génération des règles exclusives à partir des données d'entrainement
# CMC est une table qui indique si chaque prédiction est TP, TN, FP ou FN. Elle sert ensuite à générer des règles explicatives du modèle."
# prediction sur le train
y_pred_train = model.predict(x_train)
#creation cmc
train_cmc = explainer.make_cmc(x_train, y_train, y_pred_train)
len_rows = train_cmc.shape[0]
#generation des regles exclusive
exclusive_rules, uids_ex = explainer.compute_rule_coverage(explainer.get_exclusive_rules(train_cmc, ['label', 'predicted', 'CMC']), len_rows)


# 2.5 Application des règles exclusives sur le jeu de test (classe majoritaire par défaut)
x_test['label'] = y_test
explainer.apply_exclusive_rules(x_test, exclusive_rules)


# 2.6 Application des règles exclusives sur le jeu de test (prédiction du modèle)
x_test['label'] = y_test
explainer.apply_exclusive_rules(x_test, exclusive_rules, model)

