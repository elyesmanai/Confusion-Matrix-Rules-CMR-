from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score


def train_xgb_blackbox(X_train, y_train):
    """
    Entraine l'algorithme XGBoost sur les données d'entrainement.
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    return model


def get_predictions_model(model, X_train, X_test):
    """
    Retourne les prédictions sur train et test.
    """
    predictions_model_train = model.predict(X_train)
    predictions_model_test = model.predict(X_test)

    return predictions_model_train, predictions_model_test


def evaluate_xgb_blackbox(model, X_train, y_train, X_test, y_test):
    """
    Évalue XGBoost. Permet de verifier que les prédictions sont correctes et d'obtenir les métriques de base.
    """
    predictions_model_train = model.predict(X_train)
    predictions_model_test = model.predict(X_test)

    results = {
        "F1_train_model": float(f1_score(y_train, predictions_model_train, average="weighted", zero_division=0)),
        "F1_test_model": float(f1_score(y_test, predictions_model_test, average="weighted", zero_division=0)),
        "ACC_train_model": float(accuracy_score(y_train, predictions_model_train)),
        "ACC_test_model": float(accuracy_score(y_test, predictions_model_test)),
        "predictions_model_train": predictions_model_train,
        "predictions_model_test": predictions_model_test
    }

    return results