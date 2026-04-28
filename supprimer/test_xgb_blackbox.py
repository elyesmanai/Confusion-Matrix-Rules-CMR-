from utils import load_dataset, prepare_data
from xgb_blackbox import train_xgb_blackbox, get_predictions_model, evaluate_xgb_blackbox


def main():
    dataset_name = "KDD99"
    dataset_path = "../Datasets/KDD99/"

    # 1) charger les données
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, dataset_path)

    # 2) nettoyer les données
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # 3) entraîner XGBoost
    model = train_xgb_blackbox(X_train, y_train)

    # 4) récupérer predictions_model
    predictions_model_train, predictions_model_test = get_predictions_model(model, X_train, X_test)

    # 5) afficher quelques infos
    print("===== ETAPE 1 : XGBOOST =====")
    print("Nombre d'exemples train :", len(X_train))
    print("Nombre d'exemples test  :", len(X_test))
    print("Premières prédictions train :", predictions_model_train[:10])
    print("Premières prédictions test  :", predictions_model_test[:10])

    # 6) évaluer le modèle
    results = evaluate_xgb_blackbox(model, X_train, y_train, X_test, y_test)

    print("\n===== PERFORMANCE XGBOOST =====")
    print("F1_train_model  :", round(results["F1_train_model"], 4))
    print("F1_test_model   :", round(results["F1_test_model"], 4))
    print("ACC_train_model :", round(results["ACC_train_model"], 4))
    print("ACC_test_model  :", round(results["ACC_test_model"], 4))


if __name__ == "__main__":
    main()