from utils import load_dataset, prepare_data
from extract_rules import extract_all_rules
from predictions_rules import get_all_rule_predictions
import warnings

warnings.filterwarnings(
    "ignore",
    message="This use of `*` has resulted in matrix multiplication"
)


def main():
    dataset_name = "KDD99"
    dataset_path = "../Datasets/KDD99/"

    # Charger les données
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, dataset_path)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # Étape 2 : extraction des règles
    extracted = extract_all_rules(X_train, y_train, X_test, y_test)

    # Étape 3 : appliquer les règles
    results = get_all_rule_predictions(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions_model_test=extracted["predictions_model_test"],
        rules_cmr=extracted["rules_cmr"],
        ripper_model=extracted["ripper_model"],
        brcg_model=extracted["brcg_model"],
        fb_brcg=extracted["fb_brcg"],
        columns_brcg=extracted["columns_brcg"]
    )

    print("===== ETAPE : PREDICTIONS DES REGLES =====")

    print("\nPremières predictions_model_test :")
    print(results["predictions_model_test"][:10])

    print("\nPremières predictions_CMR :")
    print(results["predictions_CMR"][:10])

    print("\nPremières predictions_RIPPER :")
    print(results["predictions_RIPPER"][:10])

    print("\nPremières predictions_BRCG :")
    print(results["predictions_BRCG"][:10])

    print("\nTailles :")
    print("predictions_model_test :", len(results["predictions_model_test"]))
    print("predictions_CMR        :", len(results["predictions_CMR"]))
    print("predictions_RIPPER     :", len(results["predictions_RIPPER"]))
    print("predictions_BRCG       :", len(results["predictions_BRCG"]))


if __name__ == "__main__":
    main()