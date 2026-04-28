from utils import load_dataset, prepare_data
from extract_rules import extract_all_rules
import warnings


def main():
    dataset_name = "KDD99"
    dataset_path = "../Datasets/KDD99/"

    X_train, y_train, X_test, y_test = load_dataset(dataset_name, dataset_path)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    results = extract_all_rules(X_train, y_train, X_test, y_test)

    print("===== ETAPE : EXTRACTION DES REGLES =====")

    print("\n--- CMR ---")
    rules_cmr = results["rules_cmr"]
    print("Nombre de règles CMR :", len(rules_cmr))
    print(rules_cmr.head())

    print("\n--- RIPPER ---")
    print("Règles RIPPER :")
    print(results["rules_ripper"])

    print("\n--- BRCG ---")
    print("Règles BRCG :")
    print(results["rules_brcg"])


if __name__ == "__main__":
    main()
