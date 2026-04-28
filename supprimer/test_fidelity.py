from utils import load_dataset, prepare_data
from extract_rules import extract_all_rules
from covered_fidelity import compute_all_coverage_fidelity
import warnings

warnings.filterwarnings(
    "ignore",
    message="This use of `*` has resulted in matrix multiplication"
)


def main():
    dataset_name = "KDD99"
    dataset_path = "../Datasets/KDD99/"

    # données
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, dataset_path)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # extraction règles
    extracted = extract_all_rules(X_train, y_train, X_test, y_test)

    # calcul coverage + fidelity
    results = compute_all_coverage_fidelity(
        X_train=X_train,
        X_test=X_test,
        rules_cmr=extracted["rules_cmr"],
        ripper_model=extracted["ripper_model"],
        brcg_model=extracted["brcg_model"],
        fb_brcg=extracted["fb_brcg"],
        columns_brcg=extracted["columns_brcg"],
        predictions_model_test=extracted["predictions_model_test"]
    )

    print("===== COVERAGE + FIDELITY (COVERED) =====")

    for method, (cov, fid) in results.items():
        print(f"\n--- {method} ---")
        print(f"Coverage         : {cov:.2f}%")
        print(f"Fidelity covered : {fid:.2f}%")


if __name__ == "__main__":
    main()