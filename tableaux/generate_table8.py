import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_dataset, prepare_data
from cmr_pure import train_cmr_pure, evaluate_cmr_pure
from xgb_blackbox import train_xgb_blackbox, get_predictions_model # pour obtenir predictions_model_train
from extract_rules import extract_cmr_rules_from_model_predictions # pour obtenir les règles CMR à partir de predictions_model_train


datasets = {
    "KDD99": "../../Datasets/KDD99/",
    "BIG15": "../../Datasets/BIG15/",
    "DoH20": "../../Datasets/DoH20/",
    "UNSW-NB15": "../../Datasets/UNSW-NB15/processed/"
}

rows = []

for name, path in datasets.items():
    print(f"\n===== {name} =====")

    # X_train, y_train, X_test, y_test = load_dataset(name, path)
    # X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # for smin in [1, 3, 50]:
    #     print(f"  -> s_min = {smin}")

    #     payload = train_cmr_pure(
    #         X_train,
    #         y_train,
    #         X_test,
    #         y_test,
    #         support_min=smin
    #     )
    #debut
    X_train, y_train, X_test, y_test = load_dataset(name, path)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # 1) XGBoost -> predictions_model
    xgb_model = train_xgb_blackbox(X_train, y_train)
    predictions_model_train, predictions_model_test = get_predictions_model(
        xgb_model, X_train, X_test
    )

    for smin in [1, 3, 50]:
        print(f"  -> s_min = {smin}")

        # 2) CMR apprend sur predictions_model_train
        rules = extract_cmr_rules_from_model_predictions(
            X_train=X_train,
            predictions_model_train=predictions_model_train,
            support_min=smin
        )

        # 3) Évaluation des règles
        payload = {
            "rules": rules,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": pd.Series(y_train).reset_index(drop=True),
            "y_test": pd.Series(y_test).reset_index(drop=True)
        }
    #fin
        m = evaluate_cmr_pure(payload)

        rows.append({
            "Dataset": name,
            "s_min Value": smin,
            "Rule Coverage (%)": m["Test_Coverage"],
            "Conflict Rate": m["Conflit"],
            "Hybrid F1": np.nan,
            "F1_rmaj": m["F1_rmaj"],
            "F1": m["F1_test"],
            "Nb_règles": m["Nb_règles"],
            "Nb_conditions": m["Nb_conditions"],
            "Complexité": m["Complexité"]
        })

os.makedirs("../results/tableaux_results", exist_ok=True)

df = pd.DataFrame(rows)

df.to_csv(
    "../results/tableaux_results/table8.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="NaN"
)

print(df)