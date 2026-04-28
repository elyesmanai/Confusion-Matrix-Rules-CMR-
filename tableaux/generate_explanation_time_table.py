import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_dataset, prepare_data, align_test_for_brcg
from xgb_blackbox import train_xgb_blackbox, get_predictions_model
from extract_rules import extract_cmr_rules_from_model_predictions
from ripper_algorithm import train_ripper
from brcg_algorithm import train_brcg


datasets = {
    "KDD99": "../../Datasets/KDD99/",
    "BIG15": "../../Datasets/BIG15/",
    "DoH20": "../../Datasets/DoH20/",
    "UNSW-NB15": "../../Datasets/UNSW-NB15/processed/"
}

rows = []

for name, path in datasets.items():
    print(f"\n===== {name} =====")

    X_train, y_train, X_test, y_test = load_dataset(name, path)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # Modèle de base : utilisé seulement pour produire predictions_model
    xgb_model = train_xgb_blackbox(X_train, y_train)
    predictions_model_train, predictions_model_test = get_predictions_model(
        xgb_model,
        X_train,
        X_test
    )

    # =========================
    # CMR : temps extraction règles
    # =========================
    print("  -> CMR")

    start = time.time()
    rules_cmr = extract_cmr_rules_from_model_predictions(
        X_train,
        predictions_model_train,
        support_min=3
    )
    elapsed_cmr = time.time() - start

    rows.append({
        "Dataset": name,
        "Method": "CMR",
        "Explanation Time (seconds)": round(elapsed_cmr, 4),
        "Explanation Time (minutes)": round(elapsed_cmr / 60, 4),
        "Nb_rules": len(rules_cmr)
    })

    # =========================
    # RIPPER : temps apprentissage règles
    # =========================
    print("  -> RIPPER")

    start = time.time()
    ripper_model, ripper_metrics, rules_ripper = train_ripper(
        x_train=X_train,
        y_train=predictions_model_train,
        x_test=X_train,
        y_test=predictions_model_train
    )
    elapsed_ripper = time.time() - start

    rows.append({
        "Dataset": name,
        "Method": "RIPPER",
        "Explanation Time (seconds)": round(elapsed_ripper, 4),
        "Explanation Time (minutes)": round(elapsed_ripper / 60, 4),
        "Nb_rules": np.nan
    })

    # =========================
    # BRCG : temps apprentissage règles
    # =========================
    print("  -> BRCG")

    X_train_brcg, _ = align_test_for_brcg(X_train, X_train)

    start = time.time()
    brcg_model, fb_brcg, columns_brcg, train_time, brcg_metrics, rules_brcg = train_brcg(
        x_train=X_train_brcg,
        y_train=predictions_model_train,
        x_test=X_train_brcg,
        y_test=predictions_model_train
    )
    elapsed_brcg = time.time() - start

    rows.append({
        "Dataset": name,
        "Method": "BRCG",
        "Explanation Time (seconds)": round(elapsed_brcg, 4),
        "Explanation Time (minutes)": round(elapsed_brcg / 60, 4),
        "Nb_rules": np.nan
    })


os.makedirs("../results/tableaux_results", exist_ok=True)

df = pd.DataFrame(rows)

df.to_csv(
    "../results/tableaux_results/table_explanation_time.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="NaN"
)

print("\n===== TABLE : TEMPS DE GENERATION DES EXPLICATIONS =====")
print(df)