import sys
import os
import pandas as pd

from adultdata_preparation_evaluat import prepare_adult_dataset
from xgboost_rule_extraction import train_xgboost
from brcg_algorithm import train_brcg, evaluate_brcg
from ripper_algorithm import train_ripper, evaluate_ripper

os.makedirs("results/adult", exist_ok=True)


def main():
    original_stdout = sys.stdout

    output_path = "results/adult/adult_output.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        sys.stdout = f

        # =========================
        # || 1. Charger données  ||
        # =========================
        X_train, X_test, y_train, y_test, mapping = prepare_adult_dataset()
        print("Données Adult chargées")

        X_train = X_train.drop(columns=["label"], errors="ignore")
        X_test = X_test.drop(columns=["label"], errors="ignore")

        # =========================
        # || 2. XGBoost          || 
        # =========================
        xgb_model, y_pred_xgb, xgb_metrics = train_xgboost(
            X_train, y_train, X_test, y_test
        )

        print("\n=== XGBoost ===")
        print(xgb_metrics)

        # =========================
        # || 3. BRCG             ||
        # =========================
        brcg_model_, fb_brcg, columns_brcg, _, _, brcg_rules = train_brcg(
            X_train, y_train, X_test, y_test
        )

        brcg_eval = evaluate_brcg(
            model=brcg_model_,
            fb=fb_brcg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            elapsed=None,
            ram=None,
            columns=columns_brcg,
            rules=brcg_rules
        )

        print("\n=== BRCG ===")
        print(brcg_eval)

        # =========================
        # || 4. RIPPER           ||
        # =========================
        ripper_model_, _, _ = train_ripper(
            X_train, y_train, X_test, y_test
        )

        ripper_eval = evaluate_ripper(
            model=ripper_model_,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            elapsed=None,
            ram=None
        )

        print("\n=== RIPPER ===")
        print(ripper_eval)

        # =========================
        # || 5. Comparaison      ||
        # =========================
        print("\n===== Comparaison des modèles =====")
        print(f"XGBoost  -> Accuracy: {xgb_metrics['accuracy']:.4f} | F1: {xgb_metrics['f1']:.4f}")
        print(f"BRCG     -> Accuracy: {brcg_eval['Test_Exactitude']:.4f} | F1: {brcg_eval['F1_test']:.4f}")
        print(f"RIPPER   -> Accuracy: {ripper_eval['Test_Exactitude']:.4f} | F1: {ripper_eval['F1_test']:.4f}")

    sys.stdout = original_stdout
    print(f"Résultats enregistrés dans : {output_path}")


if __name__ == "__main__":
    main()