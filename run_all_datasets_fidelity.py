import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore",
    message="This use of `*` has resulted in matrix multiplication"
)

from tqdm import tqdm

from utils import (
    DATASETS,
    ensure_results_dirs,
    load_dataset,
    prepare_data,
    run_with_peak_ram,
)

from extract_rules import extract_all_rules
from predictions_rules import get_all_rule_predictions, predict_brcg
from fidelity import compute_all_fidelities
from covered_fidelity import compute_all_coverage_fidelity

from cmr_pure import evaluate_cmr_pure, predict_cmr_pure_test_only
from ripper_algorithm import evaluate_ripper
from brcg_algorithm import evaluate_brcg


n_iterations = 10
np.random.seed(42)

# Colonnes fichier d'itérations
COLUMNS_ITER = [
    "Dataset", "Méthode",
    "F1_train", "F1_test",
    "Train_Coverage", "Train_Exactitude",
    "Test_Coverage", "Test_Exactitude",
    "Conflit", "Non_couverts",
    "F1_rmaj", "F1_rr",
    "Nb_règles", "Nb_conditions", "Complexité",
    "Temps", "RAM_MB",
    "Coverage_total", "Fidelity_covered",
    "Iteration"
]

# Colonnes du tableau final moyen
COLUMNS_FINAL = COLUMNS_ITER[:-1]


# =========================
# CMR : payload métriques
# =========================
def build_cmr_payload(X_train, X_test, y_train, y_test, rules_cmr):
    return {
        "rules": rules_cmr,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": pd.Series(y_train).reset_index(drop=True),
        "y_test": pd.Series(y_test).reset_index(drop=True)
    }


# =========================
# CMR : temps / RAM test only
# =========================
def evaluate_cmr_block(X_train, y_train, X_test, y_test, rules_cmr):
    payload = build_cmr_payload(X_train, X_test, y_train, y_test, rules_cmr)

    _, elapsed, ram = run_with_peak_ram(predict_cmr_pure_test_only, payload)

    metrics = evaluate_cmr_pure(payload, elapsed=elapsed, ram=ram)
    return metrics


# =========================
# RIPPER : temps / RAM test only
# =========================
def evaluate_ripper_block(X_train, y_train, X_test, y_test, ripper_model):
    def test_only():
        test_data = X_test.copy()
        test_data["label"] = 0
        return ripper_model.predict(test_data)

    _, elapsed, ram = run_with_peak_ram(test_only)

    metrics = evaluate_ripper(
        model=ripper_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        elapsed=elapsed,
        ram=ram
    )
    return metrics


# =========================
# BRCG : temps / RAM test only
# =========================
def evaluate_brcg_block(X_train, y_train, X_test, y_test, extracted):
    def test_only():
        return predict_brcg(
            X_train=X_train,
            X_test=X_test,
            brcg_model=extracted["brcg_model"],
            fb_brcg=extracted["fb_brcg"],
            columns_brcg=extracted["columns_brcg"]
        )

    _, elapsed, ram = run_with_peak_ram(test_only)

    metrics = evaluate_brcg(
        model=extracted["brcg_model"],
        fb=extracted["fb_brcg"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        elapsed=elapsed,
        ram=ram,
        columns=extracted["columns_brcg"],
        rules=extracted["rules_brcg"]
    )
    return metrics


# ===================================================
# Ajout métriques coverage total + fidelity covered |
# ===================================================
def attach_new_metrics(metrics_dict, coverage_total, fidelity_covered):
    out = metrics_dict.copy()
    out["Coverage_total"] = float(coverage_total)
    out["Fidelity_covered"] = float(fidelity_covered)
    return out


def test_models(X_train, y_train, X_test, y_test):
    # sécurité : si jamais ce n'est pas déjà fait
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    # 1) extraction des règles depuis predictions_model
    extracted = extract_all_rules(X_train, y_train, X_test, y_test)

    # 2) prédictions finales des règles
    pred_results = get_all_rule_predictions(
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

    # 3) fidelity globale 
    _ = compute_all_fidelities(
        predictions_model_test=pred_results["predictions_model_test"],
        predictions_CMR=pred_results["predictions_CMR"],
        predictions_RIPPER=pred_results["predictions_RIPPER"],
        predictions_BRCG=pred_results["predictions_BRCG"]
    )

    # 4) coverage total + fidelity covered
    covered_results = compute_all_coverage_fidelity(
        X_train=X_train,
        X_test=X_test,
        rules_cmr=extracted["rules_cmr"],
        ripper_model=extracted["ripper_model"],
        brcg_model=extracted["brcg_model"],
        fb_brcg=extracted["fb_brcg"],
        columns_brcg=extracted["columns_brcg"],
        predictions_model_test=extracted["predictions_model_test"]
    )

    # 5) métriques classiques
    tqdm.write("  -> CMR")
    cmr_metrics = evaluate_cmr_block(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        rules_cmr=extracted["rules_cmr"]
    )

    tqdm.write("  -> RIPPER")
    ripper_metrics = evaluate_ripper_block(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        ripper_model=extracted["ripper_model"]
    )

    tqdm.write("  -> BRCG")
    brcg_metrics = evaluate_brcg_block(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        extracted=extracted
    )

    # 6) ajout coverage_total + fidelity_covered
    cmr_metrics = attach_new_metrics(
        cmr_metrics,
        covered_results["CMR"][0],
        covered_results["CMR"][1]
    )

    ripper_metrics = attach_new_metrics(
        ripper_metrics,
        covered_results["RIPPER"][0],
        covered_results["RIPPER"][1]
    )

    brcg_metrics = attach_new_metrics(
        brcg_metrics,
        covered_results["BRCG"][0],
        covered_results["BRCG"][1]
    )

    return {
        "CMR": cmr_metrics,
        "RIPPER": ripper_metrics,
        "BRCG": brcg_metrics
    }


def save_distributions():
    progress_path = "results/progress_iterations.csv"

    if os.path.exists(progress_path):
        df_progress = pd.read_csv(progress_path)

        for metric in ["F1_test", "Temps", "RAM_MB", "Coverage_total", "Fidelity_covered"]:
            if metric in df_progress.columns and not df_progress[metric].isna().all():
                plt.figure()
                df_progress.boxplot(column=metric, by=["Dataset", "Méthode"])
                plt.title(metric)
                plt.suptitle("")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"results/distributions/{metric}.png")
                plt.close()


def main():
    ensure_results_dirs()

    results_all_iterations = {ds: [] for ds in DATASETS}
    progress_rows = []

    for name, path in tqdm(list(DATASETS.items()), total=len(DATASETS), desc="Datasets"):
        X_train, y_train, X_test, y_test = load_dataset(name, path)

        for it in tqdm(range(n_iterations), total=n_iterations, desc=f"{name} - itérations", leave=False):
            res = test_models(X_train, y_train, X_test, y_test)
            results_all_iterations[name].append(res)

            for method in ["CMR", "RIPPER", "BRCG"]:
                row = [name, method] + [
                    res[method].get(col, np.nan) for col in COLUMNS_ITER[2:-1]
                ] + [it + 1]
                progress_rows.append(row)

            pd.DataFrame(progress_rows, columns=COLUMNS_ITER).to_csv(
                "results/progress_iterations.csv",
                index=False,
                encoding="utf-8-sig",
                na_rep="NaN"
            )

        gc.collect()

    # =========================
    # Tableau final moyen
    # =========================
    final_rows = []

    for ds in DATASETS.keys():
        runs = results_all_iterations.get(ds, [])

        for method in ["CMR", "RIPPER", "BRCG"]:
            valid_runs = [r[method] for r in runs if r is not None and method in r]

            if valid_runs:
                df_metrics = pd.DataFrame(valid_runs)
                avg = df_metrics.mean(numeric_only=True).to_dict()
                row = [ds, method] + [avg.get(c, None) for c in COLUMNS_FINAL[2:]]
                final_rows.append(row)

    df_results = pd.DataFrame(final_rows, columns=COLUMNS_FINAL)
    df_results = df_results.sort_values(by=["Dataset", "Méthode"])

    df_results.to_csv(
        "results/tableau_comparaison_moyenne.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN"
    )

    save_distributions()

    print(df_results)


if __name__ == "__main__":
    main()