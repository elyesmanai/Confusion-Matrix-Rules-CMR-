import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from brcg_algorithm import train_brcg, evaluate_brcg
from ripper_algorithm import train_ripper, evaluate_ripper
from cmr_xgb_pipeline import train_cmr, evaluate_cmr, predict_cmr_test_only
from cmr_pure import train_cmr_pure, evaluate_cmr_pure, predict_cmr_pure_test_only
from tqdm import tqdm

from utils import (
    DATASETS,
    COLUMNS,
    ensure_results_dirs,
    align_test_for_brcg,
    load_dataset,
    prepare_data,
    run_with_peak_ram,
)


n_iterations = 1
np.random.seed(42)


# |------------------------- BRCG -------------------------|
def evaluate_brcg_block(X_train, y_train, X_test, y_test):
    X_train_brcg, X_test_brcg = align_test_for_brcg(X_train, X_test)

    model, fb, columns, _, _, rules = train_brcg(
        X_train_brcg, y_train, X_test_brcg, y_test
    )

    def test_only():
        X_test_bin = fb.transform(X_test_brcg)
        return model.predict(X_test_bin)

    _, elapsed, ram = run_with_peak_ram(test_only)

    return evaluate_brcg(
        model=model,
        fb=fb,
        X_train=X_train_brcg,
        y_train=y_train,
        X_test=X_test_brcg,
        y_test=y_test,
        elapsed=elapsed,
        ram=ram,
        columns=columns,
        rules=rules
    )


# |------------------------- RIPPER -------------------------|
def evaluate_ripper_block(X_train, y_train, X_test, y_test):
    model, _, _ = train_ripper(X_train, y_train, X_test, y_test)

    def test_only():
        test_data = X_test.copy()
        test_data["label"] = y_test
        return model.predict(test_data)

    _, elapsed, ram = run_with_peak_ram(test_only)

    return evaluate_ripper(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        elapsed=elapsed,
        ram=ram
    )


# |------------------------- CMR HYBRIDE -------------------------|
def evaluate_cmr_block(X_train, y_train, X_test, y_test):
    payload = train_cmr(X_train, y_train, X_test, y_test)

    _, elapsed, ram = run_with_peak_ram(predict_cmr_test_only, payload)

    return evaluate_cmr(payload, elapsed=elapsed, ram=ram)


# |------------------------- CMR PUR -------------------------|
def evaluate_cmr_pure_block(X_train, y_train, X_test, y_test):
    payload = train_cmr_pure(X_train, y_train, X_test, y_test)

    _, elapsed, ram = run_with_peak_ram(predict_cmr_pure_test_only, payload)

    return evaluate_cmr_pure(payload, elapsed=elapsed, ram=ram)


def test_models(X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test)

    tqdm.write("  -> BRCG")
    brcg_res = evaluate_brcg_block(X_train, y_train, X_test, y_test)

    tqdm.write("  -> RIPPER")
    ripper_res = evaluate_ripper_block(X_train, y_train, X_test, y_test)

    tqdm.write("  -> CMR_hybride")
    cmr_h_res = evaluate_cmr_block(X_train, y_train, X_test, y_test)

    tqdm.write("  -> CMR_pur")
    cmr_p_res = evaluate_cmr_pure_block(X_train, y_train, X_test, y_test)

    return {
        "BRCG": brcg_res,
        "RIPPER": ripper_res,
        "CMR_hybride": cmr_h_res,
        "CMR_pur": cmr_p_res
    }


def save_distributions():
    progress_path = "results/progress_iterations.csv"

    if os.path.exists(progress_path):
        df_progress = pd.read_csv(progress_path)

        for metric in ["F1_test", "Temps", "RAM_MB"]:
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

    for name, path in tqdm(list(DATASETS.items()),total=len(DATASETS),desc="Datasets"):
        X_train, y_train, X_test, y_test = load_dataset(name, path)

        for it in tqdm(range(n_iterations),total=n_iterations,desc=f"{name} - itérations",leave=False):
            res = test_models(X_train, y_train, X_test, y_test)
            results_all_iterations[name].append(res)

            for method in ["BRCG", "RIPPER", "CMR_hybride", "CMR_pur"]:
                row = [name, method] + [
                    res[method].get(col, np.nan) for col in COLUMNS[2:-1]
                ] + [it + 1]
                progress_rows.append(row)

            pd.DataFrame(progress_rows, columns=COLUMNS).to_csv(
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

        for method in ["BRCG", "RIPPER", "CMR_hybride", "CMR_pur"]:
            valid_runs = [r[method] for r in runs if r is not None and method in r]

            if valid_runs:
                df_metrics = pd.DataFrame(valid_runs)
                avg = df_metrics.mean(numeric_only=True).to_dict()
                row = [ds, method] + [avg.get(c, None) for c in COLUMNS[2:-1]]
                final_rows.append(row)

    df_results = pd.DataFrame(final_rows, columns=COLUMNS[:-1])
    df_results = df_results.sort_values(by=["Dataset", "Méthode"])

    df_results.to_csv(
        "results/tableau_comparatif.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN"
    )

    save_distributions()

    print(df_results)


if __name__ == "__main__":
    main()