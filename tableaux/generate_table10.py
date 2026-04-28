import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


input_path = "../results/progress_iterations.csv"
output_path = "../results/tableaux_results/table10.csv"

os.makedirs("../results/tableaux_results", exist_ok=True)

df = pd.read_csv(input_path, encoding="utf-8-sig", sep=None, engine="python")
df.columns = df.columns.str.strip()


def mean_std(values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return "NaN"

    if len(values) == 1:
        return f"{values.mean():.5f} ± 0.00000"

    return f"{values.mean():.5f} ± {values.std(ddof=1):.5f}"


def mean_value(values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def wilcoxon_pvalue(cmr_values, baseline_values):
    cmr_values = pd.to_numeric(cmr_values, errors="coerce").dropna().values
    baseline_values = pd.to_numeric(baseline_values, errors="coerce").dropna().values

    n = min(len(cmr_values), len(baseline_values))
    cmr_values = cmr_values[:n]
    baseline_values = baseline_values[:n]

    if n < 2:
        return np.nan

    differences = cmr_values - baseline_values

    if np.allclose(differences, 0):
        return 1.0

    try:
        _, p_value = wilcoxon(differences)
        return float(p_value)
    except ValueError:
        return np.nan 
    
rows = []

datasets = df["Dataset"].unique()
baselines = ["RIPPER", "BRCG"]

metrics = {
    "F1": "F1_test",
    "Accuracy": "Test_Exactitude"
}

for dataset in datasets:
    cmr_data = df[
        (df["Dataset"] == dataset) &
        (df["Méthode"] == "CMR")
    ].sort_values("Iteration")

    for metric_name, metric_col in metrics.items():
        cmr_values = cmr_data[metric_col]
        cmr_mean = mean_value(cmr_values)

        for baseline in baselines:
            base_data = df[
                (df["Dataset"] == dataset) &
                (df["Méthode"] == baseline)
            ].sort_values("Iteration")

            base_values = base_data[metric_col]
            base_mean = mean_value(base_values)

            improvement = (
                cmr_mean - base_mean
                if not np.isnan(cmr_mean) and not np.isnan(base_mean)
                else np.nan
            )

            p_value = wilcoxon_pvalue(cmr_values, base_values)

            rows.append({
                "Dataset": dataset,
                "Metric": metric_name,
                "CMR (μ ± σ)": mean_std(cmr_values),
                "Baseline": baseline,
                "Baseline (μ ± σ)": mean_std(base_values),
                "Improvement (CMR - Baseline)": (
                    f"{improvement:.5f}" if not np.isnan(improvement) else "NaN"
                ),
                "p-value": (
                    f"{p_value:.5f}" if not np.isnan(p_value) else "NaN"
                )
            })


table10 = pd.DataFrame(rows)

table10.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    na_rep="NaN"
)

print(table10)