import pandas as pd
import numpy as np
import os

input_path = "../results/progress_iterations.csv"
output_path = "../results/tableaux_results/table9.csv"

os.makedirs("../results/tableaux_results", exist_ok=True)

df = pd.read_csv(input_path, encoding="utf-8-sig",)
df.columns = df.columns.str.strip()

def mean_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NaN"
    if len(s) == 1:
        return f"{s.mean():.5f} ± 0.00000"
    return f"{s.mean():.5f} ± {s.std(ddof=1):.5f}"

rows = []

for dataset in df["Dataset"].unique():
    for method in ["CMR", "RIPPER", "BRCG"]:

        sub = df[(df["Dataset"] == dataset) & (df["Méthode"] == method)]

        if sub.empty:
            continue

        rows.append({
            "Dataset": dataset,
            "Method": method,
            "Test Accuracy (μ ± σ)": mean_std(sub["Test_Exactitude"]),
            "Test Coverage (μ ± σ)": mean_std(sub["Test_Coverage"]),
            "Hybrid F1 (μ ± σ)": "NaN",
            "F1 test (μ ± σ)": mean_std(sub["F1_test"])
        })

table9 = pd.DataFrame(rows)

table9.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    na_rep="NaN"
)

print(table9)