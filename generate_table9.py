import pandas as pd
import numpy as np

df = pd.read_csv("results/progress_iterations.csv")

def mean_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"

rows = []

for dataset in df["Dataset"].unique():
    for method in ["CMR", "RIPPER", "BRCG"]:

        sub = df[(df["Dataset"] == dataset) & (df["Methode"] == method)]

        if method == "CMR":
            acc = mean_std(sub["Test_Exactitude"])
            cov = mean_std(sub["Test_Coverage"])
            f1 = mean_std(sub["F1_rr"])
        else:
            acc = mean_std(sub["Test_Exactitude"])
            cov = "NA"
            f1 = mean_std(sub["F1_rr"])

        rows.append({
            "Dataset": dataset,
            "Method": method,
            "Test Accuracy (μ ± σ)": acc,
            "Test Coverage (μ ± σ)": cov,
            "Hybrid F1 (μ ± σ)": f1
        })

table9 = pd.DataFrame(rows)
table9.to_csv("results/table9.csv", index=False)

print(table9)