import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

df = pd.read_csv("results/progress_iterations.csv")

def mean_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"

rows = []

datasets = df["Dataset"].unique()

for dataset in datasets:

    # ---------- F1 (CMR vs RIPPER) ----------
    cmr = df[(df["Dataset"] == dataset) & (df["Methode"] == "CMR")]["F1_rr"].dropna().values
    ripper = df[(df["Dataset"] == dataset) & (df["Methode"] == "RIPPER")]["F1_rr"].dropna().values

    n = min(len(cmr), len(ripper))

    if n > 0:
        cmr_vals = cmr[:n]
        ripper_vals = ripper[:n]

        improvement = np.mean(cmr_vals) - np.mean(ripper_vals)
        p = wilcoxon(cmr_vals, ripper_vals).pvalue

        rows.append({
            "Dataset": dataset,
            "Metric": "F1 Hybrid",
            "CMR": mean_std(pd.Series(cmr_vals)),
            "Baseline": mean_std(pd.Series(ripper_vals)),
            "Improvement": round(improvement, 4),
            "P-Value": round(p, 4)
        })

    # ---------- Accuracy (CMR vs BRCG) ----------
    cmr_acc = df[(df["Dataset"] == dataset) & (df["Methode"] == "CMR")]["Test_Exactitude"].dropna().values
    brcg_acc = df[(df["Dataset"] == dataset) & (df["Methode"] == "BRCG")]["Test_Exactitude"].dropna().values

    n = min(len(cmr_acc), len(brcg_acc))

    if n > 0:
        cmr_vals = cmr_acc[:n]
        brcg_vals = brcg_acc[:n]

        improvement = np.mean(cmr_vals) - np.mean(brcg_vals)
        p = wilcoxon(cmr_vals, brcg_vals).pvalue

        rows.append({
            "Dataset": dataset,
            "Metric": "Accuracy",
            "CMR": mean_std(pd.Series(cmr_vals)),
            "Baseline": mean_std(pd.Series(brcg_vals)),
            "Improvement": round(improvement, 4),
            "P-Value": round(p, 4)
        })

table10 = pd.DataFrame(rows)
table10.to_csv("results/table10.csv", index=False)

print(table10)