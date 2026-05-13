import pandas as pd
import numpy as np

df = pd.read_csv("../results/tableau_comparaison_moyenne.csv", encoding="utf-8-sig",sep=";")
rows = []

for _, row in df.iterrows():
    rows.append({
        "Dataset": row["Dataset"],
        "Method": row["Méthode"],
        "Training Accuracy": row["Train_Exactitude"],
        "Training Coverage": row["Train_Coverage"],
        "Testing Accuracy": row["Test_Exactitude"],
        "Testing Coverage": row["Test_Coverage"],
        "Rule Complexity": row["Complexité"],
        "Test Time (s)": row["Temps"],
        "Test Memory (MB)": row["RAM_MB"],
        "Training Time (TE)": row["TE"],
        "Rule Generation Time (TG)": row["TG"],
        "Total Explanation Time (TT)": row["TT"],
        "Training Memory (MB)": row["RAM_TE_MB"],
        "Rule Generation Memory (MB)": row["RAM_TG_MB"],
        "Total Explanation Memory (MB)": row["RAM_TT_MB"],
        "F1 Hybrid": row["F1_rr"],
    })

table7 = pd.DataFrame(rows)

table7.to_csv("../results/tableaux_results/table7.csv", index=False, encoding="utf-8",na_rep="NaN")

print(table7)