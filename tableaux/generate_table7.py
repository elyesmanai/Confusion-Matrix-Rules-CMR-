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
        "F1 Hybrid": row["F1_rr"],
    })

table7 = pd.DataFrame(rows)

table7.to_csv("../results/tableaux_results/table7.csv", index=False, encoding="utf-8",na_rep="NaN")

print(table7)