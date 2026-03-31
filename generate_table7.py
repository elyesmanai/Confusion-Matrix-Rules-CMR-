import pandas as pd

df = pd.read_csv("results/tableau_comparatif.csv", encoding="latin-1")

rows = []

for _, row in df.iterrows():
    rows.append({
        "Dataset": row["Dataset"],
        "Method": row["Methode"],
        "Training Accuracy": row["Train_Exactitude"],
        "Training Coverage": row["Train_Coverage"],
        "Testing Accuracy": row["Test_Exactitude"],
        "Testing Coverage": row["Test_Coverage"],
        "Rule Complexity": row["Complexite"],
        "F1 Hybrid": row["F1_rr"]
    })

table7 = pd.DataFrame(rows)
table7.to_csv("results/table7.csv", index=False, encoding="utf-8")

print(table7)