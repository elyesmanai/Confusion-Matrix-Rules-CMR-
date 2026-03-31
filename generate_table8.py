from cmr_model import run_cmr
import pandas as pd
import numpy as np

datasets = {
    "KDD99": "../Datasets/KDD99/",
    "BIG15": "../Datasets/BIG15/",
    "DoH20": "../Datasets/DoH20/",
    "UNSW-NB15": "../Datasets/UNSW-NB15/processed/",
    "CIC2017": "../Datasets/CIC2017/"
}

rows = []

for name, path in datasets.items():

    if name == "UNSW-NB15":
        train = pd.read_csv(path + "le_train.csv")
        test = pd.read_csv(path + "le_test.csv")

        X_train = train.drop(["id", "attack_cat", "label"], axis=1)
        y_train = train["label"]

        X_test = test.drop(["id", "attack_cat", "label"], axis=1)
        y_test = test["label"]

    elif name == "CIC2017":
        X_train = pd.read_parquet(path + "X_train.parquet")
        X_test = pd.read_parquet(path + "X_test.parquet")
        y_train = pd.read_csv(path + "y_train.csv").squeeze()
        y_test = pd.read_csv(path + "y_test.csv").squeeze()

    else:
        X_train = pd.read_csv(path + "X_train.csv")
        X_test = pd.read_csv(path + "X_test.csv")
        y_train = pd.read_csv(path + "y_train.csv").squeeze()
        y_test = pd.read_csv(path + "y_test.csv").squeeze()

    for smin in [1, 3, 50]:

        sample_size = 20000
        if len(X_train) > sample_size:
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]

        m = run_cmr(X_train, y_train, X_test, y_test, support_min=smin)

        rows.append({
            "Dataset": name,
            "s_min": smin,
            "Coverage": m["Test_Coverage"],
            "Conflict": m["Conflit"],
            "Hybrid F1": m["F1_rr"]
        })

df = pd.DataFrame(rows)
df.to_csv("results/table8.csv", index=False)

print(df)