import os
import time
import threading
import numpy as np
import pandas as pd
import psutil

from cmr_pure import _predict_with_rules
from utils import align_test_for_brcg


"""
Les expériences ont été réalisées dans l’environnement technique suivant :
Système d’exploitation : Windows 10
Processeur : Intel64 Family 6 Model 78 (GenuineIntel)
Mémoire RAM : 15.41 Go
Langage de programmation : Python 3.9.25 (64 bits)

Bibliothèques utilisées :

scikit-learn : 1.6.1
NumPy : 1.23.5
pandas : 1.5.3
"""


# =========================
#       Configuration
# =========================
DATASETS = {
    "KDD99": "Datasets/KDD99/",
    "BIG15": "Datasets/BIG15/",
    "UNSW-NB15": "Datasets/UNSW-NB15/processed/",
    "DoH20": "Datasets/DoH20/"
}

COLUMNS = [
    "Dataset", "Méthode", "F1_train", "F1_test", "Train_Coverage", "Train_Exactitude",
    "Test_Coverage", "Test_Exactitude", "Conflit", "Non_couverts", "F1_rmaj", "F1_rr",
    "Nb_règles", "Nb_conditions", "Complexité", "Temps", "RAM_MB", "Iteration"
]


# =========================
# Préparation des données
# =========================
def clean_data(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    return df.astype(np.float32)


def clean_labels(y):
    y = pd.Series(y)
    y = y.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    return y.astype(int).values


def align_test_for_brcg(X_train, X_test):
    """
    Remplace dans X_test les valeurs absentes de X_train par 0.
    """
    X_train_brcg = X_train.copy()
    X_test_brcg = X_test.copy()

    for col in X_train_brcg.columns:
        train_values = set(X_train_brcg[col].unique())
        X_test_brcg[col] = X_test_brcg[col].apply(
            lambda x: x if x in train_values else 0
        )

    return X_train_brcg, X_test_brcg


def load_dataset(name, path):
    """
    Charge un dataset selon son format.
    """
    if name == "UNSW-NB15":
        train = pd.read_csv(path + "le_train.csv")
        test = pd.read_csv(path + "le_test.csv")

        X_train = train.drop(["id", "attack_cat", "label"], axis=1)
        y_train = train["label"]

        X_test = test.drop(["id", "attack_cat", "label"], axis=1)
        y_test = test["label"]
    else:
        X_train = pd.read_csv(path + "X_train.csv")
        X_test = pd.read_csv(path + "X_test.csv")
        y_train = pd.read_csv(path + "y_train.csv").squeeze()
        y_test = pd.read_csv(path + "y_test.csv").squeeze()

    return X_train, y_train, X_test, y_test


def prepare_data(X_train, y_train, X_test, y_test):
    """
    Nettoyage + alignement final avant les modèles.
    """
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)

    y_train = clean_labels(y_train)
    y_test = clean_labels(y_test)

    # Supprime les colonnes constantes sur train
    X_train = X_train.loc[:, X_train.nunique() > 1]

    # Aligne test sur les colonnes du train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, y_train, X_test, y_test


# =========================
# Mesure temps / RAM
# =========================
def run_with_peak_ram(func, *args, interval=0.05, **kwargs):
    """
    Retourne :
    - résultat de la fonction
    - temps écoulé
    - RAM ajoutée en MB
    """
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = {"value": baseline_rss}
    stop_flag = {"stop": False}
    result_holder = {}

    def monitor():
        while not stop_flag["stop"]:
            try:
                rss = process.memory_info().rss
                if rss > peak_rss["value"]:
                    peak_rss["value"] = rss
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)

    start = time.time()
    t.start()

    try:
        result_holder["result"] = func(*args, **kwargs)
    finally:
        stop_flag["stop"] = True
        t.join()

    elapsed = time.time() - start
    added_ram_mb = max(0.0, (peak_rss["value"] - baseline_rss) / 1024**2)

    return result_holder["result"], elapsed, added_ram_mb

# ==============================================
# # Combinaison des métriques de temps et de RAM
# ==============================================
def combine_time_ram(te, tg, ram_te, ram_tg):
    return {
        "TE": round(te, 4),
        "TG": round(tg, 4),
        "TT": round(te + tg, 4),
        "RAM_TE_MB": round(ram_te, 4),
        "RAM_TG_MB": round(ram_tg, 4),
        "RAM_TT_MB": round(ram_te + ram_tg, 4),
    }


# =========================
# METRICS - FIDELITY
# =========================
def compute_fidelity(predictions_model, predictions_rules):
    predictions_model = np.asarray(predictions_model).astype(int)
    predictions_rules = np.asarray(predictions_rules).astype(int)

    return float((predictions_model == predictions_rules).mean() * 100)


def compute_all_fidelities(
    predictions_model_test,
    predictions_CMR,
    predictions_RIPPER,
    predictions_BRCG
):
    fidelity_cmr = compute_fidelity(predictions_model_test, predictions_CMR)
    fidelity_ripper = compute_fidelity(predictions_model_test, predictions_RIPPER)
    fidelity_brcg = compute_fidelity(predictions_model_test, predictions_BRCG)

    return {
        "Fidelity_CMR": fidelity_cmr,
        "Fidelity_RIPPER": fidelity_ripper,
        "Fidelity_BRCG": fidelity_brcg
    }


# =========================================
# METRICS - COVERAGE + FIDELITY COVERED
# =========================================

# =========================
# ||        CMR          ||
# =========================
def compute_cmr_coverage_fidelity(X_test, rules_cmr, predictions_model_test):
    """
    Coverage + Fidelity covered pour CMR
    """
    pred_info = _predict_with_rules(X_test, rules_cmr)

    covered_mask = pred_info["valid_mask"]
    y_pred_partial = pred_info["y_pred_partial"]

    coverage = float(covered_mask.mean() * 100)

    if covered_mask.sum() == 0:
        fidelity_covered = 0.0
    else:
        fidelity_covered = float(
            (
                np.asarray(predictions_model_test)[covered_mask].astype(int)
                ==
                np.asarray(y_pred_partial)[covered_mask].astype(int)
            ).mean() * 100
        )

    return coverage, fidelity_covered


# =========================
# ||      RIPPER         ||
# =========================
def compute_ripper_coverage_fidelity(X_test, ripper_model, predictions_model_test):
    """
    RIPPER couvre tout (toujours une prédiction)
    """
    test_data = X_test.copy()
    test_data["label"] = 0

    predictions_ripper = np.asarray(ripper_model.predict(test_data)).astype(int)

    coverage = 100.0  # RIPPER couvre tout

    fidelity_covered = float(
        (predictions_ripper == np.asarray(predictions_model_test).astype(int)).mean() * 100
    )

    return coverage, fidelity_covered


# =========================
# ||       BRCG          ||
# =========================
def compute_brcg_coverage_fidelity(
    X_train, X_test,
    brcg_model, fb_brcg, columns_brcg,
    predictions_model_test
):
    """
    BRCG couvre tout (modèle logique complet)
    """
    X_train_brcg, X_test_brcg = align_test_for_brcg(X_train, X_test)

    X_test_bin = fb_brcg.transform(X_test_brcg)

    if isinstance(X_test_bin, pd.DataFrame):
        X_test_bin = X_test_bin.fillna(0).astype(int)
    else:
        X_test_bin = pd.DataFrame(
            np.nan_to_num(X_test_bin),
            columns=columns_brcg
        ).astype(int)

    predictions_brcg = np.asarray(brcg_model.predict(X_test_bin)).astype(int)

    coverage = 100.0  # BRCG couvre tout

    fidelity_covered = float(
        (predictions_brcg == np.asarray(predictions_model_test).astype(int)).mean() * 100
    )

    return coverage, fidelity_covered


# =========================
# ||   GLOBAL FUNCTION   ||
# =========================
def compute_all_coverage_fidelity(
    X_train, X_test,
    rules_cmr,
    ripper_model,
    brcg_model, fb_brcg, columns_brcg,
    predictions_model_test
):
    cmr_cov, cmr_fid = compute_cmr_coverage_fidelity(
        X_test, rules_cmr, predictions_model_test
    )

    ripper_cov, ripper_fid = compute_ripper_coverage_fidelity(
        X_test, ripper_model, predictions_model_test
    )

    brcg_cov, brcg_fid = compute_brcg_coverage_fidelity(
        X_train, X_test,
        brcg_model, fb_brcg, columns_brcg,
        predictions_model_test
    )

    return {
        "CMR": (cmr_cov, cmr_fid),
        "RIPPER": (ripper_cov, ripper_fid),
        "BRCG": (brcg_cov, brcg_fid)
    }


# =========================
# Dossiers résultats
# =========================
def ensure_results_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/distributions", exist_ok=True)