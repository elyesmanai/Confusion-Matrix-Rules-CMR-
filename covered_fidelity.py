import numpy as np
import pandas as pd

from cmr_pure import _predict_with_rules
from utils import align_test_for_brcg


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