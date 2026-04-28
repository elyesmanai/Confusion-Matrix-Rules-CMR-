import numpy as np
import pandas as pd

from cmr_pure import predict_cmr_pure_test_only
from utils import align_test_for_brcg


# =========================
# ||         CMR         ||
# =========================
def predict_cmr(X_train, X_test, y_train, y_test, rules_cmr):
    """
    Raw data (X,Y) --> rules CMR --> predictions_CMR
    Ici, on applique les règles CMR avec la logique du CMR pur
    (fallback majorité).
    """
    payload = {
        "rules": rules_cmr,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": pd.Series(y_train).reset_index(drop=True),
        "y_test": pd.Series(y_test).reset_index(drop=True)
    }

    return np.asarray(predict_cmr_pure_test_only(payload)).astype(int)


# =========================
# ||       RIPPER        ||
# =========================
def predict_ripper(X_test, ripper_model):
    """
    Raw data (X,Y) --> rules RIPPER --> predictions_RIPPER
    """
    test_data = X_test.copy()

    # requis par wittgenstein/RIPPER
    test_data["label"] = 0

    return np.asarray(ripper_model.predict(test_data)).astype(int)


# =========================
# ||        BRCG         ||
# =========================
def predict_brcg(X_train, X_test, brcg_model, fb_brcg, columns_brcg):
    """
    Raw data (X,Y) --> rules BRCG --> predictions_BRCG
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

    return np.asarray(brcg_model.predict(X_test_bin)).astype(int)


# =========================
# ||   PREDICTIONS       ||
# =========================
def get_all_rule_predictions(
    X_train, X_test, y_train, y_test,
    predictions_model_test,
    rules_cmr,
    ripper_model,
    brcg_model, fb_brcg, columns_brcg
):
    predictions_cmr = predict_cmr(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        rules_cmr=rules_cmr
    )

    predictions_ripper = predict_ripper(
        X_test=X_test,
        ripper_model=ripper_model
    )

    predictions_brcg = predict_brcg(
        X_train=X_train,
        X_test=X_test,
        brcg_model=brcg_model,
        fb_brcg=fb_brcg,
        columns_brcg=columns_brcg
    )

    return {
        "predictions_model_test": np.asarray(predictions_model_test).astype(int),
        "predictions_CMR": predictions_cmr,
        "predictions_RIPPER": predictions_ripper,
        "predictions_BRCG": predictions_brcg
    }