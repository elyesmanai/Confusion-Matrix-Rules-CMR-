from xgb_blackbox import train_xgb_blackbox, get_predictions_model
from ripper_algorithm import train_ripper
from brcg_algorithm import train_brcg
from utils import align_test_for_brcg
from explainer import make_cmc, get_exclusive_rules
import pandas as pd


def extract_cmr_rules_from_model_predictions(X_train, predictions_model_train, support_min=3):
    """
    Entraîne CMR pour imiter predictions_model_train.
    """
    X_train = X_train.copy().reset_index(drop=True)
    predictions_model_train = pd.Series(predictions_model_train).astype(int).reset_index(drop=True)

    train_cmc = make_cmc(X_train.copy(), predictions_model_train, predictions_model_train)

    rules = get_exclusive_rules(
        train_cmc,
        analysis_features=["label", "predicted", "CMC"]
    )

    rules = rules[rules["support"] >= support_min].reset_index(drop=True)
    return rules


def extract_ripper_rules_from_model_predictions(X_train, predictions_model_train):
    """
    Entraîne RIPPER pour imiter predictions_model_train.
    """
    model, metrics, rules = train_ripper(
        x_train=X_train,
        y_train=predictions_model_train,
        x_test=X_train,
        y_test=predictions_model_train
    )
    return model, rules


def extract_brcg_rules_from_model_predictions(X_train, predictions_model_train):
    """
    Entraîne BRCG pour imiter predictions_model_train.
    """
    X_train_brcg, _ = align_test_for_brcg(X_train, X_train)

    model, fb, columns, train_time, metrics, rules = train_brcg(
        x_train=X_train_brcg,
        y_train=predictions_model_train,
        x_test=X_train_brcg,
        y_test=predictions_model_train
    )

    return model, fb, columns, rules


def extract_all_rules(X_train, y_train, X_test, y_test):
    """
    predictions_model --> XAI --> rules
    """

    # 1) modèle boîte noire
    xgb_model = train_xgb_blackbox(X_train, y_train)

    # 2) predictions_model
    predictions_model_train, predictions_model_test = get_predictions_model(
        xgb_model, X_train, X_test
    )

    # 3) règles CMR apprises sur predictions_model_train
    rules_cmr = extract_cmr_rules_from_model_predictions(
        X_train, predictions_model_train
    )

    # 4) règles RIPPER apprises sur predictions_model_train
    ripper_model, rules_ripper = extract_ripper_rules_from_model_predictions(
        X_train, predictions_model_train
    )

    # 5) règles BRCG apprises sur predictions_model_train
    brcg_model, fb_brcg, columns_brcg, rules_brcg = extract_brcg_rules_from_model_predictions(
        X_train, predictions_model_train
    )

    return {
        "xgb_model": xgb_model,
        "predictions_model_train": predictions_model_train,
        "predictions_model_test": predictions_model_test,
        "rules_cmr": rules_cmr,
        "ripper_model": ripper_model,
        "rules_ripper": rules_ripper,
        "brcg_model": brcg_model,
        "fb_brcg": fb_brcg,
        "columns_brcg": columns_brcg,
        "rules_brcg": rules_brcg
    }
