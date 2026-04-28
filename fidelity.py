import numpy as np


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