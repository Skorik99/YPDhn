import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    fbeta_score,
)

from .cifar_utils import calculate_cifar_metrics


__all__ = [
    "compute_metrics",
    "metrics_report",
    "print_metrics",
    "stopping_criterion",
]


def compute_metrics(
    target,
    prediction,
    pathology_names,
    probs,
):
    """
    Compute metrics from input predictions and
    ground truth labels

    :param target: Ground truth labels
    :param prediction: Predicted labels
    :param pathology_names: Class names of pathologies
    :return: pandas DataFrame with metrics and confusion matrices for each class
    """
    df = pd.DataFrame(
        columns=pathology_names,
        index=[
            "Specificity",
            "Sensitivity",
            "G-mean",
            "f1-score",
            "fbeta2-score",
            "ROC-AUC",
            "AP",
            "Precision (PPV)",
            "NPV",
        ],
    )
    conf_mat_df = pd.DataFrame(columns=["TN", "FP", "FN", "TP"], index=pathology_names)
    target = np.array(target, int)
    prediction = np.array(prediction, int)
    probs = np.array(probs)

    for i, col in enumerate(pathology_names):
        try:
            tn, fp, fn, tp = confusion_matrix(target[:, i], prediction[:, i]).ravel()
            df.loc["Specificity", col] = tn / (tn + fp)
            df.loc["Sensitivity", col] = tp / (tp + fn)
            df.loc["G-mean", col] = math.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
            df.loc["f1-score", col] = f1_score(target[:, i], prediction[:, i])
            df.loc["fbeta2-score", col] = fbeta_score(
                target[:, i], prediction[:, i], beta=2
            )
            df.loc["ROC-AUC", col] = roc_auc_score(target[:, i], probs[:, i])
            df.loc["AP", col] = average_precision_score(target[:, i], prediction[:, i])
            df.loc["Precision (PPV)", col] = tp / (tp + fp)
            df.loc["NPV", col] = 0 if tn + fn == 0 else tn / (tn + fn)

            conf_mat_df.loc[col] = [tn, fp, fn, tp]
        except ValueError:
            raise
    return df, conf_mat_df


def metrics_report(
    targets: np.ndarray,
    bin_preds: np.ndarray,
    pathology_names: list,
    probs: np.ndarray,
    verbose: bool = True,
) -> tuple:
    """Returns metrics (ROC-AUC, AP, f1-score, Specificity, Sensitivity, Precision) and confusion matrix

    :param targets: True labels
    :type targets: numpy.ndarray
    :param bin_preds: Predicted labels
    :type bin_preds: numpy.ndarray
    :param pathology_names: Pathology class names
    :type pathology_names: list
    :param verbose: Display metrics in console, defaults to True
    :type verbose: bool, optional
    :return: Metrics and confusion matrix
    :rtype: tuple
    """
    metrics, conf_matrix = compute_metrics(targets, bin_preds, pathology_names, probs)
    if verbose:
        print(metrics)
        print(conf_matrix)
        print(classification_report(targets, bin_preds, zero_division=False))
    return metrics, conf_matrix


def print_metrics(results, thresholds, pathology_names):
    print("=============METRICS REPORT=============")
    metrics_dict = {}
    for t in thresholds:
        print("Metrics with `threshold = {}`".format(t))
        metrics, conf_matrix = metrics_report(
            results["true_labels"],
            results[str(t)],
            pathology_names,
            results["probs"],
            verbose=True,
        )
        sub_dict = {"metrics": metrics, "confusion_matrix": conf_matrix}
        metrics_dict[str(t)] = sub_dict
    return metrics_dict


def stopping_criterion(
    val_loss,
    metrics,
    best_metrics,
    epochs_no_improve,
):
    """
    Define stopping criterion for metrics from config['saving_metrics']
    best_metrics is updated only if every metric from best_metrics.keys() has improved

    :param val_loss: validation loss
    :param metrics: validation metrics
    :param best_metrics: the best metrics for the current epoch
    :param epochs_no_improve: number of epochs without best_metrics updating

    :return: epochs_no_improve, best_metrics
    """
    # get average metrics by class
    metrics = dict(metrics.mean(axis=1))
    # define condition best_metric >= metric for all except for loss
    metrics_mask = all(
        metrics[key] >= best_metrics[key] for key in best_metrics.keys() - {"loss"}
    )
    if not metrics_mask:
        epochs_no_improve += 1
        return epochs_no_improve, best_metrics
    if "loss" in list(best_metrics.keys()):
        if val_loss >= best_metrics["loss"]:
            epochs_no_improve += 1
            return epochs_no_improve, best_metrics
    # Updating best_metrics
    for key in list(best_metrics.keys()):
        if key == "loss":
            best_metrics[key] = val_loss
        else:
            best_metrics[key] = metrics[key]
    # Updating epochs_no_improve
    epochs_no_improve = 0
    return epochs_no_improve, best_metrics


def calculate_metrics(
    fin_targets,
    fin_outputs,
    verbose=False,
):
    # Get results
    softmax = torch.nn.Softmax(dim=1)
    results = softmax(torch.as_tensor(fin_outputs)).max(dim=1)[1]
    fin_targets = torch.as_tensor(fin_targets)
    # Calc metrics
    metrics = calculate_cifar_metrics(fin_targets, results, verbose)
    prediction_threshold = None
    return metrics, prediction_threshold


def check_metrics_names(metrics):
    allowed_metrics = [
        "loss",
        "Specificity",
        "Sensitivity",
        "G-mean",
        "f1-score",
        "fbeta2-score",
        "ROC-AUC",
        "AP",
        "Precision (PPV)",
        "NPV",
    ]

    assert all(
        [k in allowed_metrics for k in metrics.keys()]
    ), f"federated_params.server_saving_metrics can be only {allowed_metrics}, but get {metrics.keys()}"
