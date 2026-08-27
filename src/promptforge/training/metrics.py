from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_regression_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 100.0)
    labels = np.asarray(labels, dtype=np.float64)

    mae = mean_absolute_error(labels, predictions)
    rmse = float(np.sqrt(mean_squared_error(labels, predictions)))

    correlations: list[float] = []
    for i in range(labels.shape[1]):
        try:
            corr, _ = pearsonr(labels[:, i], predictions[:, i])
            correlations.append(float(corr) if np.isfinite(corr) else 0.0)
        except Exception:
            correlations.append(0.0)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "pearson": float(np.mean(correlations)),
    }


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    """Hugging Face Trainer-compatible metrics callback."""
    predictions, labels = eval_prediction
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return compute_regression_metrics(predictions, labels)


def per_dimension_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> dict[str, dict[str, float]]:
    predictions = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 100.0)
    labels = np.asarray(labels, dtype=np.float64)

    report: dict[str, dict[str, float]] = {}
    for i, name in enumerate(label_names):
        mae = mean_absolute_error(labels[:, i], predictions[:, i])
        rmse = float(np.sqrt(mean_squared_error(labels[:, i], predictions[:, i])))
        pearson = float(pearsonr(labels[:, i], predictions[:, i])[0])
        spearman = float(spearmanr(labels[:, i], predictions[:, i])[0])
        report[name] = {
            "mae": float(mae),
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
        }
    return report


def overall_from_dimensions(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    pred_overall = np.asarray(predictions, dtype=np.float64).mean(axis=1)
    true_overall = np.asarray(labels, dtype=np.float64).mean(axis=1)
    return {
        "mae": float(mean_absolute_error(true_overall, pred_overall)),
        "rmse": float(np.sqrt(mean_squared_error(true_overall, pred_overall))),
        "pearson": float(pearsonr(true_overall, pred_overall)[0]),
        "spearman": float(spearmanr(true_overall, pred_overall)[0]),
    }
