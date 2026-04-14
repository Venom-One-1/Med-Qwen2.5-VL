"""Metrics and threshold search helpers for multi-label classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def search_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    num_points: int = 181,
) -> np.ndarray:
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    candidate_thresholds = np.linspace(0.05, 0.95, num_points)

    for label_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_score = -1.0
        label_true = y_true[:, label_idx]
        label_prob = y_prob[:, label_idx]

        for threshold in candidate_thresholds:
            label_pred = (label_prob >= threshold).astype(np.int32)
            score = f1_score(label_true, label_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds[label_idx] = best_threshold

    return thresholds


def _specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def _safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return 0.0


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.0


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)
    per_class_results = []

    for label_idx in range(y_true.shape[1]):
        label_true = y_true[:, label_idx]
        label_prob = y_prob[:, label_idx]
        label_pred = y_pred[:, label_idx]
        per_class = {
            "label_id": label_idx,
            "threshold": float(thresholds[label_idx]),
            "f1": float(f1_score(label_true, label_pred, zero_division=0)),
            "precision": float(precision_score(label_true, label_pred, zero_division=0)),
            "recall": float(recall_score(label_true, label_pred, zero_division=0)),
            "specificity": _specificity_score(label_true, label_pred),
            "ap": _safe_average_precision(label_true, label_prob),
            "auroc": _safe_roc_auc(label_true, label_prob),
        }
        per_class_results.append(per_class)

    macro_ap = float(np.mean([item["ap"] for item in per_class_results])) if per_class_results else 0.0
    macro_auroc = float(np.mean([item["auroc"] for item in per_class_results])) if per_class_results else 0.0

    results: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "mAP": macro_ap,
        "macro_auroc": macro_auroc,
        "thresholds": thresholds.tolist(),
        "per_class": per_class_results,
    }

    return results


def save_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
