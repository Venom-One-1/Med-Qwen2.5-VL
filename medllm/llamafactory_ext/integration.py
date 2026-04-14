"""Reusable hooks for plugging the task-specific logic into a Llama-Factory fork."""

from __future__ import annotations

from typing import Any

import numpy as np

from medllm.config import TrainConfig
from medllm.metrics import compute_multilabel_metrics, search_best_thresholds
from medllm.modeling_qwen25_vl_classifier import build_classifier_model
from medllm.runtime import build_dataloader, compute_pos_weight_from_manifest

try:
    import torch
except ImportError:  # pragma: no cover - import guard
    torch = None


def build_model_and_processor_for_lf(config: TrainConfig) -> tuple[Any, Any]:
    pos_weight = None
    if torch is not None:
        pos_weight_np = compute_pos_weight_from_manifest(config.train_manifest)
        pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32)

    return build_classifier_model(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        freeze_vision_encoder=config.freeze_vision_encoder,
        pos_weight=pos_weight,
    )


def build_train_dataloader_for_lf(config: TrainConfig, processor: Any, *, distributed: bool = False) -> Any:
    return build_dataloader(
        config.train_manifest,
        processor,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=True,
        distributed=distributed,
    )


def build_eval_dataloader_for_lf(config: TrainConfig, processor: Any, *, distributed: bool = False) -> Any:
    return build_dataloader(
        config.val_manifest,
        processor,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=False,
        distributed=distributed,
    )


def evaluate_predictions_for_lf(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold_search_points: int,
) -> dict[str, Any]:
    thresholds = search_best_thresholds(labels, probabilities, num_points=threshold_search_points)
    metrics = compute_multilabel_metrics(labels, probabilities, thresholds=thresholds)
    metrics["thresholds"] = thresholds.tolist()
    return metrics
