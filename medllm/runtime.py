"""Training, evaluation, and inference helpers for the classification task."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from .config import TrainConfig, save_json
from .data import OphthalmologyCollator, OphthalmologyManifestDataset
from .metrics import compute_multilabel_metrics, save_metrics, search_best_thresholds
from .modeling_qwen25_vl_classifier import build_classifier_model

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ImportError:  # pragma: no cover - import guard for environments without torch
    torch = None
    AdamW = None
    DataLoader = None
    DistributedSampler = None


def _require_runtime_dependency(name: str, module: Any) -> None:
    if module is None:
        raise ImportError(f"{name} is required for this operation but is not installed.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[bool, int, int, Any]:
    _require_runtime_dependency("torch", torch)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.distributed.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and torch is not None and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compute_pos_weight_from_manifest(path: str | Path) -> np.ndarray:
    records = load_manifest_records(path)
    labels = np.asarray([record["label_vec"] for record in records], dtype=np.float32)
    positive = labels.sum(axis=0)
    negative = labels.shape[0] - positive
    positive = np.clip(positive, 1.0, None)
    return negative / positive


def build_dataloader(
    manifest_path: str | Path,
    processor: Any,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    distributed: bool,
) -> Any:
    _require_runtime_dependency("torch", torch)
    dataset = OphthalmologyManifestDataset(manifest_path)
    collator = OphthalmologyCollator(processor)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if key in {"sample_ids", "source_keys"}:
            moved[key] = value
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _iter_trainable_named_parameters(model: Any) -> tuple[list[Any], list[Any]]:
    lora_and_projector: list[Any] = []
    head: list[Any] = []
    for name, param in model.named_parameters():
        normalized_name = name[7:] if name.startswith("module.") else name
        if not param.requires_grad:
            continue
        if normalized_name.startswith("classifier."):
            head.append(param)
        else:
            lora_and_projector.append(param)
    return lora_and_projector, head


def build_optimizer(model: Any, config: TrainConfig) -> Any:
    _require_runtime_dependency("torch", torch)
    lora_and_projector, head = _iter_trainable_named_parameters(model)
    return AdamW(
        [
            {"params": lora_and_projector, "lr": config.learning_rate_lora, "weight_decay": config.weight_decay},
            {"params": head, "lr": config.learning_rate_head, "weight_decay": config.weight_decay},
        ]
    )


def train(config: TrainConfig) -> dict[str, Any]:
    _require_runtime_dependency("torch", torch)

    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover
        raise ImportError("transformers is required for training.") from exc

    set_seed(config.seed)
    distributed, rank, _, device = setup_distributed()
    pos_weight_np = compute_pos_weight_from_manifest(config.train_manifest)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    model, processor = build_classifier_model(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        freeze_vision_encoder=config.freeze_vision_encoder,
        pos_weight=pos_weight,
    )
    if config.grad_checkpointing and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

    train_loader = build_dataloader(
        config.train_manifest,
        processor,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=True,
        distributed=distributed,
    )
    val_loader = build_dataloader(
        config.val_manifest,
        processor,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=False,
        distributed=distributed,
    )

    optimizer = build_optimizer(model, config)
    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(steps_per_epoch * config.num_train_epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []

    try:
        for epoch in range(config.num_train_epochs):
            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                batch = _move_batch_to_device(batch, device)
                outputs = model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()
                running_loss += float(loss.item())

                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            val_predictions = predict_loader(model, val_loader, device, distributed=distributed)
            if rank == 0:
                thresholds = search_best_thresholds(
                    val_predictions["labels"],
                    val_predictions["probabilities"],
                    num_points=config.threshold_search_points,
                )
                val_metrics = compute_multilabel_metrics(
                    val_predictions["labels"],
                    val_predictions["probabilities"],
                    thresholds=thresholds,
                )
                val_metrics["epoch"] = epoch + 1
                val_metrics["train_loss"] = running_loss / max(len(train_loader), 1)
                history.append(val_metrics)

                current_metric = float(val_metrics[config.metric_for_best_model])
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    checkpoint = {
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "config": config.to_dict(),
                        "thresholds": thresholds.tolist(),
                        "best_metric": best_metric,
                        "epoch": best_epoch,
                    }
                    torch.save(checkpoint, output_dir / "best_model.pt")
                    save_metrics(output_dir / "best_val_metrics.json", val_metrics)
                else:
                    epochs_without_improvement += 1

            if distributed:
                stop_tensor = torch.tensor(
                    [1 if rank == 0 and epochs_without_improvement >= config.early_stopping_patience else 0],
                    device=device,
                )
                torch.distributed.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            else:
                should_stop = epochs_without_improvement >= config.early_stopping_patience

            if should_stop:
                break

        if rank == 0:
            save_json(output_dir / "train_history.json", {"history": history, "best_epoch": best_epoch, "best_metric": best_metric})
            save_json(output_dir / "resolved_config.json", config.to_dict())
            return {"best_epoch": best_epoch, "best_metric": best_metric}
        return {"best_epoch": best_epoch, "best_metric": best_metric}
    finally:
        cleanup_distributed(distributed)


def predict_loader(model: Any, dataloader: Any, device: Any, *, distributed: bool = False) -> dict[str, Any]:
    _require_runtime_dependency("torch", torch)
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            sample_ids.extend(batch["sample_ids"])
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            label_vec = batch["labels"].detach().cpu().numpy()
            probabilities.append(probs)
            labels.append(label_vec)

    num_labels = unwrap_model(model).num_labels
    gathered = {
        "sample_ids": sample_ids,
        "probabilities": np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, num_labels)),
        "labels": np.concatenate(labels, axis=0) if labels else np.empty((0, num_labels)),
    }
    if distributed and torch.distributed.is_initialized():
        gathered_objects = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_objects, gathered)
        merged_sample_ids: list[str] = []
        merged_probabilities: list[np.ndarray] = []
        merged_labels: list[np.ndarray] = []
        for item in gathered_objects:
            merged_sample_ids.extend(item["sample_ids"])
            if item["probabilities"].size:
                merged_probabilities.append(item["probabilities"])
            if item["labels"].size:
                merged_labels.append(item["labels"])

        return {
            "sample_ids": merged_sample_ids,
            "probabilities": np.concatenate(merged_probabilities, axis=0) if merged_probabilities else np.empty((0, num_labels)),
            "labels": np.concatenate(merged_labels, axis=0) if merged_labels else np.empty((0, num_labels)),
        }

    return gathered


def evaluate(config: TrainConfig, checkpoint_path: str | Path, manifest_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    _require_runtime_dependency("torch", torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, processor = build_classifier_model(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        freeze_vision_encoder=config.freeze_vision_encoder,
        pos_weight=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    dataloader = build_dataloader(
        manifest_path,
        processor,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
    )
    predictions = predict_loader(model, dataloader, device)
    thresholds = np.asarray(checkpoint.get("thresholds"), dtype=np.float32)
    metrics = compute_multilabel_metrics(
        predictions["labels"],
        predictions["probabilities"],
        thresholds=thresholds,
    )
    save_metrics(output_path, metrics)
    return metrics


def predict(
    config: TrainConfig,
    checkpoint_path: str | Path,
    image_paths: list[str],
) -> dict[str, Any]:
    _require_runtime_dependency("torch", torch)

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required for prediction.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, processor = build_classifier_model(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        freeze_vision_encoder=config.freeze_vision_encoder,
        pos_weight=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    images = [Image.open(path).convert("RGB") for path in image_paths]
    text = [
        "你将接收一组眼科图像。第1张图像是眼底彩照（CFP），其余图像是OCT。"
        "请综合所有图像提取用于多标签判别的视觉特征。"
    ]
    batch = processor(images=[images], text=text, padding=True, return_tensors="pt")
    batch["image_counts"] = torch.tensor([len(image_paths)], dtype=torch.long)
    batch = _move_batch_to_device(batch, device)

    with torch.no_grad():
        outputs = model(**batch)
        probabilities = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]

    thresholds = np.asarray(checkpoint.get("thresholds"), dtype=np.float32)
    pred_vec = (probabilities >= thresholds).astype(np.int32).tolist()
    pred_labels = [idx for idx, value in enumerate(pred_vec) if value == 1]
    return {
        "image_paths": image_paths,
        "probabilities": probabilities.tolist(),
        "pred_vec": pred_vec,
        "pred_labels": pred_labels,
        "thresholds": thresholds.tolist(),
    }
