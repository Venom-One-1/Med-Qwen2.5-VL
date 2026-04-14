"""Configuration dataclasses and JSON helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MAX_OCT_IMAGES,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_LABELS,
)


@dataclass
class TrainConfig:
    model_name_or_path: str = DEFAULT_MODEL_PATH
    image_root: str = DEFAULT_IMAGE_ROOT
    train_manifest: str = "outputs/manifests/train.jsonl"
    val_manifest: str = "outputs/manifests/val.jsonl"
    test_manifest: str = "outputs/manifests/test.jsonl"
    output_dir: str = "outputs/qwen25_vl_multilabel"
    num_labels: int = DEFAULT_NUM_LABELS
    max_oct_images: int = DEFAULT_MAX_OCT_IMAGES
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 12
    learning_rate_lora: float = 2e-4
    learning_rate_head: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    freeze_vision_encoder: bool = True
    grad_checkpointing: bool = True
    bf16: bool = True
    metric_for_best_model: str = "macro_f1"
    early_stopping_patience: int = 3
    threshold_search_points: int = 181
    dataloader_num_workers: int = 0
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
