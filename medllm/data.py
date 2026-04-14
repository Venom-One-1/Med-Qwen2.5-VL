"""Datasets and collators for discriminative ophthalmology fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_PROMPT_TEMPLATE

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - import guard for environments without torch
    torch = None
    Dataset = object  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - import guard for environments without pillow
    Image = None


def _require_runtime_dependency(name: str, module: Any) -> None:
    if module is None:
        raise ImportError(f"{name} is required for this operation but is not installed.")


def build_modal_prompt(image_count: int) -> str:
    if image_count < 2:
        raise ValueError("Each sample must contain at least one CFP and one OCT image.")

    oct_descriptions = []
    for image_idx in range(2, image_count + 1):
        oct_descriptions.append(f"第{image_idx}张图像是OCT。")

    return " ".join(
        [
            DEFAULT_PROMPT_TEMPLATE,
            "第1张图像是眼底彩照（CFP）。",
            *oct_descriptions,
            "请输出用于二十类眼底病多标签判别的融合特征。",
        ]
    )


class OphthalmologyManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "sample_id": record["sample_id"],
            "source_key": record["source_key"],
            "image_paths": record["image_paths"],
            "labels": record["label_vec"],
            "prompt": build_modal_prompt(len(record["image_paths"])),
        }


class OphthalmologyCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        _require_runtime_dependency("torch", torch)
        _require_runtime_dependency("Pillow", Image)

        images: list[list[Any]] = []
        texts: list[str] = []
        labels: list[list[int]] = []
        sample_ids: list[str] = []
        source_keys: list[str] = []
        image_counts: list[int] = []

        for feature in features:
            sample_images = [Image.open(path).convert("RGB") for path in feature["image_paths"]]
            images.append(sample_images)
            texts.append(feature["prompt"])
            labels.append(feature["labels"])
            sample_ids.append(feature["sample_id"])
            source_keys.append(feature["source_key"])
            image_counts.append(len(feature["image_paths"]))

        batch = self.processor(
            images=images,
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        batch["image_counts"] = torch.tensor(image_counts, dtype=torch.long)
        batch["sample_ids"] = sample_ids
        batch["source_keys"] = source_keys
        return batch
