"""Discriminative multi-label classifier built on top of Qwen2.5-VL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - import guard for environments without torch
    torch = None
    class _MissingNN:
        Module = object

    nn = _MissingNN()


def _require_runtime_dependency(name: str, module: Any) -> None:
    if module is None:
        raise ImportError(f"{name} is required for this operation but is not installed.")


@dataclass
class MultiLabelOutput:
    loss: Any | None
    logits: Any
    pooled_features: Any


class MultiLabelClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, pooled_features: Any) -> Any:
        return self.net(pooled_features)


class Qwen25VLForOphthalmologyMultiLabel(nn.Module):
    def __init__(
        self,
        base_model: Any,
        *,
        num_labels: int,
        pos_weight: Any | None = None,
        dropout: float = 0.1,
    ) -> None:
        _require_runtime_dependency("torch", torch)
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_labels = num_labels
        hidden_size = self.config.hidden_size
        self.classifier = MultiLabelClassificationHead(hidden_size, num_labels, dropout=dropout)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach(), persistent=False)
        else:
            self.pos_weight = None

    def freeze_vision_encoder(self) -> None:
        for name, param in self.base_model.named_parameters():
            if name.startswith("visual.") or ".visual." in name:
                param.requires_grad = False

    def unfreeze_projector_modules(self) -> None:
        projector_keywords = ("merger", "projector", "multi_modal_projector", "mm_projector")
        for name, param in self.base_model.named_parameters():
            if any(keyword in name for keyword in projector_keywords):
                param.requires_grad = True

    def _contiguous_image_runs(self, input_ids_row: Any) -> list[list[int]]:
        image_token_ids = {
            getattr(self.config, "image_token_id", None),
            getattr(self.config, "vision_token_id", None),
        }
        image_token_ids.discard(None)

        spans: list[list[int]] = []
        current: list[int] = []
        for position, token_id in enumerate(input_ids_row.tolist()):
            if token_id in image_token_ids:
                current.append(position)
            elif current:
                spans.append(current)
                current = []

        if current:
            spans.append(current)
        return spans

    def _vision_span_blocks(self, input_ids_row: Any) -> list[list[int]]:
        start_token = getattr(self.config, "vision_start_token_id", None)
        end_token = getattr(self.config, "vision_end_token_id", None)
        if start_token is None or end_token is None:
            return []

        spans: list[list[int]] = []
        current: list[int] | None = None
        for position, token_id in enumerate(input_ids_row.tolist()):
            if token_id == start_token:
                current = []
                continue
            if token_id == end_token:
                if current:
                    spans.append(current)
                current = None
                continue
            if current is not None:
                current.append(position)
        return spans

    def _extract_image_spans(self, input_ids_row: Any, expected_images: int | None = None) -> list[list[int]]:
        spans = self._vision_span_blocks(input_ids_row)
        if expected_images is not None and len(spans) != expected_images:
            fallback = self._contiguous_image_runs(input_ids_row)
            if len(fallback) >= expected_images:
                spans = fallback[:expected_images]
        elif not spans:
            spans = self._contiguous_image_runs(input_ids_row)

        if expected_images is not None and len(spans) > expected_images:
            spans = spans[:expected_images]
        return spans

    def _hierarchical_pool(self, hidden_states: Any, input_ids: Any, image_counts: Any | None = None) -> Any:
        pooled_samples = []
        batch_size = hidden_states.shape[0]

        for batch_idx in range(batch_size):
            expected_images = None
            if image_counts is not None:
                expected_images = int(image_counts[batch_idx].item())

            spans = self._extract_image_spans(input_ids[batch_idx], expected_images)
            if not spans:
                raise ValueError("Failed to locate image-region token spans in the decoder hidden states.")

            image_embeddings = []
            for span in spans:
                token_states = hidden_states[batch_idx, span, :]
                image_embeddings.append(token_states.mean(dim=0))

            image_tensor = torch.stack(image_embeddings, dim=0)
            pooled_samples.append(image_tensor.mean(dim=0))

        return torch.stack(pooled_samples, dim=0)

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        labels: Any | None = None,
        image_counts: Any | None = None,
        **kwargs: Any,
    ) -> MultiLabelOutput:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        pooled_features = self._hierarchical_pool(hidden_states, input_ids, image_counts=image_counts)
        logits = self.classifier(pooled_features)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(logits, labels)

        return MultiLabelOutput(loss=loss, logits=logits, pooled_features=pooled_features)


def build_classifier_model(
    *,
    model_name_or_path: str,
    num_labels: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    freeze_vision_encoder: bool = True,
    pos_weight: Any | None = None,
) -> tuple[Any, Any]:
    _require_runtime_dependency("torch", torch)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - import guard for missing runtime deps
        raise ImportError(
            "transformers and peft are required to build the classifier model."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model = Qwen25VLForOphthalmologyMultiLabel(
        base_model=base_model,
        num_labels=num_labels,
        pos_weight=pos_weight,
    )

    if freeze_vision_encoder:
        model.freeze_vision_encoder()
    model.unfreeze_projector_modules()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model.base_model = get_peft_model(model.base_model, peft_config)
    return model, processor
