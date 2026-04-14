# Llama-Factory Integration Notes

This repository does not vendor a full `Llama-Factory` checkout, so the task-specific code lives in `medllm/`.
The intended integration model is:

1. Keep `Llama-Factory` as the training framework and launcher.
2. Import the local `medllm` modules from a custom fork stage or trainer hook.
3. Reuse:
   - `medllm.manifest` for manifest construction
   - `medllm.data` for dataset and collator logic
   - `medllm.modeling_qwen25_vl_classifier` for discriminative pooling + classification head
   - `medllm.metrics` for threshold search and evaluation
   - `medllm.runtime` as a reference implementation of the task-specific loop

Recommended fork integration points inside Llama-Factory:

- Register a new stage or task for multi-label classification.
- Load the local model path `Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage` instead of the official upstream model.
- Swap the standard generative loss for the classifier output in `Qwen25VLForOphthalmologyMultiLabel`.
- Use the manifest JSONL files as the dataset source.
- Run validation with per-class threshold search and choose the best checkpoint by `macro_f1`.

The example config under `examples/` mirrors the expected runtime parameters.
