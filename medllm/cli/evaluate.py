"""CLI entrypoint for evaluation."""

from __future__ import annotations

import argparse
import json

from medllm.config import TrainConfig
from medllm.runtime import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ophthalmology multi-label classifier.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL to evaluate.")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON.")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    metrics = evaluate(config, args.checkpoint, args.manifest, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
