"""CLI entrypoint for classifier training."""

from __future__ import annotations

import argparse
import json

from medllm.config import TrainConfig
from medllm.runtime import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ophthalmology multi-label classifier.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
