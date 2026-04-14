"""CLI entrypoint for ad hoc prediction."""

from __future__ import annotations

import argparse
import json

from medllm.config import TrainConfig
from medllm.runtime import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict labels for one CFP + 1~5 OCT images.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--image", action="append", required=True, help="Image path. Repeat for all input images.")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    result = predict(config, args.checkpoint, args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
