"""CLI for manifest construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medllm.constants import DEFAULT_IMAGE_ROOT, DEFAULT_SPLIT_TO_ANNO
from medllm.manifest import build_manifest_for_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL manifests for ophthalmology classification.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mapping-path", default="VisualSearch/mapping_20classes.json")
    parser.add_argument("--output-dir", default="outputs/manifests")
    parser.add_argument("--max-oct-images", type=int, default=5)
    parser.add_argument("--skip-image-validation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split, anno_path in DEFAULT_SPLIT_TO_ANNO.items():
        result = build_manifest_for_split(
            split=split,
            anno_path=anno_path,
            image_root=args.image_root,
            mapping_path=args.mapping_path,
            output_path=output_dir / f"{split}.jsonl",
            verify_files=not args.skip_image_validation,
            max_oct_images=args.max_oct_images,
        )
        summaries.append(result.to_dict())

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
