"""Manifest construction for ophthalmology multi-label classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from .constants import DEFAULT_MAX_OCT_IMAGES, DEFAULT_NUM_LABELS, IMAGE_EXTENSIONS


@dataclass
class ManifestBuildResult:
    split: str
    manifest_path: Path
    kept: int
    dropped: int
    drop_reasons: Counter[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "manifest_path": str(self.manifest_path),
            "kept": self.kept,
            "dropped": self.dropped,
            "drop_reasons": dict(self.drop_reasons),
        }


def load_label_mapping(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_anno_line(line: str) -> tuple[str, list[int], list[str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise ValueError(f"Invalid anno line with fewer than 3 columns: {line!r}")

    source_key = parts[0]
    oct_indices = [int(value) for value in parts[1].split(",") if value]
    label_names = [value for value in parts[2].split(",") if value]
    return source_key, oct_indices, label_names


def sample_name_from_source_key(source_key: str) -> str:
    return source_key.rsplit("/", 1)[-1]


def _build_oct_candidate(sample_dir: Path, sample_name: str, oct_index: int) -> Path:
    return sample_dir / f"{sample_name}_{oct_index:03d}.jpg"


def _fallback_match_oct(sample_dir: Path, oct_index: int) -> Path | None:
    pattern = re.compile(rf"_(0*{oct_index})\.[^.]+$", re.IGNORECASE)
    matches: list[Path] = []
    for child in sample_dir.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if child.name.endswith(".fundus.jpg"):
            continue
        if pattern.search(child.name):
            matches.append(child)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches.sort(key=lambda path: path.name)
        return matches[0]
    return None


def resolve_sample_paths(
    image_root: str | Path,
    source_key: str,
    oct_indices: Iterable[int],
    *,
    verify_files: bool = True,
) -> tuple[list[str], str | None]:
    sample_dir = Path(image_root) / Path(source_key)
    sample_name = sample_name_from_source_key(source_key)
    fundus_path = sample_dir / f"{sample_name}.fundus.jpg"
    image_paths = [fundus_path]

    if verify_files and not sample_dir.is_dir():
        return [], "missing_sample_dir"
    if verify_files and not fundus_path.is_file():
        return [], "missing_fundus"

    for oct_index in oct_indices:
        candidate = _build_oct_candidate(sample_dir, sample_name, oct_index)
        if verify_files and not candidate.is_file():
            fallback = _fallback_match_oct(sample_dir, oct_index)
            if fallback is None:
                return [], f"missing_oct_{oct_index}"
            candidate = fallback
        image_paths.append(candidate)

    return [str(path.as_posix()) for path in image_paths], None


def build_label_vector(
    label_names: Iterable[str],
    mapping: dict[str, int],
    *,
    num_labels: int = DEFAULT_NUM_LABELS,
) -> tuple[list[int], list[int]]:
    label_ids: list[int] = []
    seen: set[int] = set()
    for label_name in label_names:
        if label_name not in mapping:
            raise KeyError(label_name)
        label_id = mapping[label_name]
        if label_id not in seen:
            label_ids.append(label_id)
            seen.add(label_id)

    label_ids.sort()
    label_vec = [0] * num_labels
    for label_id in label_ids:
        label_vec[label_id] = 1
    return label_ids, label_vec


def build_manifest_for_split(
    *,
    split: str,
    anno_path: str | Path,
    image_root: str | Path,
    mapping_path: str | Path,
    output_path: str | Path,
    verify_files: bool = True,
    max_oct_images: int = DEFAULT_MAX_OCT_IMAGES,
    num_labels: int = DEFAULT_NUM_LABELS,
) -> ManifestBuildResult:
    anno = Path(anno_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(mapping_path)

    drop_reasons: Counter[str] = Counter()
    kept = 0
    dropped = 0

    with anno.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            if not raw_line.strip():
                continue

            source_key, oct_indices, label_names = parse_anno_line(raw_line)

            if not oct_indices:
                dropped += 1
                drop_reasons["empty_oct_indices"] += 1
                continue

            if len(oct_indices) > max_oct_images:
                dropped += 1
                drop_reasons["too_many_oct_images"] += 1
                continue

            try:
                label_ids, label_vec = build_label_vector(label_names, mapping, num_labels=num_labels)
            except KeyError:
                dropped += 1
                drop_reasons["unknown_label"] += 1
                continue

            image_paths, failure = resolve_sample_paths(
                image_root=image_root,
                source_key=source_key,
                oct_indices=oct_indices,
                verify_files=verify_files,
            )
            if failure is not None:
                dropped += 1
                drop_reasons[failure] += 1
                continue

            sample_id = sample_name_from_source_key(source_key)
            record = {
                "sample_id": sample_id,
                "split": split,
                "source_key": source_key,
                "image_paths": image_paths,
                "label_names": label_names,
                "label_ids": label_ids,
                "label_vec": label_vec,
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    return ManifestBuildResult(
        split=split,
        manifest_path=output,
        kept=kept,
        dropped=dropped,
        drop_reasons=drop_reasons,
    )
