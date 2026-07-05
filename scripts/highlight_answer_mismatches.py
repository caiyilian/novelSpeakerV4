#!/usr/bin/env python3
"""Generate an answers review file with strict label mismatches highlighted."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANSWER_RE = re.compile(r"【([^】]+)】")
DEFAULT_DATA_DIRS = [
    Path("data"),
    Path("data/volume2"),
    Path("data/volume3"),
    Path("data/volume4"),
    Path("data/volume5"),
    Path("data/volume7"),
]


def split_candidates(value: str) -> set[str]:
    return {part.strip() for part in value.split("|") if part.strip()}


def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def highlight_answers(data_dir: Path, output_name: str) -> tuple[int, int, Path, list[str]]:
    answers_path = data_dir / "answers.txt"
    labels_path = data_dir / "labeled.txt"
    output_path = data_dir / output_name
    warnings: list[str] = []

    if not answers_path.exists():
        raise FileNotFoundError(f"answers.txt not found: {answers_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labeled.txt not found: {labels_path}")

    labels = load_labels(labels_path)
    marker_index = 0
    mismatch_count = 0
    marker_count = 0
    output_lines: list[str] = []

    for line in answers_path.read_text(encoding="utf-8").splitlines():
        def replace(match: re.Match[str]) -> str:
            nonlocal marker_index, marker_count, mismatch_count
            expected = match.group(1).strip()
            marker_count += 1

            if marker_index >= len(labels):
                marker_index += 1
                mismatch_count += 1
                return f"【{expected}--><MISSING_LABEL>】"

            got = labels[marker_index].strip()
            marker_index += 1

            if got in split_candidates(expected):
                return match.group(0)

            mismatch_count += 1
            return f"【{expected}-->{got}】"

        output_lines.append(ANSWER_RE.sub(replace, line))

    if marker_count != len(labels):
        warnings.append(
            f"marker count ({marker_count}) != labeled line count ({len(labels)})"
        )

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return marker_count, mismatch_count, output_path, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create answers_review.txt with mismatched labels highlighted as 【expected-->got】."
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        type=Path,
        help="Data directory containing answers.txt and labeled.txt. Can be repeated.",
    )
    parser.add_argument(
        "--all-known",
        action="store_true",
        help="Process data, data/volume2, data/volume3, data/volume4, data/volume5, and data/volume7.",
    )
    parser.add_argument(
        "--output-name",
        default="answers_review.txt",
        help="Output file name inside each data directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dirs = []
    if args.all_known:
        data_dirs.extend(DEFAULT_DATA_DIRS)
    if args.data_dir:
        data_dirs.extend(args.data_dir)
    if not data_dirs:
        raise SystemExit("Use --all-known or at least one --data-dir.")

    for data_dir in data_dirs:
        marker_count, mismatch_count, output_path, warnings = highlight_answers(
            data_dir, args.output_name
        )
        print(f"{data_dir}: wrote {output_path} ({mismatch_count}/{marker_count} mismatches)")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
