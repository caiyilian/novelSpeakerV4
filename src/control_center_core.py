"""Non-UI services for the NovelSpeaker desktop control center."""

from __future__ import annotations

import os
import json
import hashlib
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


def find_project_root(start_paths: Iterable[Path]) -> Path | None:
    """Find the nearest ancestor containing the real first-volume novel."""
    seen: set[Path] = set()
    for start in start_paths:
        resolved = Path(start).expanduser().resolve()
        for candidate in (resolved, *resolved.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "data" / "novel.txt").is_file():
                return candidate
    return None


def _detect_project_root() -> Path:
    configured = os.environ.get("NOVELSPEAKER_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        detected = find_project_root((executable_dir, Path.cwd()))
        return detected or executable_dir

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _detect_project_root()
DIALOGUE_PATTERN = re.compile(r"「([^」]+)」")
MODEL_FILTER = "sense-nova,agnes"
FIRST_PASS_PROGRESS_PATTERN = re.compile(
    r"\[(?P<completed>\d+)\s*/\s*(?P<total>\d+)\].*?"
    r"remaining=(?P<eta>(?:\d+[dhms])+)",
    re.IGNORECASE,
)
REVIEW_PROGRESS_PATTERN = re.compile(
    r"\[Review\s+(?P<phase>[^\]\s]+)\s+"
    r"(?P<completed>\d+)\s*/\s*(?P<total>\d+).*?\].*?"
    r"ETA=(?P<eta>(?:\d+[dhms])+)",
    re.IGNORECASE,
)
_DURATION_PART_PATTERN = re.compile(r"(\d+)\s*([dhms])", re.IGNORECASE)

REVIEW_PHASE_LABELS = {
    "scene-review": "场景复审",
    "local-entity-graph": "人物实体复审",
    "mirrored-jury": "镜像裁决",
    "transition-gate": "连贯性复核",
    "complete": "全卷复审",
    "review": "全卷复审",
}


@dataclass(frozen=True)
class VolumeSpec:
    number: int
    name: str
    data_dir: Path

    @property
    def novel_path(self) -> Path:
        return self.data_dir / "novel.txt"

    @property
    def answers_path(self) -> Path:
        return self.data_dir / "answers.txt"

    @property
    def labeled_path(self) -> Path:
        return self.data_dir / "labeled.txt"

    @property
    def annotation_log_path(self) -> Path:
        return self.data_dir / "label_log.jsonl"


@dataclass(frozen=True)
class VolumeProgress:
    labeled: int
    total: int

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return min(100, round(self.labeled * 100 / self.total))

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.labeled >= self.total


@dataclass(frozen=True)
class ProgressHint:
    phase: str
    completed: int
    total: int
    eta_seconds: float

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return min(100, round(self.completed * 100 / self.total))


@dataclass(frozen=True)
class ReviewProgress:
    phase: str
    status: str
    completed: int
    total: int
    eta_seconds: float | None

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return min(100, round(self.completed * 100 / self.total))

    @property
    def active(self) -> bool:
        return self.status not in {"", "complete"} and self.phase != "complete"

    @property
    def phase_label(self) -> str:
        return REVIEW_PHASE_LABELS.get(self.phase, self.phase or "全卷复审")


def volume_specs(root: Path = PROJECT_ROOT) -> list[VolumeSpec]:
    data_root = Path(root) / "data"
    return [
        VolumeSpec(
            number=number,
            name=f"第{number}卷",
            data_dir=data_root if number == 1 else data_root / f"volume{number}",
        )
        for number in range(1, 6)
    ]


def count_dialogues(novel_path: Path) -> int:
    try:
        text = Path(novel_path).read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum(len(DIALOGUE_PATTERN.findall(line)) for line in text.splitlines())


def count_labeled(labeled_path: Path) -> int:
    try:
        with Path(labeled_path).open("r", encoding="utf-8") as handle:
            return sum(1 for _line in handle)
    except OSError:
        return 0


def read_progress(spec: VolumeSpec) -> VolumeProgress:
    return VolumeProgress(
        labeled=count_labeled(spec.labeled_path),
        total=count_dialogues(spec.novel_path),
    )


def parse_duration_seconds(value: str) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    matches = list(_DURATION_PART_PATTERN.finditer(text))
    if not matches or "".join(match.group(0) for match in matches) != text:
        return None
    multipliers = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return float(
        sum(int(match.group(1)) * multipliers[match.group(2)] for match in matches)
    )


def format_eta(seconds: float | int | None) -> str:
    if seconds is None:
        return "预计时间计算中"
    remaining = max(0, int(round(float(seconds))))
    if remaining < 60:
        return "预计不到 1 分钟"
    days, remainder = divmod(remaining, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"预计剩余 {days} 天 {hours} 小时"
    if hours:
        return f"预计剩余 {hours} 小时 {minutes} 分钟"
    return f"预计剩余 {minutes} 分钟"


def parse_progress_line(line: str) -> ProgressHint | None:
    text = str(line or "")
    review_match = REVIEW_PROGRESS_PATTERN.search(text)
    if review_match:
        eta = parse_duration_seconds(review_match.group("eta"))
        if eta is not None:
            return ProgressHint(
                phase=review_match.group("phase"),
                completed=int(review_match.group("completed")),
                total=int(review_match.group("total")),
                eta_seconds=eta,
            )

    first_pass_match = FIRST_PASS_PROGRESS_PATTERN.search(text)
    if first_pass_match:
        eta = parse_duration_seconds(first_pass_match.group("eta"))
        if eta is not None:
            return ProgressHint(
                phase="first-pass",
                completed=int(first_pass_match.group("completed")),
                total=int(first_pass_match.group("total")),
                eta_seconds=eta,
            )
    return None


def read_review_progress(spec: VolumeSpec) -> ReviewProgress | None:
    path = spec.data_dir / "volume_review_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None

    progress = state.get("progress")
    if not isinstance(progress, dict):
        return None
    try:
        completed = max(0, int(progress.get("completed") or 0))
        total = max(0, int(progress.get("total") or 0))
        eta_value = progress.get("eta_seconds")
        eta_seconds = None if eta_value is None else max(0.0, float(eta_value))
    except (TypeError, ValueError):
        return None
    return ReviewProgress(
        phase=str(progress.get("phase") or "review"),
        status=str(progress.get("status") or ""),
        completed=completed,
        total=total,
        eta_seconds=eta_seconds,
    )


def read_recent_failure(spec: VolumeSpec) -> str:
    path = spec.data_dir / "label_log.temp.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""

    for line in reversed(tail.splitlines()):
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        event_name = event.get("event")
        if event_name == "round_complete":
            return ""
        if event_name == "round_failed":
            error = event.get("error") or {}
            return str(error.get("message") or error.get("type") or "标注进程异常退出")
    return ""


def child_python_executable(executable: str | None = None) -> str:
    path = Path(executable or sys.executable)
    if getattr(sys, "frozen", False):
        return str(path)
    if path.stem.lower() == "pythonw":
        console_python = path.with_name("python.exe")
        if console_python.exists():
            return str(console_python)
    return str(path)


def build_run_arguments(
    spec: VolumeSpec,
    reset: bool = False,
    *,
    frozen: bool | None = None,
) -> list[str]:
    packaged = getattr(sys, "frozen", False) if frozen is None else frozen
    arguments = [
        "--provider",
        "api-fallback",
        "--data-dir",
        str(spec.data_dir),
        "--api-model",
        MODEL_FILTER,
        "--api-round-robin-offset",
        str(spec.number - 1),
        "--health-check",
        "none",
        "--api-context-limit",
        "0",
        "--count",
        "9999",
        "--validate",
    ]
    if reset:
        arguments.append("--reset-state")
    if packaged:
        return ["--worker", *arguments]
    return ["-X", "utf8", str(PROJECT_ROOT / "src" / "run_label.py"), *arguments]


def build_process_environment(
    key_file: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(base if base is not None else os.environ)
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8:backslashreplace",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "SENSENOVA_API_KEYS_FILE": str(Path(key_file).resolve()),
            "SENSENOVA_USE_ENV_PROXY": "0",
            "AGNES_USE_ENV_PROXY": "0",
            "NOVELSPEAKER_PROJECT_ROOT": str(PROJECT_ROOT),
        }
    )
    agnes_key_file = PROJECT_ROOT / "config" / "agnes_api_key"
    if agnes_key_file.is_file():
        environment["AGNES_API_KEY_FILE"] = str(agnes_key_file)
    return environment


def runtime_files(data_dir: Path) -> Iterable[Path]:
    names = (
        "labeled.txt",
        "labeled.first_pass.txt",
        "label_log.jsonl",
        "label_log.temp.jsonl",
        "character_state.json",
        "evidence_vault.json",
        "volume_review.jsonl",
        "volume_review.temp.jsonl",
        "volume_review_state.json",
    )
    for name in names:
        path = Path(data_dir) / name
        if path.is_file():
            yield path


def unique_backup_directory(base_dir: Path, spec: VolumeSpec) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path(base_dir) / timestamp
    candidate = parent / f"volume{spec.number}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"volume{spec.number}_{suffix}"
        suffix += 1
    return candidate


def _file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _runtime_snapshot(data_dir: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: _file_fingerprint(path)
        for path in runtime_files(data_dir)
    }


def _latest_volume_backup(backup_root: Path, spec: VolumeSpec) -> Path | None:
    candidates = [
        path
        for path in Path(backup_root).glob(f"*/volume{spec.number}*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _backup_snapshot(backup_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = backup_dir / "backup_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if isinstance(files, dict):
            return files
    except (OSError, TypeError, ValueError):
        pass
    return {
        path.name: _file_fingerprint(path)
        for path in runtime_files(backup_dir)
    }


def backup_volume(spec: VolumeSpec, backup_root: Path | None = None) -> tuple[Path, int]:
    root = Path(backup_root) if backup_root else PROJECT_ROOT / "backup" / "control_center"
    snapshot = _runtime_snapshot(spec.data_dir)
    latest = _latest_volume_backup(root, spec)
    if latest is not None and snapshot == _backup_snapshot(latest):
        return latest, 0

    target = unique_backup_directory(root, spec)
    target.mkdir(parents=True, exist_ok=False)

    copied = 0
    for source in runtime_files(spec.data_dir):
        shutil.copy2(source, target / source.name)
        copied += 1

    manifest = target / "backup_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "volume": spec.number,
                "data_dir": str(spec.data_dir),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "files": snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target, copied
