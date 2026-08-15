# -*- coding: utf-8 -*-
"""Resumable full-volume second-pass dialogue speaker review."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from quote_occurrence import align_quote_occurrences


REVIEW_VERSION = 1
QUOTE_TYPES = {
    "direct_speech",
    "partial_speech",
    "conveyed_speech",
    "thought_or_narration",
    "sound_or_text",
    "unclear",
}
EVIDENCE_BASES = {
    "explicit_attribution",
    "sequence_constraint",
    "speaker_action",
    "identity_alias",
    "quote_type",
    "inference",
    "unknown",
}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
SPEECH_ATTRIBUTION_RE = re.compile(
    r"(?:说(?:道|着|完|了)?|问(?:道)?|回答|答道|喊(?:道)?|开口|低语|嘀咕|"
    r"喃喃|笑(?:道|着说道)?|补充|回应|应声)"
)
HESITATION_END_RE = re.compile(r"(?:……|\.\.\.|——|--|呃|那个|名字).{0,3}$")
SELF_INTRO_PREFIXES = ("我叫", "我的名字是", "我名叫", "本人是", "咱叫", "在下是")
DEPARTURE_ACT_RE = re.compile(
    r"(?:先失陪|先告辞|告辞了|该走了|先走一步|就此别过|我们走吧|咱们走吧|该离开了)"
)
FAREWELL_REPLY_RE = re.compile(
    r"(?:再次相逢|一路顺风|慢走|再会|保重|后会有期|愿.{0,12}(?:神|平安|顺利|重逢))"
)
DEPARTURE_NARRATION_RE = re.compile(
    r"(?:离开|走出|告辞|转身.{0,8}走|起身.{0,12}走)"
)
REVIEW_AGENT_SPECS = (
    (
        "VolumeBlindForward",
        "Reconstruct the focus turns in chronological order from raw attribution, addressee/reply structure, "
        "and delayed attribution. You are blind to provisional labels.",
        False,
    ),
    (
        "VolumeBlindReverse",
        "Work backward from the last reliable attribution or action. Test every speaker switch and repair shifted "
        "or swapped consecutive turns. You are blind to provisional labels.",
        False,
    ),
    (
        "VolumeQuoteScope",
        "Audit the exact indexed quote occurrence. Bind pre-quote, post-quote, same-line and delayed narration to "
        "the correct quote; distinguish human speech from thought, text, sound and hypothetical wording. You are "
        "blind to provisional labels.",
        False,
    ),
    (
        "VolumeDialogueAct",
        "Map conversational acts rather than topic or style: question-answer, greeting-return, request-response, "
        "challenge-rebuttal, hesitation-information-supply, correction and interruption. A later self-introduction "
        "may answer another speaker who hesitated because they did not know the name.",
        False,
    ),
    (
        "VolumeTurnBoundary",
        "For every pair of adjacent quotes, test SAME SPEAKER and SPEAKER SWITCH as competing hypotheses. A quote "
        "that supplies information missing from a hesitant prior quote is often a response, not a continuation. "
        "Require evidence before merging turns merely because no narration separates them.",
        False,
    ),
    (
        "VolumeFloorHolder",
        "Track the conversational floor across quote-attribution-quote chains. Narration after a quote can both "
        "attribute that prior quote and establish who currently holds the floor; absent a switch cue, the next "
        "quote may continue that person. Then test whether a later quote is a response that supplies a name, "
        "correction or other missing information.",
        False,
    ),
    (
        "VolumeBaselineFalsifier",
        "Treat every provisional label as a fallible hypothesis. Search for the strongest raw-text contradiction, "
        "then keep or replace it. Prefer a verified personal identity over a temporary role when the evidence "
        "database proves they are the same person.",
        True,
    ),
)
JURY_SPECS = (
    (
        "VolumeJuryForward",
        "Decide from forward chronology. Require a raw-text explanation for every speaker switch.",
    ),
    (
        "VolumeJuryReverse",
        "Decide by reconstructing backward from reliable later attribution and actions.",
    ),
    (
        "VolumeJuryScopeSkeptic",
        "Audit exact quote scope and actively falsify the apparent majority before deciding.",
    ),
)

ASSIGNMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_volume_assignments",
        "description": "Submit one evidence-backed speaker assignment for every focus dialogue id.",
        "parameters": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "speaker": {"type": "string"},
                            "quote_type": {
                                "type": "string",
                                "enum": [
                                    "direct_speech",
                                    "partial_speech",
                                    "conveyed_speech",
                                    "thought_or_narration",
                                    "sound_or_text",
                                    "unclear",
                                ],
                            },
                            "evidence_basis": {
                                "type": "string",
                                "enum": [
                                    "explicit_attribution",
                                    "sequence_constraint",
                                    "speaker_action",
                                    "identity_alias",
                                    "quote_type",
                                    "inference",
                                    "unknown",
                                ],
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "citations": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "id",
                            "speaker",
                            "quote_type",
                            "evidence_basis",
                            "confidence",
                            "citations",
                            "reason",
                        ],
                    },
                }
            },
            "required": ["assignments"],
        },
    },
}

VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_volume_verdict",
        "description": "Submit the final evidence-backed speaker verdict for one exact quote occurrence.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "speaker": {"type": "string"},
                "quote_type": {"type": "string"},
                "evidence_basis": {"type": "string"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "reason": {"type": "string"},
            },
            "required": [
                "id",
                "speaker",
                "quote_type",
                "evidence_basis",
                "confidence",
                "citations",
                "reason",
            ],
        },
    },
}


@dataclass
class VolumeReviewRuntime:
    call_model: Callable[..., tuple[str, int, int, list[dict[str, Any]]]]
    parse_tool_arguments: Callable[[str, Any], tuple[dict[str, Any], str]]
    validate_speaker: Callable[[str], str | None]
    is_valid_speaker: Callable[[str], bool]
    same_speaker: Callable[[str, str], bool]
    canonicalize_speaker: Callable[[str], tuple[str, str]]
    request_timeout: int = 300
    retries: int = 2
    non_person_label: str = "非人物发声"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except FileNotFoundError:
        pass
    return records


def build_review_units(
    dialogue_count: int,
    *,
    focus_size: int = 8,
    context_radius: int = 5,
) -> list[dict[str, int | str]]:
    """Split a volume into non-overlapping focus ranges with overlapping context."""
    if dialogue_count < 0:
        raise ValueError("dialogue_count must be non-negative")
    focus_size = max(1, int(focus_size))
    context_radius = max(0, int(context_radius))
    units: list[dict[str, int | str]] = []
    for focus_start in range(0, dialogue_count, focus_size):
        focus_end = min(dialogue_count, focus_start + focus_size)
        units.append(
            {
                "unit_id": f"U{focus_start:06d}-{focus_end:06d}",
                "focus_start": focus_start,
                "focus_end": focus_end,
                "context_start": max(0, focus_start - context_radius),
                "context_end": min(dialogue_count, focus_end + context_radius),
            }
        )
    return units


def derive_deterministic_sequence_constraints(
    novel_lines: list[str],
    occurrences: list[dict[str, Any]],
    labels: list[str],
    *,
    is_valid_speaker: Callable[[str], bool],
    same_speaker: Callable[[str, str], bool],
) -> dict[str, dict[str, Any]]:
    """Find sparse, source-backed repair sequences without answer labels."""
    known_speakers = sorted(
        {
            str(label).strip()
            for label in labels
            if str(label).strip() and is_valid_speaker(str(label).strip())
        },
        key=len,
        reverse=True,
    )
    constraints: dict[str, dict[str, Any]] = {}
    for index in range(1, len(occurrences) - 1):
        previous = occurrences[index - 1]
        target = occurrences[index]
        following = occurrences[index + 1]
        target_text = str(target.get("text") or "").strip()
        if not HESITATION_END_RE.search(target_text):
            continue

        previous_line = int(previous["line"])
        target_line = int(target["line"])
        narrative_parts = [str(previous.get("scope_after") or "")]
        if target_line > previous_line + 1:
            narrative_parts.extend(
                str(novel_lines[line_num - 1]).rstrip("\n")
                for line_num in range(previous_line + 1, target_line)
            )
        narrative_parts.append(str(target.get("scope_before") or ""))
        narrative = "\n".join(part for part in narrative_parts if part.strip())
        if not narrative:
            continue

        attributed: list[tuple[int, str]] = []
        for speaker in known_speakers:
            for match in re.finditer(re.escape(speaker), narrative):
                tail = narrative[match.end():match.end() + 40]
                verb = SPEECH_ATTRIBUTION_RE.search(tail)
                if verb:
                    attributed.append((match.end() + verb.start(), speaker))
        if not attributed:
            continue
        attributed.sort(key=lambda item: item[0])
        floor_holder = attributed[-1][1]

        following_text = str(following.get("text") or "").strip()
        self_identified = [
            speaker
            for speaker in known_speakers
            if any(
                prefix + speaker in following_text[:50]
                for prefix in SELF_INTRO_PREFIXES
            )
        ]
        if not self_identified:
            continue
        if any(same_speaker(floor_holder, speaker) for speaker in self_identified):
            continue
        constraints[str(target["id"])] = {
            "speaker": floor_holder,
            "kind": "hesitation-before-other-self-introduction",
            "citations": sorted(
                {
                    previous_line,
                    target_line,
                    int(following["line"]),
                }
            ),
            "narrative": _clip(narrative, 500),
            "following_self_identification": self_identified[0],
        }

    for index in range(2, len(occurrences) - 1):
        previous = occurrences[index - 1]
        target = occurrences[index]
        following = occurrences[index + 1]
        if not DEPARTURE_ACT_RE.search(str(previous.get("text") or "")):
            continue
        if not FAREWELL_REPLY_RE.search(str(target.get("text") or "")):
            continue

        target_line = int(target["line"])
        following_line = int(following["line"])
        after_parts = [str(target.get("scope_after") or "")]
        if following_line > target_line + 1:
            after_parts.extend(
                str(novel_lines[line_num - 1]).rstrip("\n")
                for line_num in range(target_line + 1, following_line)
            )
        after_parts.append(str(following.get("scope_before") or ""))
        after_narration = "\n".join(part for part in after_parts if part.strip())
        if not DEPARTURE_NARRATION_RE.search(after_narration):
            continue

        counterpart = labels[index - 2]
        departing = labels[index - 1]
        if (
            not is_valid_speaker(counterpart)
            or not is_valid_speaker(departing)
            or same_speaker(counterpart, departing)
        ):
            continue
        constraints[str(target["id"])] = {
            "speaker": counterpart,
            "kind": "departure-and-farewell-reciprocity",
            "citations": sorted(
                {
                    int(occurrences[index - 2]["line"]),
                    int(previous["line"]),
                    target_line,
                    following_line,
                }
            ),
            "narrative": _clip(after_narration, 500),
            "departing_speaker": departing,
        }
    return constraints


def _clip(value: str, limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _json_values_from_text(value: str) -> list[Any]:
    """Extract strict JSON values from prose or fenced model output."""
    text = (value or "").strip()
    if not text:
        return []
    sources = [match.group(1).strip() for match in re.finditer(
        r"```(?:json)?\s*([\s\S]*?)```",
        text,
        flags=re.IGNORECASE,
    )]
    sources.append(text)
    decoder = json.JSONDecoder()
    values: list[Any] = []
    seen: set[str] = set()
    for source in sources:
        cursor = 0
        while cursor < len(source):
            match = re.search(r"[\[{]", source[cursor:])
            if match is None:
                break
            start = cursor + match.start()
            try:
                parsed, consumed = decoder.raw_decode(source[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            cursor = start + max(1, consumed)
            if not isinstance(parsed, (dict, list)):
                continue
            marker = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            values.append(parsed)
    return values


def _structured_payloads_from_text(value: str, expected_name: str) -> list[dict[str, Any]]:
    """Normalize common JSON/tool wrappers into validator-ready payloads."""
    payloads: list[dict[str, Any]] = []
    queue = list(_json_values_from_text(value))
    seen: set[str] = set()
    while queue:
        candidate = queue.pop(0)
        if isinstance(candidate, list):
            if (
                expected_name == "submit_volume_assignments"
                and candidate
                and all(isinstance(item, dict) and item.get("id") for item in candidate)
            ):
                candidate = {"assignments": candidate}
            else:
                continue
        if not isinstance(candidate, dict):
            continue
        marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        payloads.append(candidate)
        for key in (expected_name, "arguments", "result"):
            nested = candidate.get(key)
            if isinstance(nested, (dict, list)):
                queue.append(nested)
            elif isinstance(nested, str):
                queue.extend(_json_values_from_text(nested))
        function = candidate.get("function")
        if isinstance(function, dict):
            queue.append(function)
        tool_calls = candidate.get("tool_calls")
        if isinstance(tool_calls, list):
            queue.extend(tool_calls)
    return payloads


def _compact_single_target_task(user_content: str, dialogue_id: str) -> str:
    """Keep only nearby ledger/raw rows plus the target's candidate reports."""
    lines = (user_content or "").splitlines()
    target_number = int(dialogue_id[1:]) if dialogue_id[1:].isdigit() else None
    ledger_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "[Exact quote ledger]"),
        -1,
    )
    raw_start = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("[Raw novel excerpt")),
        -1,
    )
    tail_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() in {"[Single target]", "[Single recovery target]"}
        ),
        -1,
    )

    ledger_rows: list[str] = []
    target_line = None
    if ledger_start >= 0:
        ledger_end = raw_start if raw_start > ledger_start else len(lines)
        keep_block = False
        for line in lines[ledger_start + 1:ledger_end]:
            match = re.match(r"^(D(\d+))\s+\[", line)
            if match:
                number = int(match.group(2))
                keep_block = bool(target_number is None or abs(number - target_number) <= 3)
                if match.group(1) == dialogue_id:
                    line_match = re.search(r"\bL(\d+)\b", line)
                    if line_match:
                        target_line = int(line_match.group(1))
            if keep_block:
                ledger_rows.append(line)

    raw_rows: list[str] = []
    if raw_start >= 0 and target_line is not None:
        raw_end = tail_start if tail_start > raw_start else len(lines)
        for line in lines[raw_start + 1:raw_end]:
            match = re.match(r"^L(\d+):", line)
            if match and abs(int(match.group(1)) - target_line) <= 16:
                raw_rows.append(line)

    tail_rows = lines[tail_start:] if tail_start >= 0 else []
    sections = [
        f"[Compact target]\n{dialogue_id} at L{target_line or '?'}",
        "[Nearby exact quote ledger]\n" + ("\n".join(ledger_rows) or "(unavailable)"),
        "[Nearby raw novel excerpt]\n" + ("\n".join(raw_rows) or "(unavailable)"),
        "[Candidate and reviewer evidence]\n" + (_clip("\n".join(tail_rows), 12000) or "(none)"),
    ]
    return "\n\n".join(sections)


def _normalize_quote_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in QUOTE_TYPES:
        return text
    if text in {"dialogue", "speech", "spoken", "human_speech"}:
        return "direct_speech"
    if "partial" in text or "interrupted" in text:
        return "partial_speech"
    if "thought" in text or "narrat" in text:
        return "thought_or_narration"
    if "sound" in text or "text" in text:
        return "sound_or_text"
    return "unclear"


def _normalize_evidence_basis(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in EVIDENCE_BASES:
        return text
    if "attribut" in text or "speech verb" in text or "said" in text:
        return "explicit_attribution"
    if "identity" in text or "alias" in text or "name" in text:
        return "identity_alias"
    if "action" in text or "gesture" in text:
        return "speaker_action"
    if "quote" in text or "container" in text or "non-person" in text:
        return "quote_type"
    if "sequence" in text or "chronolog" in text or "dialogue" in text or "turn" in text:
        return "sequence_constraint"
    return "inference"


def _verified_identity_text(vault_data: dict[str, Any]) -> str:
    lines = ["[Verified identity evidence; generated from this volume, not source-code rules]"]
    for name, info in (vault_data.get("characters") or {}).items():
        if not isinstance(info, dict) or info.get("status") != "verified":
            continue
        aliases = [str(item) for item in info.get("aliases", []) if str(item).strip()]
        suffix = f" | aliases: {', '.join(aliases)}" if aliases else ""
        lines.append(f"  {name}{suffix}")
        for evidence in (info.get("evidence") or [])[:2]:
            if not isinstance(evidence, dict):
                continue
            lines.append(
                f"    L{evidence.get('line', '?')}: {_clip(str(evidence.get('text', '')), 120)}"
            )
    if len(lines) == 1:
        lines.append("  (none)")
    return "\n".join(lines)


def format_review_packet(
    novel_lines: list[str],
    occurrences: list[dict[str, Any]],
    labels: list[str],
    unit: dict[str, int | str],
    *,
    include_provisional: bool,
    raw_padding: int = 20,
    max_raw_chars: int = 26000,
) -> tuple[str, set[str], set[int]]:
    """Format a review packet that preserves exact quote occurrences and focus ids."""
    context_start = int(unit["context_start"])
    context_end = int(unit["context_end"])
    focus_start = int(unit["focus_start"])
    focus_end = int(unit["focus_end"])
    selected = occurrences[context_start:context_end]
    if not selected:
        raise ValueError(f"Review unit {unit['unit_id']} contains no dialogues")

    raw_start = max(1, int(selected[0]["line"]) - max(0, raw_padding))
    raw_end = min(len(novel_lines), int(selected[-1]["line"]) + max(0, raw_padding))
    raw_rows = [
        f"L{line_num}: {str(novel_lines[line_num - 1]).rstrip()}"
        for line_num in range(raw_start, raw_end + 1)
    ]
    raw_text = "\n".join(raw_rows)
    if len(raw_text) > max_raw_chars:
        focus_first_line = int(occurrences[focus_start]["line"])
        focus_last_line = int(occurrences[focus_end - 1]["line"])
        must_start = max(1, focus_first_line - 8)
        must_end = min(len(novel_lines), focus_last_line + 8)
        raw_rows = [
            f"L{line_num}: {str(novel_lines[line_num - 1]).rstrip()}"
            for line_num in range(must_start, must_end + 1)
        ]
        raw_text = "\n".join(raw_rows)
        raw_start, raw_end = must_start, must_end
        if len(raw_text) > max_raw_chars:
            raw_text = raw_text[:max_raw_chars] + "\n[raw excerpt clipped]"

    ledger_rows = ["[Exact quote ledger]"]
    focus_ids: set[str] = set()
    for occurrence in selected:
        index = int(occurrence["dialogue_index"])
        is_focus = focus_start <= index < focus_end
        if is_focus:
            focus_ids.add(str(occurrence["id"]))
        role = "FOCUS" if is_focus else "CONTEXT"
        ordinal = int(occurrence["same_line_ordinal"]) + 1
        provisional = (
            f" | provisional={labels[index]}" if include_provisional and index < len(labels) else ""
        )
        ledger_rows.append(
            f"{occurrence['id']} [{role}] L{occurrence['line']} quote#{ordinal}{provisional}: "
            f"「{_clip(str(occurrence['text']), 300)}」"
        )
        ledger_rows.append(
            f"  before={_clip(str(occurrence.get('scope_before') or '(none)'), 180)}"
        )
        ledger_rows.append(
            f"  after={_clip(str(occurrence.get('scope_after') or '(none)'), 180)}"
        )

    packet = (
        f"[Review unit {unit['unit_id']}]\n"
        f"Focus ids: {', '.join(sorted(focus_ids))}\n\n"
        + "\n".join(ledger_rows)
        + "\n\n[Raw novel excerpt]\n"
        + raw_text
    )
    return packet, focus_ids, set(range(raw_start, raw_end + 1))


class VolumeSecondPassReviewer:
    """Run full-volume joint review without touching labels until every task succeeds."""

    def __init__(
        self,
        runtime: VolumeReviewRuntime,
        *,
        novel_path: str,
        labeled_path: str,
        vault_path: str,
        review_log_path: str,
        temp_log_path: str,
        state_path: str,
        first_pass_path: str,
        focus_size: int = 8,
        context_radius: int = 5,
    ):
        self.runtime = runtime
        self.novel_path = novel_path
        self.labeled_path = labeled_path
        self.vault_path = vault_path
        self.review_log_path = review_log_path
        self.temp_log_path = temp_log_path
        self.state_path = state_path
        self.first_pass_path = first_pass_path
        self.focus_size = max(1, int(focus_size))
        self.context_radius = max(0, int(context_radius))
        self.model_calls = 0

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"volume-review-v{REVIEW_VERSION}".encode("ascii"))
        for path in (self.novel_path, self.labeled_path, self.vault_path):
            if os.path.exists(path):
                digest.update(_sha256_file(path).encode("ascii"))
        digest.update(str(self.focus_size).encode("ascii"))
        digest.update(str(self.context_radius).encode("ascii"))
        return digest.hexdigest()

    def _prepare_run(self, source_fingerprint: str, *, restart: bool) -> dict[str, Any]:
        state = _load_json(self.state_path)
        current_hash = _sha256_file(self.labeled_path)
        if (
            not restart
            and state.get("completed")
            and state.get("final_labeled_sha256") == current_hash
        ):
            return {"skip": True, "state": state}

        same_run = (
            not restart
            and not state.get("completed")
            and state.get("source_fingerprint") == source_fingerprint
        )
        if not same_run:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for path in (self.review_log_path, self.temp_log_path, self.state_path):
                if os.path.exists(path):
                    os.replace(path, f"{path}.stale-{stamp}")
            shutil.copy2(self.labeled_path, self.first_pass_path)
            state = {
                "version": REVIEW_VERSION,
                "completed": False,
                "source_fingerprint": source_fingerprint,
                "first_pass_sha256": current_hash,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "focus_size": self.focus_size,
                "context_radius": self.context_radius,
            }
            _write_json_atomic(self.state_path, state)
        elif not os.path.exists(self.first_pass_path):
            shutil.copy2(self.labeled_path, self.first_pass_path)
        return {"skip": False, "state": state}

    def _call_structured(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_content: str,
        tool: dict[str, Any],
        expected_name: str,
        validator: Callable[[dict[str, Any]], tuple[bool, Any]],
        record_meta: dict[str, Any],
        required_ids: set[str] | None = None,
        allowed_lines: set[int] | None = None,
    ) -> Any:
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        messages = list(base_messages)
        last_error = "missing required structured result"
        max_attempts = max(1, self.runtime.retries)
        can_split_assignments = bool(
            expected_name == "submit_volume_assignments"
            and required_ids
            and allowed_lines
        )
        recovery_dialogue_id = str(record_meta.get("dialogue_id") or "").strip()
        can_compact_verdict = bool(
            expected_name == "submit_volume_verdict" and recovery_dialogue_id
        )
        batch_attempts = 1 if (can_split_assignments or can_compact_verdict) else max_attempts

        def validate_payloads(
            candidates: list[tuple[dict[str, Any], str]],
        ) -> tuple[bool, Any, str]:
            candidate_error = "invalid structured result"
            for payload, note in candidates:
                valid, normalized = validator(payload)
                if valid:
                    return True, normalized, note
                errors = normalized.get("errors", []) if isinstance(normalized, dict) else []
                candidate_error = "; ".join(errors[:8]) or note or candidate_error
            return False, None, candidate_error

        for attempt in range(1, batch_attempts + 1):
            _append_jsonl(
                self.temp_log_path,
                {
                    "event": "review_call_start",
                    "agent": agent,
                    "attempt": attempt,
                    **record_meta,
                    "time": time.time(),
                },
            )
            try:
                text, pec, ec, tool_calls = self.runtime.call_model(
                    messages,
                    tools=[tool],
                    label=f"{agent}-A{attempt}",
                    request_timeout=self.runtime.request_timeout,
                    tool_choice={"type": "function", "function": {"name": expected_name}},
                    accept_text_when_tool_missing=True,
                )
            except Exception as exc:
                self.model_calls += 1
                last_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_call_failed",
                        "agent": agent,
                        "attempt": attempt,
                        "error": last_error,
                        **record_meta,
                        "time": time.time(),
                    },
                )
                if attempt >= max(1, self.runtime.retries):
                    raise
                continue

            self.model_calls += 1
            payload_candidates: list[tuple[dict[str, Any], str]] = []
            for tool_call in tool_calls or []:
                function = tool_call.get("function", {}) or {}
                if function.get("name") != expected_name:
                    continue
                parsed, parse_note = self.runtime.parse_tool_arguments(
                    expected_name,
                    function.get("arguments", "{}"),
                )
                if parsed:
                    payload_candidates.append((parsed, parse_note))
            payload_candidates.extend(
                (payload, "parsed JSON from response text")
                for payload in _structured_payloads_from_text(text, expected_name)
            )
            valid, normalized, candidate_error = validate_payloads(payload_candidates)
            if valid:
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_call_complete",
                        "agent": agent,
                        "attempt": attempt,
                        "prompt_tokens": pec,
                        "completion_tokens": ec,
                        "result": normalized,
                        "result_source": "primary",
                        **record_meta,
                        "time": time.time(),
                    },
                )
                return normalized
            last_error = candidate_error
            _append_jsonl(
                self.temp_log_path,
                {
                    "event": "review_call_rejected",
                    "agent": agent,
                    "attempt": attempt,
                    "error": last_error,
                    "text": _clip(text, 1000),
                    **record_meta,
                    "time": time.time(),
                },
            )

            if can_split_assignments or can_compact_verdict:
                break

            # SenseNova can produce a strong analysis while omitting or emptying
            # the requested tool call. Compile that reasoning through a separate
            # text-only JSON pass so review quality is not coupled to tool-call
            # transport reliability.
            schema = ((tool.get("function") or {}).get("parameters") or {})
            candidate_json = payload_candidates[-1][0] if payload_candidates else {}
            serializer_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are StructuredResultCompiler. Return exactly one JSON object and nothing else. "
                        "Do not use markdown and do not call tools. Preserve every person name and unnamed-role "
                        "label exactly in the language and script used by the source packet; never translate or "
                        "transliterate a speaker. Never invent a person. For narration, thought, displayed text, "
                        f"or non-human sound, use the exact speaker label {json.dumps(self.runtime.non_person_label, ensure_ascii=False)}. "
                        "Use the supplied reviewer analysis when available, but repair it against the original task "
                        "and validation error. Include every required id and every required field."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"[Expected result name]\n{expected_name}\n\n"
                        f"[JSON schema]\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                        f"[Validation error]\n{last_error}\n\n"
                        f"[Candidate structured payload]\n{json.dumps(candidate_json, ensure_ascii=False)}\n\n"
                        f"[Reviewer analysis]\n{_clip(text, 24000)}\n\n"
                        "Return the corrected JSON object now."
                    ),
                },
            ]
            _append_jsonl(
                self.temp_log_path,
                {
                    "event": "review_serializer_start",
                    "agent": agent,
                    "attempt": attempt,
                    "source_error": last_error,
                    **record_meta,
                    "time": time.time(),
                },
            )
            try:
                serializer_text, serializer_pec, serializer_ec, serializer_calls = self.runtime.call_model(
                    serializer_messages,
                    tools=None,
                    label=f"{agent}-Serializer-A{attempt}",
                    request_timeout=self.runtime.request_timeout,
                )
            except Exception as exc:
                self.model_calls += 1
                last_error = f"serializer {type(exc).__name__}: {str(exc)[:1000]}"
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_serializer_failed",
                        "agent": agent,
                        "attempt": attempt,
                        "error": last_error,
                        **record_meta,
                        "time": time.time(),
                    },
                )
            else:
                self.model_calls += 1
                serializer_candidates: list[tuple[dict[str, Any], str]] = []
                for tool_call in serializer_calls or []:
                    function = tool_call.get("function", {}) or {}
                    if function.get("name") != expected_name:
                        continue
                    parsed, parse_note = self.runtime.parse_tool_arguments(
                        expected_name,
                        function.get("arguments", "{}"),
                    )
                    if parsed:
                        serializer_candidates.append((parsed, parse_note))
                serializer_candidates.extend(
                    (payload, "compiled from text-only JSON")
                    for payload in _structured_payloads_from_text(serializer_text, expected_name)
                )
                serializer_valid, serializer_normalized, serializer_error = validate_payloads(
                    serializer_candidates
                )
                if serializer_valid:
                    _append_jsonl(
                        self.temp_log_path,
                        {
                            "event": "review_serializer_complete",
                            "agent": agent,
                            "attempt": attempt,
                            "prompt_tokens": serializer_pec,
                            "completion_tokens": serializer_ec,
                            "result": serializer_normalized,
                            **record_meta,
                            "time": time.time(),
                        },
                    )
                    _append_jsonl(
                        self.temp_log_path,
                        {
                            "event": "review_call_complete",
                            "agent": agent,
                            "attempt": attempt,
                            "prompt_tokens": pec + serializer_pec,
                            "completion_tokens": ec + serializer_ec,
                            "result": serializer_normalized,
                            "result_source": "text-json-serializer",
                            **record_meta,
                            "time": time.time(),
                        },
                    )
                    return serializer_normalized
                last_error = serializer_error
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_serializer_rejected",
                        "agent": agent,
                        "attempt": attempt,
                        "error": last_error,
                        "text": _clip(serializer_text, 1500),
                        **record_meta,
                        "time": time.time(),
                    },
                )

            messages = list(base_messages)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The prior result and its JSON compilation were rejected: {last_error}. "
                        f"Re-run the evidence audit, preserve source-language speaker labels exactly, and call "
                        f"{expected_name} with every required id and field."
                    ),
                }
            )
        if can_split_assignments:
            return self._recover_assignments_individually(
                agent=agent,
                system_prompt=system_prompt,
                user_content=user_content,
                validator=validator,
                record_meta=record_meta,
                required_ids=set(required_ids or set()),
                allowed_lines=set(allowed_lines or set()),
                batch_error=last_error,
            )
        if can_compact_verdict:
            return self._recover_verdict_compactly(
                agent=agent,
                system_prompt=system_prompt,
                user_content=user_content,
                validator=validator,
                record_meta=record_meta,
                dialogue_id=recovery_dialogue_id,
                original_error=last_error,
            )
        raise RuntimeError(
            f"{agent} failed to return a valid structured result after {batch_attempts} attempts: {last_error}"
        )

    def _recover_assignments_individually(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_content: str,
        validator: Callable[[dict[str, Any]], tuple[bool, Any]],
        record_meta: dict[str, Any],
        required_ids: set[str],
        allowed_lines: set[int],
        batch_error: str,
    ) -> Any:
        """Recover a failed batch as independent one-target verdicts."""
        ordered_ids = sorted(
            required_ids,
            key=lambda value: (
                0,
                int(value[1:]),
            ) if value[1:].isdigit() else (1, value),
        )
        perspective = system_prompt.split("Mandatory rules:", 1)[0].strip()
        _append_jsonl(
            self.temp_log_path,
            {
                "event": "review_batch_split_start",
                "agent": agent,
                "ids": ordered_ids,
                "batch_error": batch_error,
                **record_meta,
                "time": time.time(),
            },
        )
        assignments: list[dict[str, Any]] = []
        for dialogue_id in ordered_ids:
            single_agent = f"{agent}SingleTarget"
            single_system = f"""You are {single_agent}, recovering one item from a failed batch novel-speaker review.

Apply this reviewer perspective:
{perspective}

Analyze exactly the one target id named at the end of the user message. Ignore every other FOCUS id as an output target; they are context only. Raw attribution and exact quote scope outrank alternation, personality, topic, and style. Preserve every speaker name or unnamed-role label exactly in the source language and script; never translate or transliterate it. Use {self.runtime.non_person_label} only for narration, thought, displayed text, or non-human sound. Cite raw line numbers, never invent a person, and call submit_volume_verdict exactly once for the single target. Do not answer in prose.
"""
            single_user = (
                user_content
                + "\n\n[Single recovery target]\n"
                + f"{dialogue_id} only. Return exactly one verdict for {dialogue_id}; do not submit other ids."
            )
            single_meta = dict(record_meta)
            single_meta.update(
                {
                    "kind": "single-target-recovery",
                    "parent_agent": agent,
                    "dialogue_id": dialogue_id,
                }
            )
            verdict = self._call_structured(
                agent=single_agent,
                system_prompt=single_system,
                user_content=single_user,
                tool=VERDICT_TOOL,
                expected_name="submit_volume_verdict",
                validator=lambda args, did=dialogue_id: self._validate_verdict(
                    args,
                    did,
                    allowed_lines,
                ),
                record_meta=single_meta,
            )
            assignments.append(
                {
                    key: verdict[key]
                    for key in (
                        "id",
                        "speaker",
                        "quote_type",
                        "evidence_basis",
                        "confidence",
                        "citations",
                        "reason",
                    )
                }
            )

        recovered = {"assignments": assignments}
        valid, normalized = validator(recovered)
        if not valid:
            errors = normalized.get("errors", []) if isinstance(normalized, dict) else []
            raise RuntimeError(
                f"{agent} single-target recovery produced an invalid batch: "
                + ("; ".join(errors[:8]) or "invalid structured result")
            )
        _append_jsonl(
            self.temp_log_path,
            {
                "event": "review_batch_split_complete",
                "agent": agent,
                "ids": ordered_ids,
                "result": normalized,
                **record_meta,
                "time": time.time(),
            },
        )
        _append_jsonl(
            self.temp_log_path,
            {
                "event": "review_call_complete",
                "agent": agent,
                "attempt": 1,
                "result": normalized,
                "result_source": "single-target-recovery",
                **record_meta,
                "time": time.time(),
            },
        )
        return normalized

    def _recover_verdict_compactly(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_content: str,
        validator: Callable[[dict[str, Any]], tuple[bool, Any]],
        record_meta: dict[str, Any],
        dialogue_id: str,
        original_error: str,
    ) -> Any:
        """Retry one verdict with a compact packet that cannot consume output on unrelated turns."""
        compact_task = _compact_single_target_task(user_content, dialogue_id)
        perspective = system_prompt.split("The provisional label", 1)[0].strip()
        recovery_system = f"""You are {agent}CompactRecovery, deciding exactly one novel quote.

Reviewer perspective:
{perspective}

Use only the compact evidence packet. Do not restate or summarize it. Check exact quote scope, delayed attribution, same-speaker continuation, and reply structure. A non-colon attribution after a quote normally belongs to the preceding quote; a colon can introduce the next quote. Preserve names and unnamed-role labels exactly in the source language and script. Use {self.runtime.non_person_label} only for narration, thought, displayed text, or non-human sound. Call submit_volume_verdict as your first and only output with every required field and at least one raw-line citation. Do not answer in prose.
"""
        messages = [
            {"role": "system", "content": recovery_system},
            {"role": "user", "content": compact_task},
        ]
        last_error = original_error
        for attempt in range(1, max(1, self.runtime.retries) + 1):
            _append_jsonl(
                self.temp_log_path,
                {
                    "event": "review_compact_verdict_start",
                    "agent": agent,
                    "attempt": attempt,
                    "dialogue_id": dialogue_id,
                    "source_error": original_error,
                    **record_meta,
                    "time": time.time(),
                },
            )
            try:
                text, pec, ec, tool_calls = self.runtime.call_model(
                    messages,
                    tools=[VERDICT_TOOL],
                    label=f"{agent}-Compact-A{attempt}",
                    request_timeout=self.runtime.request_timeout,
                    tool_choice={
                        "type": "function",
                        "function": {"name": "submit_volume_verdict"},
                    },
                    accept_text_when_tool_missing=True,
                )
            except Exception as exc:
                self.model_calls += 1
                last_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_compact_verdict_failed",
                        "agent": agent,
                        "attempt": attempt,
                        "dialogue_id": dialogue_id,
                        "error": last_error,
                        **record_meta,
                        "time": time.time(),
                    },
                )
                continue

            self.model_calls += 1
            candidates: list[dict[str, Any]] = []
            for tool_call in tool_calls or []:
                function = tool_call.get("function", {}) or {}
                if function.get("name") != "submit_volume_verdict":
                    continue
                parsed, _ = self.runtime.parse_tool_arguments(
                    "submit_volume_verdict",
                    function.get("arguments", "{}"),
                )
                if parsed:
                    candidates.append(parsed)
            candidates.extend(
                _structured_payloads_from_text(text, "submit_volume_verdict")
            )
            normalized = None
            candidate_errors: list[str] = []
            for candidate in candidates:
                valid, checked = validator(candidate)
                if valid:
                    normalized = checked
                    break
                if isinstance(checked, dict):
                    candidate_errors.extend(checked.get("errors") or [])
            if normalized is not None:
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_compact_verdict_complete",
                        "agent": agent,
                        "attempt": attempt,
                        "dialogue_id": dialogue_id,
                        "prompt_tokens": pec,
                        "completion_tokens": ec,
                        "result": normalized,
                        **record_meta,
                        "time": time.time(),
                    },
                )
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_call_complete",
                        "agent": agent,
                        "attempt": attempt,
                        "result": normalized,
                        "result_source": "compact-single-verdict",
                        **record_meta,
                        "time": time.time(),
                    },
                )
                return normalized
            direct_error = (
                "; ".join(candidate_errors[:8])
                or "no parseable submit_volume_verdict payload"
            )
            _append_jsonl(
                self.temp_log_path,
                {
                    "event": "review_compact_verdict_rejected",
                    "agent": agent,
                    "attempt": attempt,
                    "dialogue_id": dialogue_id,
                    "error": direct_error,
                    "text": _clip(text, 1500),
                    **record_meta,
                    "time": time.time(),
                },
            )

            # Some OpenAI-compatible providers return a sound verdict as prose
            # even when a forced function call was requested.  The ordinary
            # structured-call path already has a JSON compiler for that case;
            # compact single-target recovery needs the same transport fallback
            # or one malformed response pattern can abort the whole volume.
            schema = ((VERDICT_TOOL.get("function") or {}).get("parameters") or {})
            serializer_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are CompactVerdictCompiler. Return exactly one JSON object and nothing else. "
                        "Do not use markdown and do not call tools. The object must follow the supplied schema. "
                        "Preserve the target id, source-language speaker label, and raw-line citations exactly. "
                        "Use the compact evidence packet and reviewer response only; never invent a person or citation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"[Target id]\n{dialogue_id}\n\n"
                        f"[JSON schema]\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                        f"[Validation error]\n{direct_error}\n\n"
                        f"[Compact evidence packet]\n{compact_task}\n\n"
                        f"[Reviewer response]\n{_clip(text, 12000)}\n\n"
                        "Compile the evidence-backed verdict as one JSON object now."
                    ),
                },
            ]
            _append_jsonl(
                self.temp_log_path,
                {
                    "event": "review_compact_serializer_start",
                    "agent": agent,
                    "attempt": attempt,
                    "dialogue_id": dialogue_id,
                    "source_error": direct_error,
                    **record_meta,
                    "time": time.time(),
                },
            )
            try:
                serializer_text, serializer_pec, serializer_ec, serializer_calls = (
                    self.runtime.call_model(
                        serializer_messages,
                        tools=None,
                        label=f"{agent}-CompactSerializer-A{attempt}",
                        request_timeout=self.runtime.request_timeout,
                    )
                )
            except Exception as exc:
                self.model_calls += 1
                last_error = f"compact serializer {type(exc).__name__}: {str(exc)[:1000]}"
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_compact_serializer_failed",
                        "agent": agent,
                        "attempt": attempt,
                        "dialogue_id": dialogue_id,
                        "error": last_error,
                        **record_meta,
                        "time": time.time(),
                    },
                )
            else:
                self.model_calls += 1
                serializer_candidates: list[dict[str, Any]] = []
                for tool_call in serializer_calls or []:
                    function = tool_call.get("function", {}) or {}
                    if function.get("name") != "submit_volume_verdict":
                        continue
                    parsed, _ = self.runtime.parse_tool_arguments(
                        "submit_volume_verdict",
                        function.get("arguments", "{}"),
                    )
                    if parsed:
                        serializer_candidates.append(parsed)
                serializer_candidates.extend(
                    _structured_payloads_from_text(
                        serializer_text,
                        "submit_volume_verdict",
                    )
                )
                serializer_errors: list[str] = []
                normalized = None
                for candidate in serializer_candidates:
                    valid, checked = validator(candidate)
                    if valid:
                        normalized = checked
                        break
                    if isinstance(checked, dict):
                        serializer_errors.extend(checked.get("errors") or [])
                if normalized is not None:
                    _append_jsonl(
                        self.temp_log_path,
                        {
                            "event": "review_compact_serializer_complete",
                            "agent": agent,
                            "attempt": attempt,
                            "dialogue_id": dialogue_id,
                            "prompt_tokens": serializer_pec,
                            "completion_tokens": serializer_ec,
                            "result": normalized,
                            **record_meta,
                            "time": time.time(),
                        },
                    )
                    _append_jsonl(
                        self.temp_log_path,
                        {
                            "event": "review_call_complete",
                            "agent": agent,
                            "attempt": attempt,
                            "prompt_tokens": pec + serializer_pec,
                            "completion_tokens": ec + serializer_ec,
                            "result": normalized,
                            "result_source": "compact-text-json-serializer",
                            **record_meta,
                            "time": time.time(),
                        },
                    )
                    return normalized
                last_error = (
                    "; ".join(serializer_errors[:8])
                    or "compact serializer returned no parseable verdict"
                )
                _append_jsonl(
                    self.temp_log_path,
                    {
                        "event": "review_compact_serializer_rejected",
                        "agent": agent,
                        "attempt": attempt,
                        "dialogue_id": dialogue_id,
                        "error": last_error,
                        "text": _clip(serializer_text, 1500),
                        **record_meta,
                        "time": time.time(),
                    },
                )
            messages = [
                {"role": "system", "content": recovery_system},
                {"role": "user", "content": compact_task},
                {
                    "role": "user",
                    "content": (
                        f"The prior verdict was rejected: {last_error}. Call submit_volume_verdict now; "
                        "do not write analysis."
                    ),
                },
            ]
        raise RuntimeError(
            f"{agent} compact recovery failed for {dialogue_id} after "
            f"{max(1, self.runtime.retries)} attempts: {last_error}"
        )

    def _validate_assignments(
        self,
        args: dict[str, Any],
        focus_ids: set[str],
        allowed_lines: set[int],
    ) -> tuple[bool, dict[str, Any]]:
        rows = args.get("assignments") if isinstance(args, dict) else None
        errors: list[str] = []
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        if not isinstance(rows, list):
            return False, {"errors": ["assignments must be an array"]}
        for row in rows:
            if not isinstance(row, dict):
                errors.append("assignment must be an object")
                continue
            dialogue_id = str(row.get("id") or "").strip()
            speaker = self.runtime.validate_speaker(str(row.get("speaker") or "")) or ""
            citations = []
            for value in row.get("citations") or []:
                try:
                    line_num = int(value)
                except (TypeError, ValueError):
                    continue
                if line_num in allowed_lines and line_num not in citations:
                    citations.append(line_num)
            if dialogue_id not in focus_ids:
                errors.append(f"unexpected id {dialogue_id!r}")
                continue
            if dialogue_id in seen:
                errors.append(f"duplicate id {dialogue_id}")
                continue
            if not speaker or not self.runtime.is_valid_speaker(speaker):
                errors.append(f"invalid speaker for {dialogue_id}")
                continue
            quote_type = _normalize_quote_type(row.get("quote_type"))
            evidence_basis = _normalize_evidence_basis(row.get("evidence_basis"))
            confidence = str(row.get("confidence") or "")
            if confidence not in CONFIDENCE_LEVELS:
                errors.append(f"invalid confidence for {dialogue_id}")
                continue
            seen.add(dialogue_id)
            normalized.append(
                {
                    "id": dialogue_id,
                    "speaker": speaker,
                    "quote_type": quote_type,
                    "evidence_basis": evidence_basis,
                    "confidence": confidence,
                    "citations": citations,
                    "reason": _clip(str(row.get("reason") or ""), 600),
                }
            )
        missing = sorted(focus_ids - seen)
        if missing:
            errors.append(f"missing ids: {', '.join(missing)}")
        return not errors, {"assignments": normalized, "errors": errors}

    def _validate_verdict(
        self,
        args: dict[str, Any],
        dialogue_id: str,
        allowed_lines: set[int],
    ) -> tuple[bool, dict[str, Any]]:
        if not isinstance(args, dict):
            return False, {"errors": ["verdict must be an object"]}
        speaker = self.runtime.validate_speaker(str(args.get("speaker") or "")) or ""
        citations = []
        for value in args.get("citations") or []:
            try:
                line_num = int(value)
            except (TypeError, ValueError):
                continue
            if line_num in allowed_lines and line_num not in citations:
                citations.append(line_num)
        errors = []
        if str(args.get("id") or "").strip() != dialogue_id:
            errors.append(f"verdict id must be {dialogue_id}")
        if not speaker or not self.runtime.is_valid_speaker(speaker):
            errors.append("invalid verdict speaker")
        if not citations:
            errors.append("at least one local citation is required")
        quote_type = _normalize_quote_type(args.get("quote_type"))
        evidence_basis = _normalize_evidence_basis(args.get("evidence_basis"))
        confidence = str(args.get("confidence") or "")
        if confidence not in CONFIDENCE_LEVELS:
            errors.append("invalid verdict confidence")
        return not errors, {
            "id": dialogue_id,
            "speaker": speaker,
            "quote_type": quote_type,
            "evidence_basis": evidence_basis,
            "confidence": confidence,
            "citations": citations,
            "reason": _clip(str(args.get("reason") or ""), 800),
            "errors": errors,
        }

    def _group_reports(self, reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for report in reports:
            speaker = report.get("speaker") or ""
            group = next(
                (
                    item
                    for item in groups
                    if self.runtime.same_speaker(str(item["speaker"]), str(speaker))
                ),
                None,
            )
            if group is None:
                groups.append({"speaker": speaker, "reports": [report]})
            else:
                group["reports"].append(report)
        groups.sort(key=lambda item: len(item["reports"]), reverse=True)
        return groups

    def _review_system_prompt(self, agent: str, instruction: str) -> str:
        return f"""You are {agent}, one independent reviewer in a generic novel dialogue-speaker system.

{instruction}

Mandatory rules:
1. Assign the source speaker of each exact FOCUS quote occurrence, not a nearby actor or addressee.
2. Raw grammatical attribution and quote container outrank alternation, personality, topic and writing style.
3. A non-colon attribution after one quote normally belongs to that preceding quote. A colon can introduce the next quote.
4. Consecutive quotes may continue the same speaker; never force alternation.
5. Conversely, do not merge adjacent quotes automatically. Hesitation followed by another quote that supplies a name or missing fact is a common cross-speaker repair sequence.
6. Use the dynamically supplied verified identity evidence. If a named identity is proved, return that identity rather than a temporary role.
7. Use the exact non-person label {self.runtime.non_person_label} only for narration, thought, displayed text or non-human sound, never merely because a human is unnamed.
8. Do not invent people. Cite raw line numbers from the packet for every assignment.
9. Preserve person names and unnamed-role labels exactly in the source language and script; never translate or transliterate them.
10. Call submit_volume_assignments exactly once with every FOCUS id. Do not answer in prose.
"""

    def _jury_system_prompt(self, agent: str, instruction: str) -> str:
        return f"""You are {agent}, an independent final juror for one exact novel quote.

{instruction}

The provisional label and reviewer reports are fallible evidence, not votes with authority. Reconstruct the raw scene yourself. Reject personality/style guesses and mechanical alternation. Check whether post-quote narration was already consumed by the preceding quote and whether a later attribution covers a block. Explicitly test whether an adjacent quote continues the same speaker or answers the prior speaker; hesitation followed by a self-introduction or missing-information supply is often a cross-speaker repair. Use dynamically supplied verified identities; never invent a person. Preserve every speaker label exactly in the source language and script; never translate or transliterate it. Use {self.runtime.non_person_label} for narration, thought, displayed text, or non-human sound. Return one speaker only through submit_volume_verdict with at least one raw-line citation.
"""

    def run(self, dialogue_list: list[tuple[int, str]], *, restart: bool = False) -> dict[str, Any]:
        source_fingerprint = self._fingerprint()
        prepared = self._prepare_run(source_fingerprint, restart=restart)
        if prepared["skip"]:
            return {
                "status": "already-complete",
                "model_calls": 0,
                "summary": prepared["state"].get("summary", {}),
            }

        novel_lines = Path(self.novel_path).read_text(encoding="utf-8").splitlines()
        labels = [
            line.strip()
            for line in Path(self.first_pass_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(labels) != len(dialogue_list):
            raise ValueError(
                f"Full-volume review requires complete labels: {len(labels)} != {len(dialogue_list)}"
            )
        occurrences = align_quote_occurrences(novel_lines, dialogue_list)
        vault_data = _load_json(self.vault_path)
        identity_text = _verified_identity_text(vault_data)
        units = build_review_units(
            len(dialogue_list),
            focus_size=self.focus_size,
            context_radius=self.context_radius,
        )

        existing = _load_jsonl(self.review_log_path)
        unit_results: dict[tuple[str, str], dict[str, Any]] = {}
        jury_results: dict[tuple[str, str], dict[str, Any]] = {}
        for record in existing:
            if record.get("kind") == "unit" and record.get("status") == "complete":
                unit_results[(str(record.get("unit_id")), str(record.get("agent")))] = record
            elif record.get("kind") == "jury" and record.get("status") == "complete":
                jury_results[(str(record.get("dialogue_id")), str(record.get("agent")))] = record

        assignments_by_id: dict[str, list[dict[str, Any]]] = {
            str(occurrence["id"]): [] for occurrence in occurrences
        }
        for unit_number, unit in enumerate(units, start=1):
            for agent, instruction, include_provisional in REVIEW_AGENT_SPECS:
                key = (str(unit["unit_id"]), agent)
                record = unit_results.get(key)
                if record is None:
                    packet, focus_ids, allowed_lines = format_review_packet(
                        novel_lines,
                        occurrences,
                        labels,
                        unit,
                        include_provisional=include_provisional,
                    )
                    result = self._call_structured(
                        agent=agent,
                        system_prompt=self._review_system_prompt(agent, instruction),
                        user_content=identity_text + "\n\n" + packet,
                        tool=ASSIGNMENT_TOOL,
                        expected_name="submit_volume_assignments",
                        validator=lambda args, ids=focus_ids, lines=allowed_lines: self._validate_assignments(
                            args, ids, lines
                        ),
                        record_meta={"kind": "unit", "unit_id": unit["unit_id"]},
                        required_ids=focus_ids,
                        allowed_lines=allowed_lines,
                    )
                    record = {
                        "kind": "unit",
                        "status": "complete",
                        "unit_id": unit["unit_id"],
                        "unit_number": unit_number,
                        "unit_count": len(units),
                        "agent": agent,
                        "assignments": result["assignments"],
                        "time": time.time(),
                    }
                    _append_jsonl(self.review_log_path, record)
                    unit_results[key] = record
                for assignment in record.get("assignments") or []:
                    item = dict(assignment)
                    item["agent"] = agent
                    assignments_by_id.setdefault(str(item.get("id")), []).append(item)

        deterministic_constraints = derive_deterministic_sequence_constraints(
            novel_lines,
            occurrences,
            labels,
            is_valid_speaker=self.runtime.is_valid_speaker,
            same_speaker=self.runtime.same_speaker,
        )

        decisions: list[dict[str, Any]] = []
        reviewed_labels = list(labels)
        for occurrence in occurrences:
            index = int(occurrence["dialogue_index"])
            dialogue_id = str(occurrence["id"])
            baseline = labels[index]
            reports = assignments_by_id.get(dialogue_id, [])
            groups = self._group_reports(reports)
            deterministic = deterministic_constraints.get(dialogue_id)
            if deterministic and not self.runtime.same_speaker(
                str(deterministic["speaker"]), baseline
            ):
                selected, canonical_note = self.runtime.canonicalize_speaker(
                    str(deterministic["speaker"])
                )
                reviewed_labels[index] = selected
                decisions.append(
                    {
                        "id": dialogue_id,
                        "baseline": baseline,
                        "speaker": selected,
                        "changed": True,
                        "source": "deterministic-conversation-repair",
                        "canonicalization": canonical_note,
                        "deterministic_constraint": deterministic,
                        "reviewer_groups": [
                            {"speaker": group["speaker"], "count": len(group["reports"])}
                            for group in groups
                        ],
                    }
                )
                continue
            alternative_group = next(
                (
                    group
                    for group in groups
                    if not self.runtime.same_speaker(str(group["speaker"]), baseline)
                    and len(group["reports"]) >= 2
                    and any(
                        report.get("confidence") in {"high", "medium"}
                        and report.get("citations")
                        and report.get("evidence_basis") not in {"unknown", "inference"}
                        for report in group["reports"]
                    )
                    and any(
                        report.get("agent") != "VolumeBaselineFalsifier"
                        for report in group["reports"]
                    )
                ),
                None,
            )
            if alternative_group is None:
                decisions.append(
                    {
                        "id": dialogue_id,
                        "baseline": baseline,
                        "speaker": baseline,
                        "changed": False,
                        "source": "no-supported-alternative",
                        "reviewer_groups": [
                            {"speaker": group["speaker"], "count": len(group["reports"])}
                            for group in groups
                        ],
                    }
                )
                continue

            candidate = str(alternative_group["speaker"])
            unit = next(
                item
                for item in units
                if int(item["focus_start"]) <= index < int(item["focus_end"])
            )
            packet, _, allowed_lines = format_review_packet(
                novel_lines,
                occurrences,
                labels,
                unit,
                include_provisional=True,
            )
            candidate_rows = []
            for report in reports:
                candidate_rows.append(
                    f"- {report.get('agent')}: speaker={report.get('speaker')}; "
                    f"basis={report.get('evidence_basis')}; confidence={report.get('confidence')}; "
                    f"citations={report.get('citations')}; reason={report.get('reason')}"
                )
            jury_prompt = (
                identity_text
                + "\n\n"
                + packet
                + f"\n\n[Single target]\n{dialogue_id}; provisional={baseline}; "
                + f"supported alternative={candidate}\n[Independent reviewer reports]\n"
                + "\n".join(candidate_rows)
            )
            verdicts: list[dict[str, Any]] = []
            for jury_agent, jury_instruction in JURY_SPECS:
                key = (dialogue_id, jury_agent)
                record = jury_results.get(key)
                if record is None:
                    verdict = self._call_structured(
                        agent=jury_agent,
                        system_prompt=self._jury_system_prompt(jury_agent, jury_instruction),
                        user_content=jury_prompt,
                        tool=VERDICT_TOOL,
                        expected_name="submit_volume_verdict",
                        validator=lambda args, did=dialogue_id, lines=allowed_lines: self._validate_verdict(
                            args, did, lines
                        ),
                        record_meta={"kind": "jury", "dialogue_id": dialogue_id},
                    )
                    record = {
                        "kind": "jury",
                        "status": "complete",
                        "dialogue_id": dialogue_id,
                        "agent": jury_agent,
                        "verdict": verdict,
                        "time": time.time(),
                    }
                    _append_jsonl(self.review_log_path, record)
                    jury_results[key] = record
                verdicts.append(dict(record.get("verdict") or {}))

            jury_groups = self._group_reports(verdicts)
            unanimous = jury_groups[0] if jury_groups and len(jury_groups[0]["reports"]) == len(JURY_SPECS) else None
            jury_evidence_is_strong = bool(
                unanimous
                and all(
                    report.get("confidence") in {"high", "medium"}
                    and report.get("citations")
                    and report.get("evidence_basis") not in {"unknown", "inference"}
                    for report in unanimous["reports"]
                )
            )
            supported = bool(
                unanimous
                and jury_evidence_is_strong
                and any(
                    self.runtime.same_speaker(
                        str(unanimous["speaker"]), str(group["speaker"])
                    )
                    and len(group["reports"]) >= 2
                    for group in groups
                )
            )
            changed = bool(
                supported
                and not self.runtime.same_speaker(str(unanimous["speaker"]), baseline)
            )
            selected = str(unanimous["speaker"]) if changed else baseline
            if changed:
                selected, canonical_note = self.runtime.canonicalize_speaker(selected)
                reviewed_labels[index] = selected
            else:
                canonical_note = ""
            decisions.append(
                {
                    "id": dialogue_id,
                    "baseline": baseline,
                    "speaker": selected,
                    "changed": changed,
                    "source": "unanimous-independent-volume-jury" if changed else "jury-retained-baseline",
                    "canonicalization": canonical_note,
                    "reviewer_groups": [
                        {"speaker": group["speaker"], "count": len(group["reports"])}
                        for group in groups
                    ],
                    "jury_groups": [
                        {"speaker": group["speaker"], "count": len(group["reports"])}
                        for group in jury_groups
                    ],
                    "jury_evidence_is_strong": jury_evidence_is_strong,
                }
            )

        output_path = Path(self.labeled_path).with_suffix(".reviewed.tmp")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(reviewed_labels) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if len(reviewed_labels) != len(dialogue_list):
            raise RuntimeError("Reviewed label count changed unexpectedly")
        os.replace(output_path, self.labeled_path)

        changed_count = sum(bool(item.get("changed")) for item in decisions)
        summary = {
            "dialogues": len(dialogue_list),
            "units": len(units),
            "review_agents": len(REVIEW_AGENT_SPECS),
            "jury_agents": len(JURY_SPECS),
            "changed": changed_count,
            "retained": len(dialogue_list) - changed_count,
            "model_calls_this_process": self.model_calls,
            "selection_sources": dict(Counter(item["source"] for item in decisions)),
        }
        _append_jsonl(
            self.review_log_path,
            {
                "kind": "summary",
                "status": "complete",
                "summary": summary,
                "decisions": decisions,
                "time": time.time(),
            },
        )
        state = _load_json(self.state_path)
        state.update(
            {
                "completed": True,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "final_labeled_sha256": _sha256_file(self.labeled_path),
                "summary": summary,
            }
        )
        _write_json_atomic(self.state_path, state)
        return {"status": "complete", "model_calls": self.model_calls, "summary": summary}
