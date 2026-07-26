"""Novel-agnostic scene segmentation and complete speaker-sequence helpers.

This module never infers a speaker. It creates stable scene blocks from raw
text and validates structured scene maps returned by external reasoners.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable


_HEADING_RE = re.compile(
    r"^(?:第.{1,12}[章节幕卷部]|序章|终章|尾声|后记|前言|楔子|插图|幕间)(?:\s.*)?$"
)
_TIME_JUMP_RE = re.compile(
    r"(?:翌日|次日|第二天|几天后|数日后|多年后|数月后|当晚|隔天|与此同时)"
)


def build_scene_segments(
    novel_lines: Iterable[str],
    dialogue_list: list[tuple[int, str]],
    *,
    max_turns: int = 8,
    max_raw_span: int = 180,
    hard_gap_lines: int = 16,
    raw_padding: int = 6,
) -> list[dict[str, Any]]:
    """Split all dialogue occurrences into stable, bounded scene segments.

    Splits first occur at strong structural boundaries. Oversized chunks are
    then divided at the strongest available narrative boundary before the hard
    turn/span limit. The result is independent of the eventual target turn.
    """
    raw_lines = [line.rstrip("\n") for line in novel_lines]
    if not dialogue_list:
        return []

    max_turns = max(4, int(max_turns))
    max_raw_span = max(30, int(max_raw_span))
    hard_gap_lines = max(4, int(hard_gap_lines))
    raw_padding = max(0, int(raw_padding))

    boundaries = [
        _boundary_info(raw_lines, dialogue_list[index], dialogue_list[index + 1], hard_gap_lines)
        for index in range(len(dialogue_list) - 1)
    ]

    strong_chunks: list[tuple[int, int]] = []
    chunk_start = 0
    for boundary_index, boundary in enumerate(boundaries):
        if boundary["strong_break"]:
            strong_chunks.append((chunk_start, boundary_index + 1))
            chunk_start = boundary_index + 1
    strong_chunks.append((chunk_start, len(dialogue_list)))

    bounded_chunks: list[tuple[int, int]] = []
    for start, end in strong_chunks:
        bounded_chunks.extend(
            _split_oversized_chunk(
                dialogue_list,
                boundaries,
                start,
                end,
                max_turns=max_turns,
                max_raw_span=max_raw_span,
            )
        )

    segments = []
    for ordinal, (start, end) in enumerate(bounded_chunks, 1):
        first_line = dialogue_list[start][0]
        last_line = dialogue_list[end - 1][0]
        raw_start = max(1, first_line - raw_padding)
        raw_end = min(len(raw_lines), last_line + raw_padding)
        segments.append(
            {
                "scene_id": f"S{ordinal:04d}-D{start + 1}-{end}",
                "start_index": start,
                "end_index": end,
                "turn_count": end - start,
                "first_dialogue_line": first_line,
                "last_dialogue_line": last_line,
                "raw_start": raw_start,
                "raw_end": raw_end,
            }
        )
    return segments


def build_scene_packet(
    novel_lines: Iterable[str],
    dialogue_list: list[tuple[int, str]],
    segment: dict[str, Any],
) -> dict[str, Any]:
    """Build one raw scene packet with stable turn ids and boundaries."""
    raw_lines = [line.rstrip("\n") for line in novel_lines]
    start = int(segment["start_index"])
    end = int(segment["end_index"])
    selected = dialogue_list[start:end]
    if not selected:
        raise ValueError("Scene segment contains no dialogue turns")

    turns = []
    for offset, (line_num, text) in enumerate(selected):
        turns.append(
            {
                "id": f"D{offset}",
                "dialogue_index": start + offset,
                "line": line_num,
                "text": text,
                "same_line_ordinal": _same_line_ordinal(dialogue_list, start + offset),
            }
        )

    boundaries = []
    for previous, current in zip(turns, turns[1:]):
        narrative = _narrative_between(raw_lines, previous["line"], current["line"])
        boundaries.append(
            {
                "from": previous["id"],
                "to": current["id"],
                "narrative_lines": narrative,
                "has_narrative_break": bool(narrative),
            }
        )

    raw_start = max(1, int(segment["raw_start"]))
    raw_end = min(len(raw_lines), int(segment["raw_end"]))
    packet = {
        "scene_id": segment["scene_id"],
        "start_index": start,
        "end_index": end,
        "raw_start": raw_start,
        "raw_end": raw_end,
        "turns": turns,
        "boundaries": boundaries,
        "raw_lines": [
            {"line": line_num, "text": raw_lines[line_num - 1]}
            for line_num in range(raw_start, raw_end + 1)
        ],
    }
    return packet


def find_scene_for_dialogue(
    segments: list[dict[str, Any]],
    dialogue_index: int,
) -> dict[str, Any]:
    """Return the stable scene segment containing an absolute dialogue index."""
    for segment in segments:
        if segment["start_index"] <= dialogue_index < segment["end_index"]:
            return segment
    raise ValueError(f"No scene segment contains dialogue index {dialogue_index}")


def format_scene_packet(
    packet: dict[str, Any],
    *,
    target_dialogue_index: int | None = None,
    max_raw_chars: int = 24000,
) -> str:
    """Render a complete scene without prior speaker labels or vote counts."""
    lines = [
        "[Complete raw scene packet - no prior speaker labels]",
        f"Scene: {packet['scene_id']}",
        f"Dialogue turns: {len(packet['turns'])}",
        "",
        "[All dialogue turns in chronological order]",
    ]
    target_turn_id = ""
    for turn in packet["turns"]:
        marker = ""
        if target_dialogue_index is not None and turn["dialogue_index"] == target_dialogue_index:
            marker = " TARGET"
            target_turn_id = turn["id"]
        lines.append(
            f"  {turn['id']} L{turn['line']} occurrence#{turn['same_line_ordinal']}{marker}: {turn['text']}"
        )

    if target_dialogue_index is not None:
        lines.insert(3, f"Target turn: {target_turn_id or 'not found'}")

    lines.extend(["", "[Narrative boundaries]"])
    if not packet["boundaries"]:
        lines.append("  (single-turn scene)")
    for boundary in packet["boundaries"]:
        narrative = boundary.get("narrative_lines") or []
        if narrative:
            excerpt = " | ".join(
                f"L{item['line']}: {item['text'][:160]}" for item in narrative[:8]
            )
            lines.append(f"  {boundary['from']} -> {boundary['to']}: {excerpt}")
        else:
            lines.append(
                f"  {boundary['from']} -> {boundary['to']}: no non-dialogue narrative line"
            )

    lines.extend(["", "[Exact original scene lines]"])
    raw_lines = []
    target_line = None
    if target_dialogue_index is not None:
        for turn in packet["turns"]:
            if turn["dialogue_index"] == target_dialogue_index:
                target_line = turn["line"]
                break
    for item in packet["raw_lines"]:
        marker = ">>" if item["line"] == target_line else "  "
        raw_lines.append(f"{marker} L{item['line']}: {item['text']}")
    lines.append(_target_preserving_compact(raw_lines, target_line, max_raw_chars))
    return "\n".join(lines)


def normalize_scene_assignments(
    raw_assignments: Any,
    packet: dict[str, Any],
    validate_speaker: Callable[[str], str | None],
) -> dict[str, Any]:
    """Validate a complete structured scene map against its supplied packet."""
    if not isinstance(raw_assignments, list):
        return {"valid": False, "assignments": {}, "missing": [], "errors": ["assignments is not a list"]}

    expected = {turn["id"]: turn for turn in packet["turns"]}
    available_lines = {item["line"] for item in packet["raw_lines"]}
    normalized: dict[str, dict[str, Any]] = {}
    errors = []

    for position, item in enumerate(raw_assignments):
        if not isinstance(item, dict):
            errors.append(f"assignment#{position + 1} is not an object")
            continue
        turn_id = str(item.get("turn_id") or item.get("id") or "").strip()
        if turn_id not in expected:
            errors.append(f"unknown turn_id {turn_id!r}")
            continue
        if turn_id in normalized:
            errors.append(f"duplicate turn_id {turn_id}")
            continue

        turn = expected[turn_id]
        try:
            output_line = int(item.get("line"))
        except (TypeError, ValueError):
            output_line = -1
        if output_line != turn["line"]:
            errors.append(f"{turn_id} line mismatch: {output_line} != {turn['line']}")
            continue

        speaker = validate_speaker(str(item.get("speaker") or "").strip())
        if not speaker:
            errors.append(f"{turn_id} has invalid speaker")
            continue

        citations = []
        for value in item.get("citations") or []:
            try:
                line_num = int(value)
            except (TypeError, ValueError):
                continue
            if line_num not in citations:
                citations.append(line_num)
        if not citations or not any(line_num in available_lines for line_num in citations):
            errors.append(f"{turn_id} has no scene-local citation")
            continue

        normalized[turn_id] = {
            "turn_id": turn_id,
            "line": turn["line"],
            "speaker": speaker,
            "quote_type": str(item.get("quote_type") or "unclear").strip().lower(),
            "evidence_basis": str(item.get("evidence_basis") or "unknown").strip().lower(),
            "confidence": str(item.get("confidence") or "low").strip().lower(),
            "citations": citations,
            "reason": str(item.get("reason") or "").strip()[:1200],
        }

    missing = [turn_id for turn_id in expected if turn_id not in normalized]
    if missing:
        errors.append("missing turns: " + ", ".join(missing))
    return {
        "valid": not errors and len(normalized) == len(expected),
        "assignments": normalized,
        "missing": missing,
        "errors": errors,
    }


def assignment_for_dialogue(
    scene_map: dict[str, Any],
    packet: dict[str, Any],
    dialogue_index: int,
) -> dict[str, Any] | None:
    """Return one normalized assignment by absolute dialogue index."""
    for turn in packet["turns"]:
        if turn["dialogue_index"] == dialogue_index:
            return (scene_map.get("assignments") or {}).get(turn["id"])
    return None


def _boundary_info(
    raw_lines: list[str],
    previous: tuple[int, str],
    current: tuple[int, str],
    hard_gap_lines: int,
) -> dict[str, Any]:
    previous_line, _ = previous
    current_line, _ = current
    narrative = _narrative_between(raw_lines, previous_line, current_line)
    narrative_text = " ".join(item["text"] for item in narrative)
    has_heading = any(_looks_like_heading(item["text"]) for item in narrative)
    gap = max(0, current_line - previous_line)
    score = len(narrative)
    if has_heading:
        score += 100
    if gap >= hard_gap_lines:
        score += 40
    if len(narrative) >= 8:
        score += 20
    if _TIME_JUMP_RE.search(narrative_text):
        score += 12
    strong_break = bool(has_heading or gap >= hard_gap_lines or len(narrative) >= 10)
    return {
        "gap": gap,
        "narrative_count": len(narrative),
        "score": score,
        "strong_break": strong_break,
    }


def _split_oversized_chunk(
    dialogue_list: list[tuple[int, str]],
    boundaries: list[dict[str, Any]],
    start: int,
    end: int,
    *,
    max_turns: int,
    max_raw_span: int,
) -> list[tuple[int, int]]:
    result = []
    cursor = start
    while cursor < end:
        candidate_end = min(end, cursor + max_turns)
        while (
            candidate_end > cursor + 1
            and dialogue_list[candidate_end - 1][0] - dialogue_list[cursor][0] > max_raw_span
        ):
            candidate_end -= 1
        if candidate_end >= end:
            result.append((cursor, end))
            break
        if candidate_end <= cursor:
            candidate_end = min(end, cursor + 1)

        minimum_split = min(candidate_end - 1, cursor + max(2, max_turns // 3))
        choices = range(minimum_split, candidate_end)
        split_at = max(
            choices,
            key=lambda boundary_index: (
                boundaries[boundary_index]["score"],
                boundary_index,
            ),
            default=candidate_end - 1,
        ) + 1
        if split_at <= cursor:
            split_at = candidate_end
        result.append((cursor, split_at))
        cursor = split_at
    return result


def _narrative_between(
    raw_lines: list[str],
    previous_line: int,
    current_line: int,
) -> list[dict[str, Any]]:
    if current_line <= previous_line:
        return []
    narrative = []
    for line_num in range(previous_line + 1, min(current_line, len(raw_lines) + 1)):
        text = raw_lines[line_num - 1].strip()
        if text and "「" not in text:
            narrative.append({"line": line_num, "text": text})
    return narrative


def _looks_like_heading(text: str) -> bool:
    cleaned = (text or "").strip(" \t\r\n　")
    return bool(cleaned and len(cleaned) <= 32 and _HEADING_RE.match(cleaned))


def _same_line_ordinal(dialogue_list: list[tuple[int, str]], target_index: int) -> int:
    target_line = dialogue_list[target_index][0]
    return 1 + sum(1 for line, _ in dialogue_list[:target_index] if line == target_line)


def _target_preserving_compact(
    raw_lines: list[str],
    target_line: int | None,
    max_chars: int,
) -> str:
    text = "\n".join(raw_lines)
    if len(text) <= max_chars:
        return text
    max_chars = max(1000, int(max_chars))
    target_index = 0
    if target_line is not None:
        marker = f"L{target_line}:"
        target_index = next((index for index, line in enumerate(raw_lines) if marker in line), 0)
    focus_start = max(0, target_index - 8)
    focus_end = min(len(raw_lines), target_index + 9)
    focus = "\n".join(raw_lines[focus_start:focus_end])
    remaining = max(0, max_chars - len(focus) - 80)
    before = "\n".join(raw_lines[:focus_start])
    after = "\n".join(raw_lines[focus_end:])
    before = before[-(remaining // 2):]
    after = after[: remaining - len(before)]
    return "\n".join(
        part for part in (
            before,
            "... [target-centered scene excerpt] ...",
            focus,
            "... [remaining scene excerpt] ...",
            after,
        ) if part
    )
