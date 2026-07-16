"""Novel-agnostic helpers for block-level dialogue attribution.

The helpers in this module never infer a speaker. They only preserve the raw
text, dialogue order, quote location, and line-number boundaries that separate
LLM judgments from one another.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_TAG_RE_TEMPLATE = r"<{tag}>(.*?)</{tag}>"
_LINE_CITATION_RE = re.compile(r"\bL\s*(\d+)\b", re.IGNORECASE)


def _compact_text(text: str, max_chars: int, keep: str) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    marker = "\n... [truncated] ...\n"
    keep_chars = max(0, max_chars - len(marker))
    if keep == "head":
        return text[:keep_chars] + marker
    return marker + text[-keep_chars:]


def get_tag(text: str, tag: str, default: str = "") -> str:
    """Return a trimmed XML-like tag value without interpreting its content."""
    match = re.search(_TAG_RE_TEMPLATE.format(tag=re.escape(tag)), text or "", re.DOTALL)
    return match.group(1).strip() if match else default


def parse_line_citations(value: str) -> list[int]:
    """Extract explicit L123-style source references from an agent response."""
    seen: set[int] = set()
    citations: list[int] = []
    for match in _LINE_CITATION_RE.finditer(value or ""):
        line_num = int(match.group(1))
        if line_num not in seen:
            seen.add(line_num)
            citations.append(line_num)
    return citations


def build_dialogue_packet(
    novel_lines: Iterable[str],
    dialogue_list: list[tuple[int, str]],
    line_num: int,
    dialogue: str,
    dialogue_radius: int = 4,
    raw_padding: int = 5,
) -> dict[str, Any]:
    """Build a bounded raw-text packet around one dialogue occurrence.

    The packet includes neighboring dialogue turns and the exact original lines
    around them. It deliberately contains no prior speaker predictions, aliases,
    or novel-specific vocabulary.
    """
    raw_lines = [line.rstrip("\n") for line in novel_lines]
    target_index = _find_dialogue_index(dialogue_list, line_num, dialogue)
    if target_index is None:
        raise ValueError(f"Dialogue not found in extracted dialogue list: L{line_num}")

    radius = max(1, int(dialogue_radius))
    start_idx = max(0, target_index - radius)
    end_idx = min(len(dialogue_list), target_index + radius + 1)
    selected = dialogue_list[start_idx:end_idx]

    raw_start = max(1, selected[0][0] - max(0, int(raw_padding)))
    raw_end = min(len(raw_lines), selected[-1][0] + max(0, int(raw_padding)))
    same_line_ordinal = _same_line_ordinal(dialogue_list, target_index)

    turns = []
    for offset, (turn_line, turn_text) in enumerate(selected):
        absolute_index = start_idx + offset
        turns.append(
            {
                "id": f"D{offset}",
                "dialogue_index": absolute_index,
                "line": turn_line,
                "text": turn_text,
                "is_target": absolute_index == target_index,
                "same_line_ordinal": _same_line_ordinal(dialogue_list, absolute_index),
            }
        )

    boundaries = []
    for previous, current in zip(turns, turns[1:]):
        narrative_lines = _narrative_between(raw_lines, previous["line"], current["line"])
        boundaries.append(
            {
                "from": previous["id"],
                "to": current["id"],
                "has_narrative_break": bool(narrative_lines),
                "narrative_lines": narrative_lines[:4],
            }
        )
    for turn, boundary in zip(turns, boundaries):
        turn["following_narrative"] = boundary["narrative_lines"]
    if turns:
        final_turn = turns[-1]
        final_narrative = _narrative_between(raw_lines, final_turn["line"], raw_end + 1)
        final_turn["following_narrative"] = final_narrative[:4]

    return {
        "target": {
            "line": line_num,
            "text": dialogue,
            "dialogue_index": target_index,
            "same_line_ordinal": same_line_ordinal,
        },
        "turns": turns,
        "boundaries": boundaries,
        "raw_start": raw_start,
        "raw_end": raw_end,
        "raw_lines": [
            {"line": current_line, "text": raw_lines[current_line - 1]}
            for current_line in range(raw_start, raw_end + 1)
        ],
    }


def format_dialogue_packet(packet: dict[str, Any], max_raw_chars: int = 14000) -> str:
    """Render one packet for a model without adding any inferred speaker data."""
    target = packet["target"]
    lines = [
        "[Raw dialogue evidence packet - no prior speaker labels]",
        (
            f"Target: L{target['line']} dialogue occurrence "
            f"#{target['same_line_ordinal']}: {target['text']}"
        ),
        "",
        "[Dialogue turns in local block]",
    ]
    for turn in packet["turns"]:
        marker = " TARGET" if turn["is_target"] else ""
        lines.append(f"  {turn['id']} L{turn['line']}{marker}: {turn['text']}")

    lines.append("")
    lines.append("[Narrative boundaries between turns]")
    if not packet["boundaries"]:
        lines.append("  (No neighboring dialogue turns available.)")
    for boundary in packet["boundaries"]:
        if boundary["has_narrative_break"]:
            excerpts = " | ".join(
                f"L{item['line']}: {item['text'][:120]}"
                for item in boundary["narrative_lines"]
            )
            lines.append(f"  {boundary['from']} -> {boundary['to']}: narrative break: {excerpts}")
        else:
            lines.append(f"  {boundary['from']} -> {boundary['to']}: no non-dialogue line between them")

    lines.append("")
    lines.append("[Immediate narrative after each dialogue turn]")
    found_following_narrative = False
    for turn in packet["turns"]:
        following = turn.get("following_narrative") or []
        if not following:
            continue
        found_following_narrative = True
        excerpts = " | ".join(
            f"L{item['line']}: {item['text'][:120]}" for item in following
        )
        lines.append(f"  {turn['id']} may be clarified by: {excerpts}")
    if not found_following_narrative:
        lines.append("  (No immediate narrative line after the displayed dialogue turns.)")

    lines.append("")
    lines.append("[Exact original lines]")
    raw_lines = []
    for item in packet["raw_lines"]:
        marker = ">>" if item["line"] == target["line"] else "  "
        raw_lines.append(f"{marker} L{item['line']}: {item['text']}")
    raw_text = "\n".join(raw_lines)
    if len(raw_text) > max_raw_chars:
        target_index = next(
            index for index, item in enumerate(packet["raw_lines"])
            if item["line"] == target["line"]
        )
        focus_start = max(0, target_index - 4)
        focus_end = min(len(raw_lines), target_index + 5)
        focus = "\n".join(raw_lines[focus_start:focus_end])
        remaining = max(0, max_raw_chars - len(focus) - 100)
        before = _compact_text("\n".join(raw_lines[:focus_start]), remaining // 2, keep="tail")
        after = _compact_text("\n".join(raw_lines[focus_end:]), remaining - len(before), keep="head")
        raw_text = "\n".join(
            part for part in (
                before,
                "... [target-centered raw packet] ...",
                focus,
                "... [remaining raw context] ...",
                after,
            ) if part
        )
    lines.append(raw_text)
    return "\n".join(lines)


def packet_line_numbers(packet: dict[str, Any]) -> set[int]:
    """Return the exact raw line numbers available to an agent for citation."""
    return {item["line"] for item in packet.get("raw_lines", [])}


def citations_are_local(citations: Iterable[int], packet: dict[str, Any]) -> bool:
    """Require at least one cited line to be present in the supplied raw packet."""
    available = packet_line_numbers(packet)
    return any(line_num in available for line_num in citations)


def _find_dialogue_index(dialogue_list: list[tuple[int, str]], line_num: int, dialogue: str) -> int | None:
    for idx, (candidate_line, candidate_text) in enumerate(dialogue_list):
        if candidate_line == line_num and candidate_text == dialogue:
            return idx
    for idx, (candidate_line, _) in enumerate(dialogue_list):
        if candidate_line == line_num:
            return idx
    return None


def _same_line_ordinal(dialogue_list: list[tuple[int, str]], target_index: int) -> int:
    target_line = dialogue_list[target_index][0]
    return 1 + sum(1 for line, _ in dialogue_list[:target_index] if line == target_line)


def _narrative_between(raw_lines: list[str], previous_line: int, current_line: int) -> list[dict[str, Any]]:
    if current_line <= previous_line:
        return []
    narrative = []
    for line_num in range(previous_line + 1, min(current_line, len(raw_lines) + 1)):
        text = raw_lines[line_num - 1].strip()
        if text and "「" not in text:
            narrative.append({"line": line_num, "text": text})
    return narrative
