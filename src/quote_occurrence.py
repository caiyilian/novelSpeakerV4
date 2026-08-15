# -*- coding: utf-8 -*-
"""Exact quote-occurrence indexing for dialogue attribution."""

from __future__ import annotations

import re
from typing import Any, Iterable


QUOTE_RE = re.compile(r"「([^」]+)」")
FALSE_SPEECH_PATTERNS = (
    re.compile(r"(?:还用|不用|无须|何必|怎么|怎能|岂能|难以|无法|不能|不敢|未曾)说(?:吗|呢|吧|得|出|清|明)?"),
    re.compile(r"(?:与其|不如|所谓|也就是|换句话|一般来|严格来)说"),
)


def build_quote_occurrences(novel_lines: Iterable[str]) -> list[dict[str, Any]]:
    """Return one stable record for every corner-bracket quote in reading order."""
    records: list[dict[str, Any]] = []
    for line_num, raw_line in enumerate(novel_lines, start=1):
        line = str(raw_line).rstrip("\n")
        spans = list(QUOTE_RE.finditer(line))
        for ordinal, match in enumerate(spans):
            previous_end = spans[ordinal - 1].end() if ordinal else 0
            next_start = spans[ordinal + 1].start() if ordinal + 1 < len(spans) else len(line)
            records.append(
                {
                    "dialogue_index": len(records),
                    "id": f"D{len(records)}",
                    "line": line_num,
                    "same_line_ordinal": ordinal,
                    "quote_start": match.start(),
                    "quote_end": match.end(),
                    "text": match.group(1),
                    "raw_line": line,
                    "scope_before": line[previous_end:match.start()].strip(),
                    "scope_after": line[match.end():next_start].strip(),
                }
            )
    return records


def align_quote_occurrences(
    novel_lines: Iterable[str],
    dialogue_list: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Build records and fail early if dialogue extraction and quote indexing diverge."""
    records = build_quote_occurrences(novel_lines)
    if len(records) != len(dialogue_list):
        raise ValueError(
            "Quote occurrence count does not match dialogue list: "
            f"{len(records)} != {len(dialogue_list)}"
        )
    for index, (record, dialogue) in enumerate(zip(records, dialogue_list)):
        line_num, text = dialogue
        if record["line"] != line_num or record["text"] != text:
            raise ValueError(
                "Quote occurrence alignment failed at dialogue "
                f"#{index + 1}: indexed L{record['line']} {record['text']!r}, "
                f"extracted L{line_num} {text!r}"
            )
    return records


def occurrence_context(
    record: dict[str, Any],
    novel_lines: list[str],
    *,
    adjacent_lines: int = 1,
) -> dict[str, Any]:
    """Attach nearby raw lines without losing the exact same-line quote scope."""
    line_num = int(record["line"])
    start = max(1, line_num - max(0, adjacent_lines))
    end = min(len(novel_lines), line_num + max(0, adjacent_lines))
    enriched = dict(record)
    enriched["nearby_lines"] = [
        {"line": number, "text": str(novel_lines[number - 1]).rstrip("\n")}
        for number in range(start, end + 1)
    ]
    return enriched


def contains_false_speech_phrase(text: str) -> bool:
    """Detect rhetorical or descriptive uses of speech verbs."""
    return any(pattern.search(text or "") for pattern in FALSE_SPEECH_PATTERNS)


def format_occurrence_hint(record: dict[str, Any]) -> str:
    """Format the exact target quote for model prompts."""
    ordinal = int(record.get("same_line_ordinal", 0)) + 1
    lines = [
        "[Exact target quote occurrence]",
        (
            f"  id={record.get('id')} line=L{record.get('line')} "
            f"same-line-quote=#{ordinal} chars={record.get('quote_start')}:{record.get('quote_end')}"
        ),
        f"  exact quote: 「{record.get('text', '')}」",
        f"  same-line scope before: {record.get('scope_before') or '(none)'}",
        f"  same-line scope after: {record.get('scope_after') or '(none)'}",
    ]
    return "\n".join(lines)
