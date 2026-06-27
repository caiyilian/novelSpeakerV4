# -*- coding: utf-8 -*-
"""
EvidenceVault - Citation-based evidence storage for character identity.

Tiers:
  - verified: direct quote from novel (self-intro, narrator statement)
  - candidate: inference-based (e.g., a temporary descriptor seen near a named character)
  
Only SearchAgent can write verified evidence. Labeler is READ-ONLY.
"""

import json
import os


class EvidenceVault:
    def __init__(self, path=None):
        self.path = path
        self.data = self._load_or_init(path)

    def _load_or_init(self, path):
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"characters": {}}

    def save(self, path=None):
        p = path or self.path
        if p:
            parent = os.path.dirname(p)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_evidence(self, character, aliases, evidence_list, status, introduction_line=None):
        if character not in self.data["characters"]:
            self.data["characters"][character] = {
                "aliases": [],
                "status": status,
                "evidence": [],
                "introduction_line": None,
                "last_seen_line": None
            }
        entry = self.data["characters"][character]
        for a in aliases:
            if a not in entry["aliases"] and a != character:
                entry["aliases"].append(a)
        existing_lines = {e.get("line") for e in entry["evidence"]}
        for ev in evidence_list:
            if ev.get("line") not in existing_lines:
                entry["evidence"].append(ev)
                existing_lines.add(ev.get("line"))
        if status == "verified":
            entry["status"] = "verified"
        if introduction_line:
            if entry["introduction_line"] is None or introduction_line < entry["introduction_line"]:
                entry["introduction_line"] = introduction_line
        max_line = max(ev.get("line", 0) for ev in evidence_list) if evidence_list else 0
        if max_line > 0:
            if entry["last_seen_line"] is None or max_line > entry["last_seen_line"]:
                entry["last_seen_line"] = max_line

    def update_last_seen(self, character, line_num):
        real = self.resolve_alias(character)
        if real not in self.data["characters"]:
            self.data["characters"][real] = {
                "aliases": [],
                "status": "candidate",
                "evidence": [],
                "introduction_line": None,
                "last_seen_line": None
            }
        entry = self.data["characters"][real]
        if entry["last_seen_line"] is None or line_num > entry["last_seen_line"]:
            entry["last_seen_line"] = line_num

    def get_active_characters(self, current_line, window=80):
        active = []
        for name, info in self.data["characters"].items():
            ls = info.get("last_seen_line")
            if ls and (current_line - ls) < window:
                active.append(name)
        return active

    def resolve_alias(self, name):
        if name in self.data["characters"]:
            return name
        for char_name, info in self.data["characters"].items():
            if name in info.get("aliases", []):
                return char_name
        return name

    def get_state_text(self, current_line=0):
        lines = []
        active = self.get_active_characters(current_line)
        if active:
            lines.append("[Character Evidence Database]")
            for name in active:
                info = self.data["characters"].get(name, {})
                aliases = info.get("aliases", [])
                status = info.get("status", "candidate")
                evidence = info.get("evidence", [])
                alias_str = f" (aka: {', '.join(aliases)})" if aliases else ""
                status_mark = "VERIFIED" if status == "verified" else "CANDIDATE"
                lines.append(f"  [{status_mark}] {name}{alias_str}")
                if evidence:
                    for ev in evidence[:2]:
                        lines.append(f"     L{ev.get('line', '?')}: {ev.get('text', '')[:60]}")
                last = info.get("last_seen_line", "?")
                lines.append(f"     last seen: ~L{last}")
        else:
            lines.append("(No character evidence yet)")
        return "\n".join(lines)

    def get_character(self, name):
        real = self.resolve_alias(name)
        return self.data["characters"].get(real)

    def has_character(self, name):
        real = self.resolve_alias(name)
        return real in self.data["characters"]

    def all_characters(self):
        return list(self.data["characters"].keys())