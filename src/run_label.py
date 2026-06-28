# -*- coding: utf-8 -*-
"""
Multi-agent novel dialogue speaker annotation system v4

Architecture:
  Python orchestrator -> Labeler (tool calling, English prompt)
                             -> ShortMem (recent N rounds)
                             -> CharacterState (validated, clean)

Key design principles:
  1. Python controls flow - Agents are read-only
  2. One dialogue per round - quality over speed
  3. English prompt - LLM reasons better in English
  4. Strict character name validation - no polluted state
  5. Token budget tracking - prevent context overflow
  6. Active backward search for unnamed characters (girl -> real name)
"""
import sys, io

import sys
import os
import re
import json
import requests
import argparse
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Read Ollama config
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "ip_config")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:32b"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OLLAMA_BASE_URL="):
                OLLAMA_BASE_URL = line.split("=", 1)[1]
            elif line.startswith("OLLAMA_MODEL="):
                OLLAMA_MODEL = line.split("=", 1)[1]

NOVEL_PATH = os.path.join(ROOT_DIR, "data", "novel.txt")
LABELED_PATH = os.path.join(ROOT_DIR, "data", "labeled.txt")
ANSWERS_PATH = os.path.join(ROOT_DIR, "data", "answers.txt")
LOG_PATH = os.path.join(ROOT_DIR, "data", "label_log.jsonl")
STATE_PATH = os.path.join(ROOT_DIR, "data", "character_state.json")
VAULT_PATH = os.path.join(ROOT_DIR, "data", "evidence_vault.json")

# Token budget: stop searching when cumulative tokens exceed this
TOKEN_BUDGET = 18000
MAX_TOOL_ROUNDS = 10

# Temporary descriptors that need forward name search
TEMP_DESCRIPTORS = {"女孩", "少年", "老人", "大汉", "年轻人", "男孩", "少女", "妇人", "老妇", "男子", "女子", "青年", "孩子"}

# Forward search patterns for name reveal (must be actual name-introduction, not generic "I am X")
# Priority order: longer patterns first to avoid partial matches
NAME_REVEAL_PATTERNS = ["咱的名字是", "我的名字是", "吾乃", "名字是"]


def search_forward_for_name(line_num, descriptor):
    """
    Search forward in novel text for a real name for a temporary descriptor.
    Searches ALL remaining lines (no hard limit), returns first valid match.
    Returns (real_name, reveal_line) or None if not found.
    Only matches when the pattern is clearly introducing a character's name.
    """
    import re
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    end = len(lines)
    for i in range(line_num, end):
        line = lines[i].strip()
        if len(line) < 3:
            continue
        for pattern in NAME_REVEAL_PATTERNS:
            if pattern in line:
                idx = line.index(pattern) + len(pattern)
                # Extract name: take chars until we hit punctuation or non-name chars
                raw_name = ""
                for ch in line[idx:]:
                    if ch in '「」『』""''。，！？、，：:；;）\)\]\s':
                        break
                    raw_name += ch
                # Validate: Chinese name is 2-4 chars
                name_match = re.match(r'^([\u4e00-\u9fff]{2,4})', raw_name)
                if name_match:
                    name = name_match.group(1)
                    cleaned = validate_char_name(name)
                    if cleaned and cleaned != descriptor:
                        return cleaned, i + 1
                # If validation failed but pattern matched, continue searching
                continue
    return None, None


# ============================================================
# Logging
# ============================================================

def log_entry(entry):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def new_round(round_idx, line_num, dialogue):
    return {
        "round": round_idx,
        "dialogue_line": line_num,
        "dialogue_text": dialogue,
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "tool_calls": [],
        "result": None,
    }


def log_agent(round_log, agent_name, role, input_messages, response_text, pec, ec,
               tool_calls_list=None, total_pec=None, total_ec=None):
    entry = {
        "agent": agent_name,
        "role": role,
        "input_summary": input_messages[-1]["content"][:200] if input_messages else "",
        "response": response_text,
        "prompt_eval_count": total_pec if total_pec else pec,
        "eval_count": total_ec if total_ec else ec,
        "total_tokens": (total_pec if total_pec else pec) + (total_ec if total_ec else ec),
    }
    if tool_calls_list:
        entry["tool_calls_detail"] = tool_calls_list
    round_log["agents"][agent_name] = entry


# ============================================================
# Utility functions
# ============================================================

def read_novel_lines(start, count):
    """Read lines from novel.txt. start is 1-based."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = []
    for i in range(start - 1, min(start - 1 + count, len(lines))):
        result.append(f"{i+1}: {lines[i].rstrip()}")
    return "\n".join(result)


def search_novel(keyword, context_lines=1):
    """Search novel.txt for a keyword. Returns matching lines with context."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    results = []
    for i, line in enumerate(lines):
        if keyword in line:
            ctx_start = max(0, i - context_lines)
            ctx_end = min(len(lines), i + context_lines + 1)
            block = []
            for j in range(ctx_start, ctx_end):
                marker = ">>>" if j == i else "   "
                block.append(f"{marker}{j+1}: {lines[j].rstrip()}")
            results.append("\n".join(block))
    return "\n---\n".join(results) if results else f"(No matches for '{keyword}')"


def get_narrative_before(line_num, max_lines=5):
    """Extract narrative (non-dialogue) lines before a given line."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    narrative = []
    for i in range(line_num - 2, max(-1, line_num - 2 - max_lines), -1):
        if i < 0:
            break
        text = lines[i].rstrip()
        if "「" in text:
            break
        if text.strip():
            narrative.insert(0, text.strip())
    return " | ".join(narrative) if narrative else ""


def deep_search_identity(temp_name, around_line, search_forward=200, search_backward=100):
    """Search for identity clues for a temporary descriptor near a line."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = max(0, around_line - 1 - search_backward)
    end = min(len(lines), around_line - 1 + search_forward)
    intro_patterns = ["我叫", "咱是", "我是", "名字是", "吾乃", "咱的名字是"]

    results = [f"=== Deep identity search for '{temp_name}' near L{around_line} ==="]
    results.append(f"Range: L{start+1} to L{end} ({end-start} lines)\n")

    intro_matches = []
    for i in range(start, end):
        line_text = lines[i]
        for pat in intro_patterns:
            if pat in line_text:
                intro_matches.append((i + 1, line_text.strip()))

    if intro_matches:
        results.append(f"--- Self-introduction patterns ({len(intro_matches)} matches) ---")
        for ln, text in intro_matches:
            rel = ">" if abs(ln - around_line) < 50 else " "
            results.append(f"  {rel}L{ln}: {text[:100]}")

    temp_matches = []
    for i in range(start, end):
        if temp_name in lines[i]:
            temp_matches.append((i + 1, lines[i].strip()))

    if temp_matches:
        results.append(f"\n--- '{temp_name}' occurrences ({len(temp_matches)} matches) ---")
        for ln, text in temp_matches[:15]:
            results.append(f"  L{ln}: {text[:100]}")
        if len(temp_matches) > 15:
            results.append(f"  ... and {len(temp_matches) - 15} more")

    if not intro_matches and not temp_matches:
        results.append("(No relevant matches found)")

    return "\n".join(results)


def find_all_references(name, max_results=10):
    """Find all occurrences of a name in the novel with context."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    results = []
    count = 0
    for i, line in enumerate(lines):
        if name in line:
            ctx_start = max(0, i - 1)
            ctx_end = min(len(lines), i + 2)
            block = []
            for j in range(ctx_start, ctx_end):
                marker = ">>>" if j == i else "   "
                block.append(f"{marker}{j+1}: {lines[j].rstrip()}")
            results.append("\n".join(block))
            count += 1
            if count >= max_results:
                break
    return "\n---\n".join(results) if results else f"(No references to '{name}')"


def build_context_index(line_num, context_window=40):
    """Build an abstract map of novel around target line (no full text).
    Forces the Labeler to use read_novel_lines to see actual content."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = max(1, line_num - context_window)
    end = min(len(lines), line_num + context_window)
    result = []
    result.append(f"[Novel Map near L{line_num} (read full text with read_novel_lines)]")
    result.append(f"  -> = target  [D] = has dialogue  [N] = narrative")

    last_type = None
    block_start = None
    block_lines = []

    for i in range(start - 1, end):
        ln = i + 1
        text = lines[i].rstrip()
        has_dialogue = "\u300c" in text
        line_type = "D" if has_dialogue else "N"

        if line_type == "D":
            if last_type == "N" and block_start is not None:
                result.append(f"     L{block_start}-L{ln-1} [N] ({len(block_lines)} lines)")
            marker = "->" if ln == line_num else "  "
            dlg = re.search(r"\u300c([^\u300d]+)\u300d", text)
            dlg_text = dlg.group(1)[:40] + "..." if dlg and len(dlg.group(1)) > 40 else (dlg.group(1) if dlg else "")
            result.append(f"  {marker}L{ln} [D] \u300c{dlg_text}\u300d")
            last_type = "D"
            block_start = None
            block_lines = []
        else:
            if last_type != "N":
                if block_start is not None:
                    result.append(f"     L{block_start}-L{ln-1} [N] ({len(block_lines)} lines)")
                block_start = ln
                block_lines = []
            block_lines.append(text)
            last_type = "N"

    if block_start is not None and block_lines:
        result.append(f"     L{block_start}-L{min(len(lines), end)} [N] ({len(block_lines)} lines)")

    return "\n".join(result)


def get_dialogue_list():
    """Extract all dialogues from novel as (line_num, text) pairs."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    dialogues = []
    for line_num, line in enumerate(content.split("\n"), start=1):
        matches = re.findall(r"「([^」]+)」", line)
        for dialogue in matches:
            dialogues.append((line_num, dialogue))
    return dialogues


def get_labeled_count():
    if not os.path.exists(LABELED_PATH):
        return 0
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        return len(f.readlines())


def write_label(name):
    with open(LABELED_PATH, "a", encoding="utf-8") as f:
        f.write(name + "\n")


def call_ollama(messages, tools=None, label=""):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(url, json=payload, timeout=300)
    data = resp.json()
    pec = data.get("prompt_eval_count", 0)
    ec = data.get("eval_count", 0)
    text = data.get("message", {}).get("content", "")
    tool_calls = data.get("message", {}).get("tool_calls", [])
    return text, pec, ec, tool_calls


# ============================================================
# Tool definition
# ============================================================

TOOL_READ_NOVEL = {
    "type": "function",
    "function": {
        "name": "read_novel_lines",
        "description": "Read lines from novel.txt. start is 1-based line number. count is how many lines to read. Returns lines in 'line_number: content' format.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "Starting line number (1-based)"},
                "count": {"type": "integer", "description": "Number of lines to read"}
            },
            "required": ["start", "count"]
        }
    }
}

TOOL_SEARCH_NOVEL = {
    "type": "function",
    "function": {
        "name": "search_novel",
        "description": "Search the novel text for a keyword or character name. Returns matching lines with surrounding context. Use this to find where a character was introduced, where their name appears, or to confirm identity clues.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "The keyword or character name to search for"},
                "context_lines": {"type": "integer", "description": "Number of context lines before and after each match (default: 2)"}
            },
            "required": ["keyword"]
        }
    }
}

LABELER_TOOLS = [TOOL_READ_NOVEL, TOOL_SEARCH_NOVEL]


# ============================================================
# Character State - clean, validated, no pollution
# ============================================================

def validate_char_name(name):
    """
    Validate and clean a character name.
    Returns cleaned name or None if invalid.
    Rejects: names with description text appended.
    """
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if not name:
        return None
    # Reject if contains description patterns (model sometimes appends these)
    bad_patterns = ["--", "—", "–", "\u2014", "\u2013", "根据", "原文", "禁止", "错误示例"]
    for bp in bad_patterns:
        if bp in name:
            return None
    # Reject if too long (>15 chars means it's probably a description)
    if len(name) > 15:
        return None
    # Reject if contains whitespace
    if re.search(r"\s", name):
        return None
    # Reject if contains Chinese parentheses or brackets with content
    if re.search(r"[（(【\[].*[）)】\]]", name):
        return None
    return name


class CharacterState:
    """Clean character state with strict validation."""

    def __init__(self):
        self.state = self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Validate loaded data - clean if corrupted
                if self._is_valid_state(data):
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return {"characters": {}, "alias_map": {}, "speech_order": []}

    def _is_valid_state(self, data):
        if not isinstance(data, dict):
            return False
        if "characters" not in data or "alias_map" not in data:
            return False
        # Check for corrupted entries
        for name in data.get("characters", {}):
            if not validate_char_name(name):
                return False
        return True

    def update_speech_order(self, speaker_name):
        """Track the order of who spoke, for alternating pattern detection."""
        cleaned = validate_char_name(speaker_name)
        if not cleaned:
            return
        real_name = self.resolve_alias(cleaned)
        order = self.state.setdefault("speech_order", [])
        # Keep last 10 entries
        order.append(real_name)
        if len(order) > 10:
            order.pop(0)

    def get_recent_speakers_text(self):
        """Get a hint about who spoke recently - useful for alternating pattern."""
        order = self.state.get("speech_order", [])
        if not order:
            return ""
        unique = []
        for s in order:
            if s not in unique:
                unique.append(s)
            if len(unique) >= 3:
                break
        last = order[-1] if order else ""
        if last:
            text = f"  [Last speaker: {last}]"
            if len(unique) >= 2:
                text += f"\n  [Exchange: {unique[0]} ↔ {unique[1]}]"
            # Check if the last 4 show clear alternation
            if len(order) >= 4:
                recent = order[-4:]
                if recent[0] != recent[1] and recent[2] != recent[3] and recent[0] == recent[2] and recent[1] == recent[3]:
                    text += "\n  [Alternating pattern confirmed]"
            return text
        return ""

    def save(self, round_num=0):
        if round_num > 0 and os.path.exists(STATE_PATH):
            history_dir = os.path.join(ROOT_DIR, "data", "state_history")
            os.makedirs(history_dir, exist_ok=True)
            backup_path = os.path.join(history_dir, f"character_state_round_{round_num:04d}.json")
            with open(STATE_PATH, "r", encoding="utf-8") as src:
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def resolve_alias(self, name):
        """Resolve an alias to the canonical character name."""
        alias_map = self.state.get("alias_map", {})
        if name in alias_map:
            return alias_map[name]
        for char_name, char_info in self.state.get("characters", {}).items():
            if name in char_info.get("aliases", []):
                return char_name
        return name

    def add_character(self, name, aliases=None):
        """Add a new character with validated name."""
        cleaned = validate_char_name(name)
        if not cleaned:
            return
        if cleaned not in self.state["characters"]:
            self.state["characters"][cleaned] = {
                "aliases": [cleaned],
                "first_seen_line": None,
                "last_seen_line": None,
                "recent_lines": [],
            }
        for alias in (aliases or []):
            cleaned_alias = validate_char_name(alias)
            if cleaned_alias and cleaned_alias not in self.state["characters"][cleaned]["aliases"]:
                self.state["characters"][cleaned]["aliases"].append(cleaned_alias)
                self.state["alias_map"][cleaned_alias] = cleaned

    def update_character(self, name, line_num, dialogue_text=""):
        """Update character's last seen line and recent dialogue."""
        cleaned = validate_char_name(name)
        if not cleaned:
            return
        real_name = self.resolve_alias(cleaned)
        if real_name not in self.state["characters"]:
            self.add_character(real_name)
        char = self.state["characters"][real_name]
        if char["first_seen_line"] is None:
            char["first_seen_line"] = line_num
        char["last_seen_line"] = line_num
        if dialogue_text:
            char["recent_lines"].append((line_num, dialogue_text))
            if len(char["recent_lines"]) > 5:
                char["recent_lines"].pop(0)

    def add_alias(self, real_name, alias, is_identity_reveal=False):
        """Add an alias mapping."""
        cleaned_real = validate_char_name(real_name)
        cleaned_alias = validate_char_name(alias)
        if not cleaned_real or not cleaned_alias:
            return
        if is_identity_reveal:
            # If the alias already exists as its own character, merge it into real_name
            if cleaned_alias in self.state["characters"] and cleaned_alias != cleaned_real:
                alias_char = self.state["characters"].pop(cleaned_alias)
                # Transfer recent_lines and first/last_seen
                if cleaned_real not in self.state["characters"]:
                    self.add_character(cleaned_real)
                target = self.state["characters"][cleaned_real]
                target["recent_lines"].extend(alias_char.get("recent_lines", []))
                if len(target["recent_lines"]) > 5:
                    target["recent_lines"] = target["recent_lines"][-5:]
                if alias_char.get("first_seen_line") and (target["first_seen_line"] is None or alias_char["first_seen_line"] < target["first_seen_line"]):
                    target["first_seen_line"] = alias_char["first_seen_line"]
                if alias_char.get("last_seen_line") and (target["last_seen_line"] is None or alias_char["last_seen_line"] > target["last_seen_line"]):
                    target["last_seen_line"] = alias_char["last_seen_line"]
                # Remove old alias_map entries pointing to the aliased character
                for old_alias, mapped in list(self.state.get("alias_map", {}).items()):
                    if mapped == cleaned_alias:
                        self.state["alias_map"][old_alias] = cleaned_real
            if cleaned_real not in self.state["characters"]:
                self.add_character(cleaned_real)
            if cleaned_alias not in self.state["characters"][cleaned_real]["aliases"]:
                self.state["characters"][cleaned_real]["aliases"].append(cleaned_alias)
            self.state["alias_map"][cleaned_alias] = cleaned_real

    def get_active_characters(self, last_n_lines=50, current_line=0):
        """Get characters active within last N lines."""
        active = []
        for name, info in self.state.get("characters", {}).items():
            ls = info.get("last_seen_line")
            if ls and (current_line - ls) <= last_n_lines:
                active.append(name)
        return active

    def get_state_text(self, current_line=0):
        """Generate character state text for Labeler."""
        lines = []
        active = self.get_active_characters(last_n_lines=50, current_line=current_line)
        lines.append("[Character State]")
        for name in active:
            info = self.state["characters"].get(name, {})
            aliases = info.get("aliases", [name])
            recent = info.get("recent_lines", [])
            alias_str = ", ".join(a for a in aliases if a != name)
            if alias_str:
                lines.append(f"  {name} (aka: {alias_str})")
            else:
                lines.append(f"  {name}")
            last_ls = info.get('last_seen_line', '?')
            lines.append(f"    last seen: ~L{last_ls}")
            if recent:
                recent_str = "; ".join([f"L{ln}「{t[:20]}」" for ln, t in recent[-3:]])
                lines.append(f"    recent: {recent_str}")
        if len(self.state.get("characters", {})) > len(active):
            inactive = len(self.state["characters"]) - len(active)
            lines.append(f"  ({inactive} inactive characters)")
        return "\n".join(lines)

    def get_scene_summary(self, current_line=0):
        """Generate brief scene summary."""
        active = self.get_active_characters(last_n_lines=30, current_line=current_line)
        if not active:
            return "[Scene] Initial scene"
        chars_str = ", ".join(active)
        lines_list = [self.state["characters"][c]["last_seen_line"] for c in active
                      if self.state["characters"][c].get("last_seen_line")]
        if lines_list:
            return f"[Scene] L{min(lines_list)}-L{max(lines_list)} | Active: {chars_str}"
        return f"[Scene] Active: {chars_str}"

    def parse_state_update(self, text, line_num, dialogue=""):
        """Parse <state_update> block from Labeler output, with strict validation."""
        match = re.search(r"<state_update>(.*?)</state_update>", text, re.DOTALL)
        if not match:
            return
        body = match.group(1).strip()
        for action_line in body.split("\n"):
            action_line = action_line.strip()
            if not action_line:
                continue

            # UPDATE character: name.field = value
            m = re.match(r"UPDATE\s+character:\s+(\S+)\.(\S+)\s*=\s*(.+)", action_line)
            if m:
                char_name = validate_char_name(m.group(1))
                if not char_name:
                    continue
                field = m.group(2)
                value = m.group(3).strip()
                self.update_character(char_name, line_num, dialogue_text=dialogue)
                if field == "alias":
                    cleaned_val = validate_char_name(value)
                    if cleaned_val:
                        self.add_alias(char_name, cleaned_val)
                continue

            # UPDATE alias: alias_name -> real_name
            m = re.match(r"UPDATE\s+alias:\s+(.+?)\s*->\s*(.+)", action_line)
            if m:
                alias = validate_char_name(m.group(1).strip())
                real = validate_char_name(m.group(2).strip())
                if alias and real:
                    # Check for identity reveal patterns
                    intro_patterns = ["名字是", "我叫", "咱是", "咱的名字是", "吾乃", "我是", "名は"]
                    is_reveal = any(p in dialogue for p in intro_patterns)
                    self.add_alias(real, alias, is_identity_reveal=is_reveal)
                continue

            # NEW character: name
            m = re.match(r"NEW\s+character:\s+(.+)", action_line)
            if m:
                raw_name = m.group(1).strip()
                cleaned = validate_char_name(raw_name)
                if not cleaned:
                    continue
                # Don't create duplicate if name already exists
                if cleaned not in self.state["characters"]:
                    self.add_character(cleaned)
                continue


# ============================================================
# Short-term Memory Agent
# ============================================================

class ShortMemAgent:
    def __init__(self, max_rounds=20):
        self.max_rounds = max_rounds
        self.history = []  # [(line_num, dialogue_text, speaker, reason, narrative_before)]
        self.phase_violation = False  # set by Boss when model contradicts alternating pattern

    def update(self, line_num, dialogue_text, speaker, reason="", narrative_before=""):
        self.history.append((line_num, dialogue_text, speaker, reason, narrative_before))
        if len(self.history) > self.max_rounds:
            self.history.pop(0)

    def detect_rapid_exchange(self, min_length=3):
        """Detect if last N entries alternate between two speakers."""
        if len(self.history) < min_length:
            return False
        recent = [entry[2] for entry in self.history[-min_length:]]
        unique = list(dict.fromkeys(recent))
        if len(unique) != 2:
            return False
        for i in range(1, len(recent)):
            if recent[i] == recent[i-1]:
                return False
        return True

    def _get_exchange_rhythm(self):
        """Detect rapid 2-person exchange pattern and return (text, next_expected)."""
        if len(self.history) < 4:
            return None, None
        recent = self.history[-8:]  # last 8 rounds max
        speakers = [sp for _, _, sp, _, _ in recent]

        # Get the last two distinct speakers (in order of first appearance)
        seen = []
        for sp in speakers:
            if sp not in seen:
                seen.append(sp)
            if len(seen) == 2:
                break
        if len(seen) != 2:
            return None  # Not a 2-person exchange

        a, b = seen[0], seen[1]

        # Check if the recent sequence is a rapidly alternating pattern
        # Criteria: at least 4 rounds, both speakers appear at least 2 times
        count_a = speakers.count(a)
        count_b = speakers.count(b)
        if count_a < 2 or count_b < 2:
            return None, None

        # Check how well they alternate (allow one deviation)
        expected = a
        switches = 0
        for sp in speakers:
            if sp == expected:
                switches += 1
                expected = b if expected == a else a
        # If more than half the positions match an alternating pattern, it's likely a dialogue
        if switches >= len(speakers) * 0.6:
            next_expected = b if speakers[-1] == a else a
            text = (f"[Exchange rhythm - last {len(speakers)} rounds]\n"
                    f"  Alternating: {a} \u2194 {b}\n"
                    f"  Current expects: {next_expected} (opposite of previous speaker)\n"
                    f"  NEXT EXPECTED: {next_expected}")
            return text, next_expected
        return None, None

    def get_next_expected(self):
        """Get the expected next speaker based on alternating pattern, or None."""
        _, next_exp = self._get_exchange_rhythm()
        if next_exp:
            return next_exp
        # Short-window fallback: just look at last 2
        if len(self.history) >= 2:
            last_two = set(entry[2] for entry in self.history[-2:])
            if len(last_two) == 2:
                speakers = list(last_two)
                last = self.history[-1][2]
                return speakers[1] if speakers[0] == last else speakers[0]
        return None

    def get_summary(self):
        if not self.history:
            return "(No prior annotations)"
        lines = ["[Recent Dialogues]"]
        for line_num, dlg, speaker, reason, narr in self.history:
            narr_note = f" [before: {narr[:60]}]" if narr else ""
            line = f"  L{line_num}「{dlg[:50]}」-> {speaker}{narr_note}"
            lines.append(line)
        # Add exchange rhythm hint
        rhythm, _ = self._get_exchange_rhythm()
        if rhythm:
            lines.append("")
            lines.append(rhythm)

        # Add phase violation constraint for the next round
        if self.phase_violation:
            next_exp = self.get_next_expected()
            if next_exp:
                lines.append("")
                lines.append(f"[ANCHOR CONSTRAINT] Last round violated the alternating pattern!")
                lines.append(f"The last {min(len(self.history), 8)} rounds form {self._alternating_pair_name()}.")
                lines.append(f"NEXT SPEAKER MUST BE: {next_exp} (strict alternation - do NOT repeat the previous speaker)")
                lines.append("RULE: Only override this if the narrative clearly assigns speech to a different character.")

        # Add rapid exchange warning
        if self.detect_rapid_exchange(4):
            lines.append("")
            lines.append("RAPID EXCHANGE: Last 4 dialogues alternate between two speakers.")
            lines.append("RULE: Narrative attribution (#1) before alternation (#3). Read 3-5 lines before target.")
        return "\n".join(lines)

    def _alternating_pair_name(self):
        """Get the two alternating speaker names as a readable string."""
        if len(self.history) < 4:
            return "?"
        speakers = [entry[2] for entry in self.history[-min(len(self.history), 8):]]
        seen = []
        for s in speakers:
            if s not in seen:
                seen.append(s)
            if len(seen) == 2:
                break
        if len(seen) == 2:
            return f"{seen[0]} ↔ {seen[1]} alternation"
        return "alternation"


# ============================================================
# FactCurator Agent - structured character fact maintenance
# ============================================================

class FactCurator:
    """Maintains structured character facts, runs every N rounds."""

    def __init__(self, curator_every=10):
        self.curator_every = curator_every
        self.curation_count = 0
        self.pending_rounds = []
        self.fact_summary = "(No character fact database yet)"

    def add_round(self, line_num, speaker, summary):
        self.pending_rounds.append({"line": line_num, "speaker": speaker, "summary": summary})

    def should_curate(self):
        return len(self.pending_rounds) >= self.curator_every

    def curate(self, short_mem_text, char_state_json, round_log):
        if not self.pending_rounds:
            return
        summaries = "\n".join([f"#{r['line']}: {r['speaker']} | {r['summary']}" for r in self.pending_rounds])
        system_prompt = "You are a character database curator. Examine the current character state and recent annotations. Output ONLY a JSON object: {\"updates\":[...]}. Each update: {\"action\":\"add_character\"|\"add_alias\"|\"none\",\"target\":\"name\",\"detail\":{}}. If nothing new, output {\"updates\":[{\"action\":\"none\"}]}."
        user_content = f"""Current character state:
{json.dumps(char_state_json, ensure_ascii=False, indent=2)[:1500]}

Recent annotations:
{short_mem_text[:1000]}

Summaries:
{summaries[:1000]}

Check for new characters, alias relationships, or character facts."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        text, pec, ec, _ = call_ollama(messages, label="FactCurator")
        log_agent(round_log, "FactCurator", "curator", messages, text, pec, ec)
        # Build fact summary from current character state
        chars = char_state_json.get("characters", {})
        summary_lines = ["[Character Fact Database]"]
        if chars:
            for name, info in chars.items():
                aliases = info.get("aliases", [])
                non_self = [a for a in aliases if a != name]
                alias_str = f" (also: {', '.join(non_self)})" if non_self else ""
                last = info.get("last_seen_line", "?")
                summary_lines.append(f"  {name}{alias_str} (last L{last})")
        else:
            summary_lines.append("  (no characters yet)")
        self.fact_summary = "\n".join(summary_lines)
        self.pending_rounds = []
        self.curation_count += 1

    def get_summary(self):
        return self.fact_summary


# ============================================================
# Labeler Agent
# ============================================================

class LabelerAgent:
    def __init__(self):
        self.tool_call_count = 0

    def label(self, line_num, dialogue, short_mem_text, fact_summary,
               char_state_text, navigation_text, scene_summary, round_log, quiet=False,
               override_force_tool=True, recent_speakers_hint=""):
        """
        Label one dialogue's speaker. Returns (speaker, summary, reason, pec, ec).
        """

        system_prompt = """You are a dialogue speaker annotation assistant. Your ONLY task is to identify who speaks a given line of dialogue in a Chinese novel.

========================================
TOOL - Your ONLY source for novel text
========================================

Tool: read_novel_lines(start, count)
- Reads 'count' lines from novel.txt starting at line 'start' (1-based)
- Returns each line as: "line_number: content"
- You can call this multiple times, in any order, for any range
**You do NOT have the novel text. You MUST use read_novel_lines to read it.**

========================================
EVIDENCE HIERARCHY (Priority)
========================================

1. [HIGHEST] Speech verbs naming the speaker in the IMMEDIATE context
   Lines like: "XX说", "XX喊道", "XX开口", "XX回答", "XX问", "XX叹息", "XX回答"
   If you find one within 5 lines of the dialogue → USE IT. You are done.

2. [HIGH] Narrative-position evidence
   The paragraph/sentence structure around the dialogue. E.g.:
   - "XX做了某事，然后说：" → XX is the speaker
   - "XX说道：" → XX is the speaker
   - "XX回答那人说：" → XX is the speaker
   The line IMMEDIATELY before a 「dialogue」 is the most important.

3. [MEDIUM] Alternating dialogue pattern
   When two characters trade short lines in quick succession.
   BUT: only trust the pattern when there is NO narrative paragraph between lines.
   A narrative paragraph between two lines BREAKS the alternating pattern.

4. [LOW] Character availability / scene presence
   Just because a character was mentioned or recently active does NOT mean they are speaking.

========================================
CRITICAL RULES - Do NOT Violate
========================================

RULE A: Speech verbs over everything
- If you find "说/喊道/问/开口/回答/继续说/低语" naming a character within 5 lines, THAT is the speaker
- Ignore any narrative that describes appearance/thoughts/actions of a different character
- Narrative describing how someone looks/feels does NOT prove they are speaking

RULE B: Read locally FIRST
- Start reading from 5-10 lines BEFORE the dialogue line
- The evidence you need is almost always within 5 lines
- Only expand search range if you find NO speech verb in the immediate context
- Do NOT search 40+ lines away unless local search found nothing

RULE C: Distinguish "speaker" from "mentioned person"
- "被XX的人物" = mentioned, not speaking
- "XX的人物" = describes someone, not speaking
- A line like "XX看着YY" describes the observer (XX), not the speaker
- Look for the pattern: [Narrative about character A + speech verb]「dialogue」→ A is speaker

RULE D: Speaker names - use what the text actually uses
- Has a specific name → use that name
- Has a role/group identity but no name → use the role: 村民, 骑士, 商人, 众人
- Group/collective speech → use the group name, NOT "非人物发声"
- Temporary descriptor (女孩, 少年, 老汉, 大汉, etc.) → only use if no real name is found after forward search
- Do NOT make up a name or use the wrong role

RULE E: Non-person speech - ONLY for these cases
- Ambient sounds, object sounds, sound effects → "非人物发声"
- If the text explicitly says a character made the sound, credit the character
- Collective shouting/group speech is NOT "非人物发声"

RULE F: Alternating dialogue - specific rules
- Two characters exchanging short lines (<20 chars each) → fast exchange likely
- If there is NO narrative between two adjacent dialogues, the speaker likely alternates
- But: if a narrative paragraph appears between two dialogues → the pattern RESETS
- One character CAN speak multiple consecutive lines (not always alternating)
- RULE: If immediate narrative contains a speech verb → that overrides ALL alternating patterns
- RULE: If there is no speech verb AND no narrative between → use alternating + who-last-spoke

========================================
WORKFLOW
========================================

1. CALL read_novel_lines for lines around the target (±15 lines)
   Read lines 10-15 BEFORE the dialogue first. Look for speech verbs naming the speaker.

2. If you find a speech verb within 5 lines → CONFIRM and output answer immediately
   Do NOT search further. You have your answer.

3. If no speech verb found → expand search range (±30 lines)
   Check for:
   a. Who spoke last (from auxiliary info and text)
   b. Is there an alternating pattern?
   c. Did the scene change?

4. If using a temporary descriptor (女孩, 少年, etc.) → search forward 100-200 lines for a name reveal
   Look for: "我叫XX", "咱的名字是XX", "名字是XX", "吾乃XX"
   If found → use the revealed name. If not found after 200 lines → use the descriptor.

5. Output with <answer>, <reason>, <summary>. Optionally add <discovery> if you find new character info.

========================================
OUTPUT FORMAT
========================================

<answer>speaker_name</answer>
- One single speaker name, or "非人物发声"
- Do NOT use "|" to separate multiple names
- If unsure, use a descriptive label (what the novel calls them). This triggers an automated search.

<reason>your reasoning</reason>
- In English or Chinese (both OK)
- Cite specific line numbers as evidence
- State which speech verb or evidence you found
- If auxiliary info contradicts the novel text, explain why

<summary>brief summary</summary>
- Format: speaker_name | what they said/did
- Keep concise for memory

<discovery>(Optional) Identity evidence discovered</discovery>
- Only include if you found new evidence about a character's identity
- Example: "角色在L42自我介绍为XX"
- This will be processed by a separate search agent"""

        user_content = f"""Annotate the speaker of this dialogue:

Line {line_num}: 「{dialogue}」

========================================
Auxiliary info - original text in navigation is 100% accurate.
Short-term memory labels and character state are reference only.
========================================

{scene_summary}

[Original Text Navigation - around target line]
{navigation_text}

{short_mem_text}

{fact_summary}

{recent_speakers_hint}

[Character State]
{char_state_text}

========================================
You can use TWO tools:
1. read_novel_lines(start, count) — read specific lines
2. search_novel(keyword, context_lines=2) — search by keyword/name
Start reading 5-10 lines BEFORE the target line. Look for speech verbs naming the speaker.
If found, you are done. If not, expand gradually or use search_novel.
If you think an auxiliary label is wrong, say so in <reason>.
========================================="""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        total_pec = 0
        total_ec = 0
        tool_call_log = []
        cumulative_tokens = 0

        for round_i in range(MAX_TOOL_ROUNDS):
            text, pec, ec, tool_calls = call_ollama(messages, tools=LABELER_TOOLS, label=f"Labeler-R{round_i+1}")
            total_pec += pec
            total_ec += ec
            cumulative_tokens += pec + ec

            if not tool_calls:
                # Force at least one tool call on first round (unless override)
                if round_i == 0 and override_force_tool:
                    if not quiet:
                        print(f"    Model didn't call tool, forcing...")
                    messages.append({"role": "user", "content": "You MUST call read_novel_lines to read the novel text. You cannot answer without reading."})
                    continue
                # Check for garbage output on non-first rounds
                temp_speaker = self._parse_answer(text)
                if self._is_garbage_speaker(temp_speaker):
                    if not quiet:
                        print(f"    Garbage output '{temp_speaker}', asking model to self-correct...")
                    messages.append({"role": "user", "content": f"Your answer '{temp_speaker}' is not a valid character name. Think again based on the novel text and output a proper <answer>."})
                    continue
                log_agent(round_log, "Labeler", "labeler", messages, text, pec, ec,
                          total_pec=total_pec, total_ec=total_ec,
                          tool_calls_list=tool_call_log if tool_call_log else None)
                break

            tc_details = []
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                tc_details.append({
                    "function": func.get("name"),
                    "arguments": func_args
                })

            tool_call_log.append({
                "round": round_i + 1,
                "function": tc_details[0]["function"] if tc_details else "",
                "args": tc_details[0]["arguments"] if tc_details else {},
                "prompt_eval_count": pec,
                "eval_count": ec,
            })

            messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})

            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                name = func.get("name", "")
                if name == "read_novel_lines":
                    s = func_args.get("start", 1)
                    c = func_args.get("count", 10)
                    result = read_novel_lines(s, c)
                    self.tool_call_count += 1
                    if not quiet:
                        print(f"    Tool call #{self.tool_call_count}: read_novel_lines({s}-{s+c-1}) -> {len(result)} chars, pec={pec}, ec={ec}")
                    messages.append({"role": "tool", "content": result})
                elif name == "search_novel":
                    keyword = func_args.get("keyword", "")
                    ctx = func_args.get("context_lines", 2)
                    result = search_novel(keyword, ctx)
                    self.tool_call_count += 1
                    if not quiet:
                        print(f"    Tool call #{self.tool_call_count}: search_novel('{keyword}') -> {len(result)} chars, pec={pec}, ec={ec}")
                    messages.append({"role": "tool", "content": result})

            # Token budget check
            if cumulative_tokens >= TOKEN_BUDGET:
                if not quiet:
                    print(f"    Token budget reached ({cumulative_tokens} >= {TOKEN_BUDGET}), stopping search")
                # Force final answer on next call
                messages.append({"role": "user", "content": "Token budget reached. Output your best answer with <answer>, <reason>, and <summary> tags."})
                continue

        else:
            if not quiet:
                print(f"    Max tool rounds reached ({MAX_TOOL_ROUNDS})")

        round_log["tool_calls"] = tool_call_log

        # Track how many tool rounds the Labeler used (for Verifier decision)
        tool_rounds_used = len(tool_call_log)

        speaker = self._parse_answer(text)
        summary = self._parse_summary(text)
        reason = self._parse_reason(text)

        return speaker, summary, reason, total_pec, total_ec, tool_rounds_used

    def _is_garbage_speaker(self, speaker):
        """Check if a speaker name is clearly garbage (not a valid character name)."""
        if not speaker:
            return True
        # Contains English letters -> garbage (Chinese novels should have Chinese names)
        if re.search(r'[a-zA-Z]', speaker):
            return True
        # Question words = model is confused
        if speaker in ("什么", "啥", "谁", "哪个", "哪里", "为什么", "怎么"):
            return True
        # Contains only punctuation/symbols
        if not re.search(r'[\u4e00-\u9fff]', speaker):
            return True
        return False

    def _parse_answer(self, text):
        """Extract and validate speaker name from <answer> tag."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            cleaned = validate_char_name(raw)
            if cleaned:
                return cleaned
            # If validation failed, try to extract first valid name
            # Sometimes model outputs "name (description)" - take just the name
            parts = re.split(r'\s*[—\-–(（]', raw)
            if parts:
                cleaned = validate_char_name(parts[0].strip())
                if cleaned:
                    return cleaned
            return raw  # Return as-is if we can't clean it
        # Fallback: first line of response
        fallback = text.strip().split("\n")[0][:50]
        return fallback

    def _parse_summary(self, text):
        match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_reason(self, text):
        match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if match:
            return match.group(1).strip()[:200]
        return ""


# ============================================================
# Verifier Agent - independent cross-check of Labeler's output
# ============================================================

class VerifierAgent:
    """
    Independent verification agent.
    Each round, after Labeler outputs a speaker, Verifier reads the same context
    and independently determines the speaker. If it disagrees with Labeler,
    returns the suggested correction.
    """

    def __init__(self):
        self.tool_call_count = 0

    def verify(self, line_num, dialogue, navigation_text, short_mem_text,
               scene_summary, char_state_text, labeler_speaker, round_log, quiet=False):
        """
        Independent verification. Returns (verdict, suggested_speaker, reason).
        verdict: "confirm" or "disagree"
        """
        system_prompt = """You are an independent dialogue speaker VERIFIER. A primary annotator has already labeled a dialogue. Your job is to independently determine the speaker and compare.

You have the same information available as the primary annotator:
1. Original novel text in navigation (100% accurate)
2. Recent annotation history
3. Character state
4. The primary annotator's answer

RULES:
- Read the novel text independently using read_novel_lines
- Form your OWN conclusion about who is speaking
- Then compare with the primary answer
- If you agree, output <verdict>confirm</verdict>
- If you disagree, output <verdict>disagree</verdict> and provide your suggested speaker

OUTPUT:
<verdict>confirm</verdict> or <verdict>disagree</verdict>
<suggested_speaker>name</suggested_speaker>
<reason>your evidence citing specific line numbers</reason>"""

        user_content = f"""Verify the speaker of this dialogue:

Line {line_num}: 「{dialogue}」

Primary annotator's answer: {labeler_speaker}

========================================
Original text navigation:
{navigation_text}

{short_mem_text}

{scene_summary}

[Character State]
{char_state_text}

========================================
Call read_novel_lines to verify. Form your own conclusion first, then compare.
========================================"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Allow up to 1 tool round for quick verification
        for round_i in range(1):
            text, pec, ec, tool_calls = call_ollama(messages, tools=[TOOL_READ_NOVEL], label=f"Verifier-R{round_i+1}")
            if not tool_calls:
                break
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if func.get("name") == "read_novel_lines":
                    s = func_args.get("start", 1)
                    c = func_args.get("count", 10)
                    result = read_novel_lines(s, c)
                    self.tool_call_count += 1
                    messages.append({"role": "tool", "content": result})
            messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})

        # Parse verdict
        verdict = "confirm"
        suggested = labeler_speaker
        reason = ""
        if "<verdict>disagree</verdict>" in text:
            verdict = "disagree"
            m = re.search(r"<suggested_speaker>(.*?)</suggested_speaker>", text, re.DOTALL)
            if m:
                raw = m.group(1).strip()
                cleaned = validate_char_name(raw)
                if cleaned:
                    suggested = cleaned
        m = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if m:
            reason = m.group(1).strip()[:200]

        if not quiet:
            if verdict == "disagree":
                self._safe_print(f"  Verifier disagrees: Labeler={labeler_speaker} -> Verifier={suggested}")
            else:
                self._safe_print(f"  Verifier confirms: {labeler_speaker}")

        return verdict, suggested, reason


# ============================================================
# Boss (Python orchestrator)
# ============================================================

class Boss:
    def __init__(self, short_mem_rounds=20):
        self.labeler = LabelerAgent()
        self.short_mem = ShortMemAgent(max_rounds=short_mem_rounds)
        self.dialogue_list = get_dialogue_list()
        self.total_tokens = 0
        self.round_count = 0
        self.search_agent_triggers = 0
        self.corrections = 0
        # Import EvidenceVault and SearchAgent
        from evidence_vault import EvidenceVault
        from search_agent import SearchAgent
        self.vault = EvidenceVault(VAULT_PATH)
        self.search_agent = SearchAgent(
            call_ollama_fn=call_ollama,
            read_novel_fn=read_novel_lines,
            deep_search_fn=deep_search_identity,
            find_refs_fn=find_all_references
        )

    def _build_navigation(self, line_num, nav_range=25):
        """Build original text navigation around target line - NO speaker labels."""
        lines = []
        start = max(1, line_num - nav_range)
        end = line_num + nav_range
        novel_lines = read_novel_lines(start, end - start + 1).split("\n")

        for line_text in novel_lines:
            if not line_text.strip():
                continue
            parts = line_text.split(":", 1)
            if len(parts) < 2:
                continue
            try:
                ln = int(parts[0].strip())
            except ValueError:
                continue
            content = parts[1].strip()

            marker = " "
            if ln == line_num:
                marker = ">>"

            if "\u300c" in content:
                display = content[:120]
                lines.append(f"  {marker}L{ln}: \u300c{display}\u300d")
            else:
                display = content[:120]
                lines.append(f"  {marker}L{ln}: [narr] {display}")

        return "\n".join(lines)

    def _safe_print(self, msg):
        """Print to stderr, resilient on Windows even after Ollama API calls that may corrupt stdout."""
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except (ValueError, OSError):
            pass

    def process_one(self, line_num, dialogue, round_log, quiet=False):
        self.round_count += 1

        if not quiet:
            self._safe_print(f"\n{'='*60}")
            self._safe_print(f"  Round {self.round_count}")
            self._safe_print(f"  Dialogue: L{line_num}\u300c{dialogue}\u300d")
            self._safe_print(f"{'='*60}")

        # 1. Build context: Novel Map (structure overview) + navigation (full text)
        context_text = build_context_index(line_num, 40)
        navigation_text = self._build_navigation(line_num, nav_range=25)

        # 2. Get evidence and memory
        evidence_text = self.vault.get_state_text(current_line=line_num)
        short_mem_text = self.short_mem.get_summary()

        # 3. Detect rapid exchange for extra warning
        if self.short_mem.detect_rapid_exchange(4):
            short_mem_text += ("\n\nRAPID EXCHANGE: Last 4+ dialogues alternate between two speakers.\n"
                               "RULE: Narrative attribution (#1) takes priority over alternation (#3).\n"
                               "Read 3-5 lines BEFORE the target to check for speech verbs.")

        # 4. Log boss task
        round_log["boss_task"] = {
            "dialogue_line": line_num,
            "dialogue_text": dialogue,
            "short_mem": short_mem_text,
            "evidence": evidence_text,
            "navigation": navigation_text,
        }

        # 5. Call Labeler (READ-ONLY, no state writing)
        speaker, summary, reason_text, pec, ec, tool_rounds_used = self.labeler.label(
            line_num, dialogue, short_mem_text, "",
            evidence_text, navigation_text, "",
            round_log, quiet=quiet, override_force_tool=True,
            recent_speakers_hint=""
        )
        self.total_tokens += pec + ec

        if not quiet:
            self._safe_print(f"  Labeler: {speaker} (tools={tool_rounds_used})")

        # 5b. Phase anchor check: did Labeler violate the alternating pattern?
        corrected = False
        next_exp = self.short_mem.get_next_expected()
        if next_exp and speaker != next_exp:
            self.short_mem.phase_violation = True
            if not quiet:
                self._safe_print(f"  Phase violation: expected {next_exp}, got {speaker}")
        else:
            self.short_mem.phase_violation = False

        # 6. SearchAgent: conditional trigger for temporary descriptors
        if speaker in TEMP_DESCRIPTORS:
            self.search_agent_triggers += 1
            if not quiet:
                try:
                    sys.stderr.write(f"  Temporary name: '{speaker}' -> triggering SearchAgent...\n")
                    sys.stderr.flush()
                except (ValueError, OSError):
                    pass

            search_result, s_pec, s_ec, s_tool_log = self.search_agent.investigate(
                speaker, line_num, max_tool_rounds=4, quiet=quiet
            )
            self.total_tokens += s_pec + s_ec

            if search_result["found"] and search_result["character"]:
                character = search_result["character"]
                aliases = search_result.get("aliases", [])
                evidence_list = search_result.get("evidence", [])
                status = search_result.get("status", "candidate")
                intro_line = search_result.get("introduction_line")

                self.vault.add_evidence(character, aliases, evidence_list, status, intro_line)
                if not quiet:
                    self._safe_print(f"  SearchAgent found: '{character}' ({status})")

                if status == "verified":
                    old_speaker = speaker
                    speaker = character
                    corrected = True
                    self.corrections += 1
                    if not quiet:
                        self._safe_print(f"  Corrected: '{old_speaker}' -> '{character}'")
            else:
                if not quiet:
                    self._safe_print(f"  SearchAgent: no identity found for '{speaker}'")

        # 7. Update EvidenceVault last_seen
        if speaker and speaker != "非人物发声" and speaker != "non-human":
            self.vault.update_last_seen(speaker, line_num)

        # 8. Fallback
        if not speaker:
            speaker = "?"

        # 9. Write to labeled.txt
        write_label(speaker)

        # 10. Update ShortMem with narrative context
        narr_before = get_narrative_before(line_num)
        self.short_mem.update(line_num, dialogue, speaker, reason_text, narrative_before=narr_before)

        # 11. Save vault periodically
        if self.round_count % 50 == 0:
            self.vault.save()

        round_log["result"] = {
            "speaker": speaker,
            "summary": summary,
            "reason": reason_text,
            "corrected": corrected,
        }

        return speaker


# ============================================================
# Validation
# ============================================================

def validate():
    if not os.path.exists(LABELED_PATH):
        print("labeled.txt not found")
        return 0, 0, 0

    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        labeled = [line.strip() for line in f.readlines() if line.strip()]

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answer_lines = f.readlines()

    answers = []
    answer_line_nums = []
    for i, line in enumerate(answer_lines):
        match = re.search(r"【([^】]+)】", line)
        if match:
            answers.append(match.group(1))
            answer_line_nums.append(i + 1)

    total = min(len(labeled), len(answers))
    correct = 0
    wrong = []

    for i in range(total):
        label = labeled[i]
        answer = answers[i]
        acceptable = answer.split("|")
        label_parts = label.split("|")
        if any(part in acceptable for part in label_parts):
            correct += 1
        else:
            wrong.append({
                "idx": i + 1,
                "answer_line": answer_line_nums[i] if i < len(answer_line_nums) else "?",
                "expected": answer,
                "got": label,
            })

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Validation Report")
    print(f"{'='*60}")
    print(f"  Total dialogues: {total}")
    print(f"  Correct:         {correct}")
    print(f"  Wrong:           {len(wrong)}")
    print(f"  Accuracy:        {accuracy:.1f}%")

    if wrong:
        print(f"\n  Error details (first 30):")
        for w in wrong[:30]:
            print(f"    #{w['idx']} (novel L{w['answer_line']}): expected={w['expected']} | got={w['got']}")

    return total, correct, len(wrong)


# ============================================================
# Main
# ============================================================

def _writeline(msg):
    """Write status line during annotation."""
    print(msg)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent novel dialogue speaker annotation v4")
    parser.add_argument("--start", type=int, default=0, help="Start from dialogue index (0=resume)")
    parser.add_argument("--count", type=int, default=1, help="Number of dialogues to annotate")
    parser.add_argument("--short-mem", type=int, default=20, help="Short-term memory rounds")
    parser.add_argument("--validate", action="store_true", help="Run validation after annotation")
    parser.add_argument("--reset-state", action="store_true", help="Reset character state before starting")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-agent Novel Dialogue Speaker Annotation v4")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Server: {OLLAMA_BASE_URL}")
    print(f"  Short-term memory: {args.short_mem} rounds")
    print(f"  Token budget: {TOKEN_BUDGET}")
    print(f"  Max tool rounds: {MAX_TOOL_ROUNDS}")

    # Roster check
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from check_roster import scan_py_files, DEFAULT_ROSTER
        violations = scan_py_files(SCRIPT_DIR, DEFAULT_ROSTER)
        if violations:
            print(f"\nWARNING: {len(violations)} character name leaks found in .py files:")
            for fp, lineno, word, line in violations[:10]:
                rel = os.path.relpath(fp, ROOT_DIR)
                print(f"  {rel}:L{lineno} matched '{word}'")
            if len(violations) > 10:
                print(f"  ... and {len(violations)-10} more")
            print("  Please remove novel-specific names for generality.\n")
        else:
            print("  Roster check: PASSED (no leaks)")
    except ImportError:
        pass

    print("=" * 60)

    # Reset state if requested
    if args.reset_state:
        print("  Resetting all state...")
        for p in [STATE_PATH, LOG_PATH, LABELED_PATH, VAULT_PATH]:
            if os.path.exists(p):
                os.remove(p)
        print("  State cleared. Starting fresh.")

    # Get dialogues
    dialogues = get_dialogue_list()
    labeled_count = get_labeled_count()

    print(f"\n  Total dialogues in novel: {len(dialogues)}")
    print(f"  Already labeled: {labeled_count}")
    print(f"  Remaining: {len(dialogues) - labeled_count}")

    start_idx = args.start if args.start > 0 else labeled_count
    end_idx = min(start_idx + args.count, len(dialogues))

    if start_idx >= len(dialogues):
        print("\n  All dialogues annotated!")
        return

    print(f"  This run: #{start_idx+1} to #{end_idx} ({end_idx - start_idx} dialogues)")

    def fmt_duration(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    start_time = time.time()
    total_rounds = end_idx - start_idx
    quiet_mode = total_rounds > 20

    boss = Boss(short_mem_rounds=args.short_mem)

    batch_tool_calls = 0
    batch_tokens = 0

    for idx in range(start_idx, end_idx):
        line_num, dialogue = dialogues[idx]
        round_log = new_round(idx - start_idx + 1, line_num, dialogue)
        speaker = boss.process_one(line_num, dialogue, round_log, quiet=quiet_mode)
        log_entry(round_log)

        batch_tool_calls += len(round_log.get("tool_calls", []))
        labeler = round_log.get("agents", {}).get("Labeler", {})
        batch_tokens += labeler.get("total_tokens", 0)

        done = idx - start_idx + 1
        elapsed = time.time() - start_time
        avg_sec = elapsed / done
        remaining_sec = avg_sec * (total_rounds - done)

        if quiet_mode:
            elapsed = time.time() - start_time
            avg_sec = elapsed / done
            remaining_sec = avg_sec * (total_rounds - done)
            _writeline(f"  [{done}/{total_rounds}] L{line_num} -> {speaker:<8s} | "
                       f"tools={batch_tool_calls:>3d} avg={avg_sec:.0f}s "
                       f"elapsed={fmt_duration(elapsed)} remaining={fmt_duration(remaining_sec)}")
        else:
            _writeline(f"  [{done}/{total_rounds}] L{line_num} -> {speaker:<8s} | "
                       f"tools={batch_tool_calls:>3d} avg={avg_sec:.0f}s "
                       f"elapsed={fmt_duration(elapsed)} remaining={fmt_duration(remaining_sec)}")

    if quiet_mode:
        _writeline("")

    _writeline(f"\n{'='*60}")
    _writeline(f"  Annotation complete")
    _writeline(f"  Dialogues annotated: {end_idx - start_idx}")
    _writeline(f"  Tool calls: {boss.labeler.tool_call_count}")
    _writeline(f"  SearchAgent triggers: {boss.search_agent_triggers}")
    _writeline(f"  Corrections: {boss.corrections}")
    _writeline(f"  Evidence vault: {VAULT_PATH}")
    _writeline(f"  Log: {LOG_PATH}")
    _writeline(f"{'='*60}")

    if args.validate:
        validate()


if __name__ == "__main__":
    main()
