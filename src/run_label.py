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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
        return {"characters": {}, "alias_map": {}}

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
        self.history = []  # [(line_num, dialogue_text, speaker, reason)]

    def update(self, line_num, dialogue_text, speaker, reason=""):
        self.history.append((line_num, dialogue_text, speaker, reason))
        if len(self.history) > self.max_rounds:
            self.history.pop(0)

    def _get_exchange_rhythm(self):
        """Detect rapid 2-person exchange pattern and return a hint."""
        if len(self.history) < 4:
            return None
        recent = self.history[-8:]  # last 8 rounds max
        speakers = [sp for _, _, sp, _ in recent]

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
            return None

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
            return (f"[Exchange rhythm - last {len(speakers)} rounds]\n"
                    f"  Alternating: {a} ↔ {b}\n"
                    f"  Current expects: {next_expected} (opposite of previous speaker)")
        return None

    def get_summary(self):
        if not self.history:
            return "(No prior annotations)"
        lines = ["[Recent Dialogue History - Original Text]"]
        for line_num, dlg, speaker, reason in self.history:
            line = f"  #{line_num} [L{line_num}] 「{dlg}」 -> {speaker}"
            if reason:
                line += f"\n      reason: {reason[:120]}"
            lines.append(line)
        # Add exchange rhythm hint
        rhythm = self._get_exchange_rhythm()
        if rhythm:
            lines.append("")
            lines.append(rhythm)
        return "\n".join(lines)


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
               override_force_tool=True):
        """
        Label one dialogue's speaker. Returns (speaker, summary, reason, pec, ec).
        """

        system_prompt = """You are a dialogue speaker annotation assistant. Your ONLY task is to identify who speaks a given line of dialogue in a Chinese novel.

========================================
YOUR TOOL - Your ONLY source for novel text
========================================

Tool: read_novel_lines(start, count)
- Reads 'count' lines from novel.txt starting at line 'start' (1-based)
- Returns: each line as "line_number: content"
- You can call this multiple times, in any order, for any range

**You do NOT have the novel text. You MUST use read_novel_lines to read it. There is no other way.**

========================================
WORKFLOW
========================================

[Phase 1 - Initial Read]
1. You receive: target dialogue line number + text, plus auxiliary info
2. FIRST: call read_novel_lines for the target area. Recommended: target line +/- 40 lines
3. Read carefully. Look for:
   - Narrative BEFORE the dialogue: "XX说", "XX喊道", "XX开口", "XX回答"
   - Narrative AFTER the dialogue: "刚才是XX说的", "XX说完"
   - Dialogue tone/voice matching a known character

[Phase 2 - Deep Search (if needed)]
4. If Phase 1 is not enough:
   a. Expand range (+/- 60-80 lines) for more context
   b. If a character appears for the first time, search BACK for their introduction
      - Real names appear near: "我叫XX", "咱是XX", "名字是XX", "吾乃XX"
      - Before a real name is revealed, use whatever the text uses
   c. If scene changed, the speaking group may have changed too
   d. **CRITICAL - Unnamed descriptor**: If you see a temporary descriptor like "女孩"/"少年"/"老人"/"大汉"
      that describes ONE specific person, you MUST search FORWARD 150-300 lines for a real name reveal.
      Look for self-introduction patterns. If found, USE THE REAL NAME.
      If not found after 300 lines, use the descriptor.
   e. **Token budget**: If cumulative tool call tokens exceed ~15K, STOP searching and output best guess.

[Phase 3 - Output]
5. When you are done searching, output the final answer in the format below.

========================================
ANNOTATION RULES（标注规则 - 关键）
========================================

【规则1：说话人名称——用原文实际使用的称呼】
- 有具体姓名 → 用具体姓名
- 无姓名但有身份/群体 → 用身份称呼：村民、骑士、商人、众人、未知
- 群体呼喊、集体对话 → 用群体称呼，不要标"非人物发声"
- 临时描述词（女孩、少年等）→ 必须前向搜索真名（见Phase 2d）
- 群体称呼（村民、骑士、众人）→ 直接使用，无需搜索

【规则2：叙事标记优先于推理】
- "XX说""XX喊道"等叙事句中的称呼是权威证据，直接使用
- 对话交替规律只是辅助线索，不能替代叙事标记
- 同一角色可以连续说多句，不要强行假设交替

【规则3：非人物发声——仅用于以下情况】
- 环境声、物体声、音效 → "非人物发声"
- 如果文本明确说明某个角色发出了声音（喊叫、叹息、嚎叫），标注该角色
- 群体呼喊不是"非人物发声"

【规则4：新角色/角色不在状态表】
- 区分"谁在说话"和"在说谁"——被提到的角色名未必是说话人
- 如果角色不在状态表中，如实使用原文中的称呼

【规则5：快速对话交换——防止角色混淆】
- 当两个角色快速交换短对话（每句<15字）时：
  a. 先查看原文中是否有"XX说"等叙事标记
  b. 如果没有叙事标记，用最近标注记录追踪轮换顺序
  c. **特别注意**：快速交换中同一角色不太可能连续说多句
  d. 如果最近几轮中两个角色各说了几句，按此交替规律推理
  e. 叙事标记始终覆盖交替规律

========================================
OUTPUT FORMAT（严格 - 4个标签都必须有）
========================================

<answer>speaker name</answer>
- 单个角色名，或"非人物发声"
- 不要用"|"分隔多个名字

<reason>your reasoning in Chinese</reason>
- 引用具体的行号作为证据
- 说明使用了哪些叙事标记或上下文线索

<summary>一句话摘要</summary>
- 格式：说话人 | 做了什么/说了什么
- 保持简洁，用于短期记忆

<state_update>
UPDATE character: speaker.last_seen_line = {line_num}
(add only if new discovery:)
(UPDATE alias: alias_name -> real_name  -- only if text explicitly reveals identity)
(NEW character: character_name  -- only if CharacterState doesn't have this character)
</state_update>

RULES for state_update:
- Character names ONLY - no descriptions, no annotations
- CORRECT: NEW character: 骑士
- FORBIDDEN: NEW character: some_name -- according to L123
- FORBIDDEN: NEW character: 骑士 (a new character)
- If you append any description after the name, the entry will be rejected"""

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

[Character State]
{char_state_text}

========================================
You do NOT have the novel text. You MUST call read_novel_lines to read it.
Start with target line +/- 40 lines, then search deeper as needed.
If you think an auxiliary label is wrong, say so in <reason>.
========================================"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        total_pec = 0
        total_ec = 0
        tool_call_log = []
        cumulative_tokens = 0

        for round_i in range(MAX_TOOL_ROUNDS):
            text, pec, ec, tool_calls = call_ollama(messages, tools=[TOOL_READ_NOVEL], label=f"Labeler-R{round_i+1}")
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
                if func.get("name") == "read_novel_lines":
                    s = func_args.get("start", 1)
                    c = func_args.get("count", 10)
                    result = read_novel_lines(s, c)
                    self.tool_call_count += 1
                    if not quiet:
                        print(f"    Tool call #{self.tool_call_count}: read_novel_lines({s}-{s+c-1}) -> {len(result)} chars, pec={pec}, ec={ec}")
                    messages.append({"role": "tool", "content": result})

            # Token budget check
            if cumulative_tokens >= TOKEN_BUDGET:
                if not quiet:
                    print(f"    Token budget reached ({cumulative_tokens} >= {TOKEN_BUDGET}), stopping search")
                # Force final answer on next call
                messages.append({"role": "user", "content": "You have enough information. Now output your final answer with <answer>, <reason>, <summary>, and <state_update> tags."})
                continue

        else:
            if not quiet:
                print(f"    Max tool rounds reached ({MAX_TOOL_ROUNDS})")

        round_log["tool_calls"] = tool_call_log

        speaker = self._parse_answer(text)
        summary = self._parse_summary(text)
        reason = self._parse_reason(text)

        return speaker, summary, reason, total_pec, total_ec

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

        # Allow up to 3 tool rounds for the verifier
        for round_i in range(3):
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
                print(f"  ⚠️ Verifier disagrees: Labeler={labeler_speaker} -> Verifier={suggested}")
            else:
                print(f"  ✓ Verifier confirms: {labeler_speaker}")

        return verdict, suggested, reason


# ============================================================
# Boss (Python orchestrator)
# ============================================================

class Boss:
    def __init__(self, short_mem_rounds=20, fact_curator_every=10):
        self.labeler = LabelerAgent()
        self.verifier = VerifierAgent()
        self.short_mem = ShortMemAgent(max_rounds=short_mem_rounds)
        self.fact_curator = FactCurator(curator_every=fact_curator_every)
        self.char_state = CharacterState()
        self.dialogue_list = get_dialogue_list()
        self.total_tokens = 0
        self.verifier_disagreements = 0
        self.round_count = 0

    def _build_navigation(self, line_num, nav_range=20):
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
                marker = "->"

            if "「" in content:
                lines.append(f"  {marker}L{ln}: 「{content[:60]}」")
            else:
                lines.append(f"  {marker}L{ln}: [narrative] {content[:70]}")

        return "\n".join(lines)

    def _is_swap_error(self, labeler_speaker, verifier_speaker, short_mem_text):
        """Heuristic: if two characters have been alternating and Verifier suggests the opposite, trust Verifier."""
        # Simple check: if short_mem shows alternation pattern and Verifier suggests the opposite speaker
        if "Exchange rhythm" in short_mem_text and "Alternating" in short_mem_text:
            return True
        return False

    def process_one(self, line_num, dialogue, round_log, quiet=False):
        self.round_count += 1

        if not quiet:
            print(f"\n{'='*60}")
            print(f"  Round {self.round_count}")
            print(f"  Dialogue: L{line_num}「{dialogue}」")
            print(f"{'='*60}")

        # 1. Get memory and state
        short_mem_text = self.short_mem.get_summary()
        fact_summary = self.fact_curator.get_summary()
        char_state_text = self.char_state.get_state_text(current_line=line_num)
        scene_summary = self.char_state.get_scene_summary(current_line=line_num)

        # 2. Build navigation
        navigation_text = self._build_navigation(line_num, nav_range=20)

        # 3. Log boss task
        round_log["boss_task"] = {
            "dialogue_line": line_num,
            "dialogue_text": dialogue,
            "short_mem": short_mem_text,
            "fact_summary": fact_summary,
            "char_state": char_state_text,
            "navigation": navigation_text,
        }

        # 4. Call Labeler
        speaker, summary, reason_text, pec, ec = self.labeler.label(
            line_num, dialogue, short_mem_text, fact_summary,
            char_state_text, navigation_text, scene_summary,
            round_log, quiet=quiet, override_force_tool=True
        )
        self.total_tokens += pec + ec

        if not quiet:
            print(f"  Labeler: {speaker}")

        # 5. Verifier Agent: independent cross-check
        verdict, verifier_speaker, verifier_reason = self.verifier.verify(
            line_num, dialogue, navigation_text, short_mem_text,
            scene_summary, char_state_text, speaker, round_log, quiet=quiet
        )

        if verdict == "disagree" and verifier_speaker != speaker:
            self.verifier_disagreements += 1
            if not quiet:
                print(f"  🔄 Labeler re-annotating based on Verifier feedback...")
            # Ask Labeler to re-annotate with Verifier's concern
            recheck_prompt = f"The Verifier suggests '{verifier_speaker}' instead of '{speaker}'. Read the novel text carefully and reconsider. Evidence: {verifier_reason}"
            speaker, summary, reason_text, pec2, ec2 = self.labeler.label(
                line_num, dialogue, short_mem_text, fact_summary,
                char_state_text, navigation_text, scene_summary,
                round_log, quiet=True, override_force_tool=False
            )
            self.total_tokens += pec2 + ec2
            if not quiet:
                print(f"  Labeler after re-check: {speaker}")
            # Use Verifier's answer if Labeler still disagrees
            if speaker == verifier_speaker or self._is_swap_error(speaker, verifier_speaker, short_mem_text):
                speaker = verifier_speaker

        # 6. Python-side forward search for temporary descriptors
        if speaker in TEMP_DESCRIPTORS:
            resolved = self.char_state.resolve_alias(speaker)
            if resolved == speaker:  # No alias yet
                real_name, reveal_line = search_forward_for_name(line_num, speaker)
                if real_name:
                    if not quiet:
                        print(f"  🔍 Forward search: '{speaker}' -> '{real_name}' at L{reveal_line}")
                    self.char_state.add_alias(real_name, speaker, is_identity_reveal=True)
                    # Update fact_summary with this discovery
                    speaker = real_name  # Override with real name!

        # 6. Update memory
        self.short_mem.update(line_num, dialogue, speaker, reason_text)

        # 6. Update character state
        if speaker != "non-human" and speaker != "非人物发声":
            clean_speaker = validate_char_name(speaker)
            if clean_speaker:
                self.char_state.update_character(clean_speaker, line_num, dialogue_text=dialogue)

        labeler_response = round_log["agents"].get("Labeler", {}).get("response", "")
        self.char_state.parse_state_update(labeler_response, line_num, dialogue=dialogue)
        self.char_state.save(round_num=self.round_count)

        # 7. FactCurator: every N rounds
        self.fact_curator.add_round(line_num, speaker, summary)
        if self.fact_curator.should_curate():
            if not quiet:
                print(f"  FactCurator: maintaining character facts...")
            fc_mem = self.short_mem.get_summary()
            char_state_json = self.char_state.state
            self.fact_curator.curate(fc_mem, char_state_json, round_log)
            self.char_state.save(round_num=self.round_count)
            if not quiet:
                print(f"  Character facts updated.")

        round_log["result"] = {
            "speaker": speaker,
            "summary": summary,
            "reason": reason_text,
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
        print("  Resetting character state...")
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        if os.path.exists(LABELED_PATH):
            os.remove(LABELED_PATH)
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
        write_label(speaker)
        log_entry(round_log)

        batch_tool_calls += len(round_log.get("tool_calls", []))
        labeler = round_log.get("agents", {}).get("Labeler", {})
        batch_tokens += labeler.get("total_tokens", 0)

        done = idx - start_idx + 1
        elapsed = time.time() - start_time
        avg_sec = elapsed / done
        remaining_sec = avg_sec * (total_rounds - done)

        if quiet_mode:
            progress_bar = f"[{'#' * (done * 20 // total_rounds):20s}]"
            line = (f"\r  {done:>4d}/{total_rounds:<4d} {progress_bar} "
                    f"L{line_num:<5d} -> {speaker:<8s} "
                    f"tools={batch_tool_calls:>3d} "
                    f"tokens={batch_tokens:>8d} "
                    f"avg={avg_sec:.0f}s/round "
                    f"elapsed={fmt_duration(elapsed)} "
                    f"remaining={fmt_duration(remaining_sec)}")
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            print(f"  [{done}/{total_rounds}] L{line_num} -> {speaker:<8s} | "
                  f"tools={batch_tool_calls:>3d} avg={avg_sec:.0f}s "
                  f"elapsed={fmt_duration(elapsed)} remaining={fmt_duration(remaining_sec)}")

    if quiet_mode:
        print()

    print(f"\n{'='*60}")
    print(f"  Annotation complete")
    print(f"  Dialogues annotated: {end_idx - start_idx}")
    print(f"  Tool calls: {boss.labeler.tool_call_count}")
    print(f"  Log: {LOG_PATH}")
    print(f"{'='*60}")

    if args.validate:
        validate()


if __name__ == "__main__":
    main()
