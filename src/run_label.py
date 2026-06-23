# -*- coding: utf-8 -*-
"""
多 Agent 小说对话角色标注系统 v2

架构：
  Python 主控 → Boss → Labeler（+ 工具调用 + 角色状态表）
                       → ShortMem
                       → LongMem（累积式 + 结构化）

改进（v2）：
  1. 结构化角色状态表（JSON 字段，替代自由文本 LongMem）
  2. 别名系统独立维护（alias_map）
  3. Labeler prompt 强化工具调用引导
  4. 每轮每个 agent 的 token 消耗精确记录
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

# 从 config/ip_config 读取服务器配置（不暴露 IP 到 GitHub）
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


# ============================================================
# 日志记录
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
    """
    记录一个 Agent 的完整调用
    total_pec/total_ec: 如果有工具调用多次，传累计值；否则等于 pec/ec
    """
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
# 工具函数
# ============================================================

def read_novel_lines(start, count):
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = []
    for i in range(start - 1, min(start - 1 + count, len(lines))):
        result.append(f"{i+1}: {lines[i].rstrip()}")
    return "\n".join(result)


def get_dialogue_list():
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
# 工具定义
# ============================================================

TOOL_READ_NOVEL = {
    "type": "function",
    "function": {
        "name": "read_novel_lines",
        "description": "读取 novel.txt 的指定行范围。start 是起始行号(1-based)，count 是读取行数。每行以「行号: 内容」格式返回。",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "起始行号（从1开始）"},
                "count": {"type": "integer", "description": "要读取的行数"}
            },
            "required": ["start", "count"]
        }
    }
}


# ============================================================
# 结构化角色状态表 + 别名系统
# ============================================================

class CharacterState:
    """
    结构化角色状态表
    
    维护：
      - 角色别名映射（如：真名 = 临时称呼 = 身份称号），双重确认机制
      - 角色关系（如：主角A 与 主角B：同行/伙伴）
      - 最后出现行号 + 最近对话采样
    """
    
    def __init__(self):
        self.state = self._load_or_init()
    
    def _load_or_init(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "characters": {},
            "alias_map": {},
            "pending_aliases": {},  # 待确认的别名映射
        }
    
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
    
    def get_character(self, name):
        real_name = self.resolve_alias(name)
        return self.state["characters"].get(real_name)
    
    def resolve_alias(self, name):
        alias_map = self.state["alias_map"]
        if name in alias_map:
            return alias_map[name]
        for char_name, char_info in self.state["characters"].items():
            if name in char_info.get("aliases", []):
                return char_name
        return name
    
    def add_character(self, name, aliases=None, context=""):
        if name not in self.state["characters"]:
            self.state["characters"][name] = {
                "aliases": aliases or [name],
                "relations": {},
                "first_seen_line": None,
                "last_seen_line": None,
                "context_summary": context,
                "recent_lines": [],  # (line_num, text) 最近3-5句
            }
            for alias in (aliases or [name]):
                self.state["alias_map"][alias] = name
    
    def update_character(self, name, line_num, dialogue_text="", context=""):
        real_name = self.resolve_alias(name)
        if real_name not in self.state["characters"]:
            self.add_character(real_name, aliases=[real_name])
        
        char = self.state["characters"][real_name]
        if char["first_seen_line"] is None:
            char["first_seen_line"] = line_num
        char["last_seen_line"] = line_num
        if context:
            char["context_summary"] = context
        if dialogue_text:
            char["recent_lines"].append((line_num, dialogue_text))
            if len(char["recent_lines"]) > 5:
                char["recent_lines"].pop(0)
    
    def add_alias(self, real_name, alias, evidence="", is_identity_reveal=False):
        """
        添加角色别名（双重确认机制）
        - is_identity_reveal=True: 角色自报姓名，直接更新
        - is_identity_reveal=False: 推断性映射，先标记待确认
        """
        if is_identity_reveal:
            # 直接确认
            if real_name not in self.state["characters"]:
                self.add_character(real_name, aliases=[real_name])
            if alias not in self.state["characters"][real_name]["aliases"]:
                self.state["characters"][real_name]["aliases"].append(alias)
            self.state["alias_map"][alias] = real_name
            # 如有待确认项，一并清除
            self.state["pending_aliases"].pop(alias, None)
        else:
            # 待确认
            self.state["pending_aliases"][alias] = {
                "proposed": real_name,
                "evidence": evidence,
            }
    
    def confirm_pending_alias(self, alias):
        """确认一个待确认的别名（第二次独立确认后调用）"""
        pending = self.state["pending_aliases"].pop(alias, None)
        if pending:
            real_name = pending["proposed"]
            if real_name not in self.state["characters"]:
                self.add_character(real_name, aliases=[real_name])
            if alias not in self.state["characters"][real_name]["aliases"]:
                self.state["characters"][real_name]["aliases"].append(alias)
            self.state["alias_map"][alias] = real_name
    
    def add_relation(self, char_a, char_b, relation):
        real_a = self.resolve_alias(char_a)
        real_b = self.resolve_alias(char_b)
        for c in [real_a, real_b]:
            if c not in self.state["characters"]:
                self.add_character(c, aliases=[c])
        self.state["characters"][real_a]["relations"][real_b] = relation
        self.state["characters"][real_b]["relations"][real_a] = relation
    
    def get_active_characters(self, last_n_lines=50, current_line=0):
        active = []
        for name, info in self.state["characters"].items():
            ls = info.get("last_seen_line")
            if ls and (current_line - ls) <= last_n_lines:
                active.append(name)
        return active
    
    def get_state_text(self, current_line=0, scene_summary=""):
        """生成给 Labeler 看的角色状态文本"""
        lines = []
        active = self.get_active_characters(last_n_lines=50, current_line=current_line)
        
        lines.append("[角色状态]")
        for name in active:
            info = self.state["characters"].get(name, {})
            aliases = info.get("aliases", [name])
            relations = info.get("relations", {})
            rel_str = "; ".join([f"{k}({v})" for k, v in relations.items()])
            recent = info.get("recent_lines", [])
            
            alias_str = ", ".join(aliases)
            lines.append(f"  {name} (别名: {alias_str})")
            if rel_str:
                lines.append(f"    关系: {rel_str}")
            lines.append(f"    上轮行号: ~第{info.get('last_seen_line', '?')}行")
            if recent:
                recent_str = "; ".join([f"L{ln}「{t[:20]}」" for ln, t in recent[-3:]])
                lines.append(f"    最近对话: {recent_str}")
        
        if len(self.state["characters"]) > len(active):
            inactive = len(self.state["characters"]) - len(active)
            lines.append(f"  (其他 {inactive} 个角色非活跃)")
        
        return "\n".join(lines)
    
    def parse_state_update(self, text, line_num, dialogue=""):
        """从 Labeler 的 <state_update> 中解析状态更新"""
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
                char_name = m.group(1)
                field = m.group(2)
                value = m.group(3).strip()
                self.update_character(char_name, line_num, dialogue_text=dialogue)
                if field == "alias":
                    self.add_alias(char_name, value)
                elif field == "relation":
                    parts = value.split(":")
                    if len(parts) == 2:
                        self.add_relation(char_name, parts[0].strip(), parts[1].strip())
                continue
            
            # UPDATE alias: alias_name -> real_name
            m = re.match(r"UPDATE\s+alias:\s+(.+?)\s*->\s*(.+)", action_line)
            if m:
                alias = m.group(1).strip()
                real_name = m.group(2).strip()
                # 检查是否是身份揭露（self-introduction pattern）
                is_reveal = any(p in dialogue for p in ["名字是", "我叫", "咱是", "吾乃", "我是"])
                self.add_alias(real_name, alias, evidence=f"L{line_num}: {dialogue[:50]}", 
                              is_identity_reveal=is_reveal)
                continue
            
            # NEW character: name
            m = re.match(r"NEW\s+character:\s+(.+)", action_line)
            if m:
                char_name = m.group(1).strip()
                self.add_character(char_name, aliases=[char_name], context=f"首次出现于 L{line_num}")
                continue
    
    def get_scene_summary(self, current_line=0):
        """推断当前场景摘要"""
        active = self.get_active_characters(last_n_lines=30, current_line=current_line)
        if not active:
            return "[场景摘要] 初始场景"
        
        chars_str = ", ".join(active)
        # 从角色最近行号推断场景范围
        lines_list = [self.state["characters"][c]["last_seen_line"] for c in active 
                      if self.state["characters"][c].get("last_seen_line")]
        if lines_list:
            min_l = min(lines_list)
            max_l = max(lines_list)
            return f"[场景摘要] L{min_l}-{max_l} | 活跃角色: {chars_str}"
        return f"[场景摘要] 活跃角色: {chars_str}"


# ============================================================
# ShortMem Agent（原文+理由存储）
# ============================================================

class ShortMemAgent:
    def __init__(self, max_rounds=25):
        self.max_rounds = max_rounds
        self.history = []  # [(line_num, dialogue_text, speaker, reason)]
    
    def update(self, line_num, dialogue_text, speaker, reason=""):
        self.history.append((line_num, dialogue_text, speaker, reason))
        if len(self.history) > self.max_rounds:
            self.history.pop(0)
    
    def get_summary(self):
        if not self.history:
            return "（暂无历史标注记录）"
        lines = ["[最近对话历史 - 原文]"]
        for line_num, dlg, speaker, reason in self.history:
            line = f"  #{line_num} [L{line_num}] 「{dlg}」 → {speaker}"
            if reason:
                line += f"\n      理由: {reason[:120]}"
            lines.append(line)
        return "\n".join(lines)


# ============================================================
# LongMem Agent（累积式压缩，保留历史）
# ============================================================

class LongMemAgent:
    def __init__(self, compress_every=9999):
        self.compress_every = compress_every
        self.compression_count = 0
        self.long_memory_condensed = ""
        self.long_memory_detailed = ""
        self.pending_summaries = []
        self.compression_history = []
    
    def add_summary(self, line_num, speaker, summary):
        self.pending_summaries.append(f"#{line_num}: {speaker} | {summary}")
    
    def should_compress(self):
        return len(self.pending_summaries) >= self.compress_every
    
    def compress(self, round_log):
        if not self.pending_summaries:
            return
        
        history_text = "\n".join(self.pending_summaries)
        
        messages = [
            {
                "role": "system",
                "content": """你是一个小说分析记忆压缩助手。将最近的标注摘要压缩为长期记忆。

输出两个版本，用 ===CONDENSED=== 和 ===DETAILED=== 分隔：

===CONDENSED===
角色：
- 角色名：身份描述

主线：
- 当前情节进展

别名映射：
- A = B = C

===DETAILED===
角色：
- 角色名：身份描述，性格特点，关键信息
- 每个角色的详细描述

关键情节节点：
- L行号: 事件描述（列出所有重要事件）

别名状态表：
- 别名 = 真名（揭露行号）
- 别名 = 真名（未揭露则写"待确认"）"""
            },
            {
                "role": "user",
                "content": f"以下是最近的标注摘要：\n\n{history_text}\n\n请压缩为长期记忆，输出精简版和详细版。"
            }
        ]
        
        text, pec, ec, _ = call_ollama(messages, label="LongMem")
        
        log_agent(round_log, "LongMem", "compressor", messages, text, pec, ec)
        
        # 分割双版本
        parts = text.split("===DETAILED===")
        if len(parts) >= 2:
            condensed = parts[0].replace("===CONDENSED===", "").strip()
            detailed = "===DETAILED===".join(parts[1:]).strip()
        else:
            condensed = text
            detailed = text
        
        self.compression_history.append({"condensed": condensed, "detailed": detailed})
        # 保留最近3次
        if len(self.compression_history) > 3:
            self.compression_history.pop(0)
        
        # 精简版拼接最近3次
        condensed_parts = [h["condensed"] for h in self.compression_history]
        self.long_memory_condensed = "\n---\n".join(condensed_parts)
        
        # 详细版只保留最近1次
        self.long_memory_detailed = detailed
        

# ============================================================
# FactCurator Agent
# ============================================================

class FactCurator:
    def __init__(self, curator_every=10):
        self.curator_every = curator_every
        self.curation_count = 0
        self.pending_rounds = []
        self.fact_summary = "（暂无角色事实库）"
    
    def add_round(self, line_num, speaker, summary):
        self.pending_rounds.append({"line": line_num, "speaker": speaker, "summary": summary})
    
    def should_curate(self):
        return len(self.pending_rounds) >= self.curator_every
    
    def curate(self, short_mem_text, char_state_json, round_log):
        if not self.pending_rounds:
            return []
        summaries = "\n".join([f"#{r['line']}: {r['speaker']} | {r['summary']}" for r in self.pending_rounds])
        system_prompt = """你是一个角色数据库维护助手（FactCurator）。
输入：当前角色库 JSON + 近期对话原文和标注摘要
输出严格 JSON：{"updates":[{"action":"add_character"|"add_alias"|"add_relation"|"none","target":"角色名","evidence_line":行号,"detail":{...}}]}
规则：每个更新必须引用原文行号；不要编造信息；没有新发现输出 action: "none"。"""
        user_content = f"""当前角色库：
{json.dumps(char_state_json, ensure_ascii=False, indent=2)}

近期对话和标注：
{short_mem_text}

摘要：
{summaries}

检查是否有新的角色、别名或关系需要记录。没有则输出 action: "none"。"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        text, pec, ec, _ = call_ollama(messages, label="FactCurator")
        log_agent(round_log, "FactCurator", "curator", messages, text, pec, ec)
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                updates = json.loads(json_match.group()).get("updates", [])
                summary_lines = ["[角色事实库]"]
                chars = char_state_json.get("characters", {})
                if chars:
                    for name, info in chars.items():
                        aliases = info.get("aliases", [])
                        non_self = [a for a in aliases if a != name]
                        alias_str = f"（也称: {', '.join(non_self)}）" if non_self else ""
                        last = info.get("last_seen_line", "?")
                        rels = info.get("relations", {})
                        rel_str = ""
                        if rels:
                            rel_parts = [f"{k}:{v}" for k, v in rels.items()]
                            rel_str = f" [关系: {', '.join(rel_parts)}]"
                        summary_lines.append(f"  {name}{alias_str} (最近L{last}){rel_str}")
                else:
                    summary_lines.append("  （暂无角色记录）")
                self.fact_summary = "\n".join(summary_lines)
                self.pending_rounds = []
                self.curation_count += 1
                return updates
        except:
            pass
        self.pending_rounds = []
        return []
    
    def get_summary(self):
        return self.fact_summary
        self.pending_summaries = []
        self.compression_count += 1
    
    def get_memory(self):
        if not self.long_memory_condensed:
            return ("（暂无长期记忆）", "（暂无长期记忆）")
        return self.long_memory_condensed, self.long_memory_detailed


# ============================================================
# Labeler Agent（v2：强化工具调用 + 角色状态表）
# ============================================================

class LabelerAgent:
    def __init__(self):
        self.tool_call_count = 0
    
    def label(self, line_num, dialogue, short_mem_text, fact_summary, neighbor_context,
              char_state_text, navigation_text, scene_summary, round_log, quiet=False):
        """
        标注一条对话的说话人（无固定上下文，模型必须用工具读取原文）
        返回: (speaker, summary, reason, pec, ec)
        """
        system_prompt = """你是一个小说对话角色标注助手，在一个多 Agent 协作系统中工作。你的唯一任务是判断小说中指定对话的说话人。

======================================================================
一、你的工具 —— 这是你阅读小说的唯一途径
======================================================================

工具：read_novel_lines(start, count)
- 读取 novel.txt 从第 start 行开始的 count 行
- 返回格式：每行以「行号: 内容」格式呈现
- 适用范围：小说的任何行，不限调用次数，不限调用顺序

**你拿不到小说原文。你必须、也只能通过 read_novel_lines 工具来阅读小说。没有其他途径。**

======================================================================
二、标准工作流程（你必须遵守）
======================================================================

【阶段 1 — 初始阅读】
1. 你收到一条目标对话的行号和内容，以及系统提供的辅助信息（见第三节）
2. **第一步必须调用 read_novel_lines** 读取目标行附近的上下文，建议范围：目标行 ±40 行
   （如果目标行靠近小说开头或结尾，相应地调整起始行）
3. 仔细阅读返回的原文，关注以下线索：
   - 目标行之前的叙事句：是否明确说出"XX说"、"XX开口道"、"XX喊道"等
   - 目标行之后的叙事句：是否事后追溯"刚才是XX说的"
   - 对话的语气、措辞习惯是否符合某个已出场角色的说话风格

【阶段 2 — 深度搜索（如果需要）】
4. 如果第一阶段获得的信息不足以确定说话人，进行深度搜索：
   a. 先扩展阅读范围（±60-80 行），获取更多上下文
   b. 如果该角色是首次在这个区域出现，往回跳跃搜索该角色首次出场的段落
      - 角色名字通常在首次出场时有描述性介绍
      - 真名揭露通常出现在"名字是""我叫""咱是""吾乃""我是"等句式附近
      - 别名（如"女孩""商人""骑士"）和真名之间的对应关系，
        在读到真名揭露的段落之前，只能用当时语境中使用的称呼
   c. 如果场景切换了（如从室内转到户外、从对话转到战斗），注意说话人群可能也变了

【阶段 3 — 输出标注结果】
5. 在你确定说话人并且不再需要更多工具调用时，输出最终答案，
   严格按第五节格式输出，四个标签都必须有。

======================================================================
三、辅助信息 —— 可信度分级
======================================================================

系统会在 User Message 中提供辅助信息。不同信息的可信度不同：

可直接使用（100% 准确）：
  - 原文导航中的行号和内容片段（来自 novel.txt）
  - 短期记忆中的原文对话文本

参考使用（以原文验证为准）：
  - 短期记忆中的说话人标注（以原文叙事标记为最终权威）
  - 角色事实库中的角色描述

核心原则：原文 > 辅助信息。当原文中的叙事标记与辅助信息冲突时，以原文为准。

======================================================================
四、标注规则（详细版）
======================================================================

【规则 1：说话人名称 —— 使用原文中实际出现的称呼】
- 有具体名字 → 使用具体名字
- 如果没有姓名但上下文有身份或群体，请用中文身份词，例如：村民、骑士、店员、商人、众人、未知。
- 不要用临时行为关系替代更稳定身份；例如上下文说明是村落居民时，用"村民"而不是"顾客"。
- 群体对话（如集体呼喊、众人齐声）也用群体称呼，不要标为"非人物发声"。
- 临时身份词（如"女孩""少年""老人""大汉"等描述具体个体的称呼）：去后文有限范围搜索稳定姓名或固定称呼，找到则使用
- 群体称呼（如"村民""骑士""众人""店员"等）：不需要去后文搜索姓名，直接使用该群体称呼即可
- **搜索阈值：仅对明显是可追踪的具体个体的临时称呼向前搜索，对群体称呼和一次性路人不要无限搜索**

【规则 2：叙事标记优先于推理】
- "XX说""XX开口道""XX喊道""XX回答"这类叙事句中的称呼是权威信号，直接确定说话人
- 对话交替规律只是辅助推测手段，必须让位于叙事标记
- 同一角色可能连续说多句，不强行假设交替

【规则 3：非人物发声（仅在以下情况使用）】
- 环境声、物体声音、心理比喻声或声音效果 → 标注为"非人物发声"
- 如果文本明确说明某个角色发出该声音（如喊叫、叹息、笑声、嚎叫），仍标注该角色
- 群体呼喊、集体对话不是"非人物发声"

【规则 4：角色不在场 / 新角色】
- 区分"谁在说话"和"在说谁"——对话中提到的角色名未必是说话人
- 如果出现角色状态表中没有的角色，如实使用原文中的称呼标注

======================================================================
五、输出格式（严格遵循）
======================================================================

你必须输出以下四个标签，缺一不可：

<answer>说话人名字</answer>
- 单个角色的具体名字，或"非人物发声"
- 不要用"|"分隔多个名字

<reason>判断依据</reason>
- 必须引用具体的行号作为证据
- 写明你读了哪些行、从哪些叙事句中得出了结论
- 如果与短期记忆或角色状态表的信息有矛盾，在这里说明
- 如果判断中存在不确定性，也在这里说明

<summary>一句话摘要</summary>
- 格式：说话人 | 做了什么/说了什么
- 用于后续轮次的短期记忆，帮助系统追踪对话流转
- 保持简洁，一句话概括

<state_update>
UPDATE character: 说话人.last_seen_line = {line_num}
（仅当本轮有新发现时才追加：）
（UPDATE alias: 别名 -> 真名  —— 仅当原文中明确揭露了角色身份）
（UPDATE relation: 角色A:关系描述 —— 仅当原文中明确了角色间关系）
（NEW character: 角色名 —— 仅当本轮发现了 CharacterState 中不存在的新角色）
</state_update>

【state_update 写入规则】
- 仅写角色名，不附加描述"""
        
        user_content = f"""请标注以下对话的说话人：

第{line_num}行：「{dialogue}」

====================================================================
以下辅助信息由系统其他 Agent 生成。原文导航中的行号和内容片段来自小说原文，100% 准确。
短期记忆中的说话人标注以及角色事实库仅供参考，以你通过 read_novel_lines 读到的原文为准。
====================================================================

{scene_summary}

[原文导航 - 目标行附近]
{navigation_text}

{neighbor_context}

{short_mem_text}

[角色事实库]
{fact_summary}

{char_state_text}

====================================================================
我没有把小说原文直接给你。你必须调用 read_novel_lines 工具来阅读小说。
第一步请以目标行 ±40 行为范围开始阅读，然后根据需要进行深度搜索。
如果你认为辅助信息中的某个判断有误，请在 <reason> 中明确指出。
===================================================================="""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        total_pec = 0
        total_ec = 0
        tool_call_log = []
        max_tool_rounds = 10
        
        for round_i in range(max_tool_rounds):
            text, pec, ec, tool_calls = call_ollama(messages, tools=[TOOL_READ_NOVEL], label=f"Labeler-R{round_i+1}")
            total_pec += pec
            total_ec += ec
            
            if not tool_calls:
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
                        print(f"    🔧 工具调用 #{self.tool_call_count}: read_novel_lines({s}-{s+c-1}) → {len(result)} 字符, pec={pec}, ec={ec}")
                    messages.append({"role": "tool", "content": result})
        else:
            if not quiet:
                print(f"    ⚠️ 达到最大工具调用次数 {max_tool_rounds}")
        
        round_log["tool_calls"] = tool_call_log
        
        speaker = self._parse_answer(text)
        summary = self._parse_summary(text)
        reason = self._parse_reason(text)
        
        return speaker, summary, reason, total_pec, total_ec
    
    def _parse_answer(self, text):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip().split("\n")[0][:50]
    
    def _parse_summary(self, text):
        match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _parse_reason(self, text):
        """解析 <reason> 标签内容"""
        match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if match:
            return match.group(1).strip()[:200]
        return ""


# ============================================================
# Boss（Python 脚本）
# ============================================================

class Boss:
    def __init__(self, short_mem_rounds=25, long_mem_compress_every=9999, fact_curator_every=10):
        self.labeler = LabelerAgent()
        self.short_mem = ShortMemAgent(max_rounds=short_mem_rounds)
        self.long_mem = LongMemAgent(compress_every=long_mem_compress_every)
        self.fact_curator = FactCurator(curator_every=fact_curator_every)
        self.char_state = CharacterState()
        self.dialogue_list = get_dialogue_list()
        self.total_tokens = 0
        self.round_count = 0
    
    def _build_navigation(self, line_num, nav_range=20):
        """构建原文导航：目标行附近标注过的行显示结果，未标注的标记为叙事"""
        lines = []
        start = max(1, line_num - nav_range)
        end = line_num + nav_range
        novel_lines = read_novel_lines(start, end - start + 1).split("\n")
        
        # 建立行号→标注结果的映射
        labeled_map = {}
        dialog_map = {}
        for ln, dlg in self.dialogue_list:
            dialog_map[ln] = dlg
            if ln >= start and ln <= end:
                # 检查 short_mem 中是否有该行的标注
                for h_ln, h_dlg, h_sp, h_reason in self.short_mem.history:
                    if h_ln == ln:
                        labeled_map[ln] = (h_sp, h_reason)
        
        for line_text in novel_lines:
            if not line_text.strip():
                continue
            # 格式: "行号: 内容"
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
                marker = "→"  # 当前待标注行
            
            if ln in labeled_map:
                sp, reason = labeled_map[ln]
                lines.append(f"  {marker}L{ln}: 「{dialog_map.get(ln, content[:40])}»")
            elif "「" in content:
                lines.append(f"  {marker}L{ln}: 「{content[:50]}」")
            else:
                lines.append(f"  {marker}L{ln}: [叙事] {content[:60]}")
        
        return "\n".join(lines)
    
    def process_one(self, line_num, dialogue, round_log, quiet=False):
        self.round_count += 1
        
        if not quiet:
            print(f"\n{'='*60}")
            print(f"  第 {self.round_count} 轮标注")
            print(f"  对话：第{line_num}行「{dialogue}」")
            print(f"{'='*60}")
        
        # 1. 获取记忆
        short_mem_text = self.short_mem.get_summary()
        fact_summary = self.fact_curator.get_summary()
        char_state_text = self.char_state.get_state_text(current_line=line_num)
        scene_summary = self.char_state.get_scene_summary(current_line=line_num)
        
        # 2. 构建原文导航
        navigation_text = self._build_navigation(line_num, nav_range=20)
        neighbor_context = ""
        
        # 记录 Boss 任务
        round_log["boss_task"] = {
            "dialogue_line": line_num,
            "dialogue_text": dialogue,
            "short_mem": short_mem_text,
            "fact_summary": fact_summary,
            "neighbor_context": neighbor_context,
            "char_state": char_state_text,
            "navigation": navigation_text,
        }
        
        # 3. 调用 Labeler
        speaker, summary, reason_text, pec, ec = self.labeler.label(
            line_num, dialogue, short_mem_text, fact_summary, neighbor_context,
            char_state_text, navigation_text, scene_summary, round_log,
            quiet=quiet
        )
        self.total_tokens += pec + ec
        
        if not quiet:
            print(f"  📝 标注结果: {speaker}")
        
        # 4. 更新记忆（存原文+理由）
        self.short_mem.update(line_num, dialogue, speaker, reason_text)
        self.long_mem.add_summary(line_num, speaker, summary)
        
        # 5. 更新角色状态
        if speaker != "非人物发声":
            self.char_state.update_character(speaker, line_num, dialogue_text=dialogue)
        labeler_response = round_log["agents"].get("Labeler", {}).get("response", "")
        self.char_state.parse_state_update(labeler_response, line_num, dialogue=dialogue)
        self.char_state.save(round_num=self.round_count)
        
        # 6. FactCurator：结构化事实库维护
        self.fact_curator.add_round(line_num, speaker, summary)
        if self.fact_curator.should_curate():
            if not quiet:
                print(f"  🔄 触发 FactCurator 角色事实库更新...")
            fc_mem = self.short_mem.get_summary()
            char_state_json = self.char_state.state
            updates = self.fact_curator.curate(fc_mem, char_state_json, round_log)
            self.char_state.save(round_num=self.round_count)
            if not quiet:
                print(f"  📝 角色事实库已更新")
        
        round_log["result"] = {
            "speaker": speaker,
            "summary": summary,
            "reason": reason_text,
        }
        
        return speaker


# ============================================================
# 验证
# ============================================================

def validate():
    if not os.path.exists(LABELED_PATH):
        print("labeled.txt 不存在，无法验证")
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
    print(f"  验证报告")
    print(f"{'='*60}")
    print(f"  总对话数:    {total}")
    print(f"  正确数:      {correct}")
    print(f"  错误数:      {len(wrong)}")
    print(f"  正确率:      {accuracy:.1f}%")
    
    if wrong:
        print(f"\n  错误详情（前30条）:")
        for w in wrong[:30]:
            print(f"    #{w['idx']} (小说第{w['answer_line']}行): 期望={w['expected']} | 实际={w['got']}")
    
    return total, correct, len(wrong)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多 Agent 小说对话角色标注系统 v2")
    parser.add_argument("--start", type=int, default=0, help="从第几条对话开始（默认0=从上次继续）")
    parser.add_argument("--count", type=int, default=1, help="标注几条对话（默认1）")
    parser.add_argument("--short-mem", type=int, default=15, help="短期记忆轮数（默认15）")
    parser.add_argument("--long-mem-every", type=int, default=5, help="长期记忆压缩频率（默认每5轮）")
    parser.add_argument("--validate", action="store_true", help="标注完成后验证")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  多 Agent 小说对话角色标注系统 v2")
    print(f"  模型: {OLLAMA_MODEL}")
    print(f"  服务器: {OLLAMA_BASE_URL}")
    print(f"  策略: 无固定上下文，模型通过 read_novel_lines 工具自主读取")
    print(f"  短期记忆: {args.short_mem}轮")
    print(f"  长期记忆压缩: 每{args.long_mem_every}轮")
    print(f"  日志文件: {LOG_PATH}")
    print(f"  角色状态: {STATE_PATH}")
    print("=" * 60)
    
    # 获取所有对话
    dialogues = get_dialogue_list()
    labeled_count = get_labeled_count()
    
    print(f"\n  小说总对话数: {len(dialogues)}")
    print(f"  已标注: {labeled_count}")
    print(f"  未标注: {len(dialogues) - labeled_count}")
    
    start_idx = args.start if args.start > 0 else labeled_count
    end_idx = min(start_idx + args.count, len(dialogues))
    
    if start_idx >= len(dialogues):
        print("\n  所有对话已标注完毕！")
        return
    
    print(f"  本次标注: 第 {start_idx+1} 到 {end_idx} 条（共 {end_idx - start_idx} 条）")
    
    # 进度显示辅助
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
    
    boss = Boss(
        short_mem_rounds=args.short_mem,
        long_mem_compress_every=args.long_mem_every
    )
    
    batch_tool_calls = 0
    batch_tokens = 0
    
    for idx in range(start_idx, end_idx):
        line_num, dialogue = dialogues[idx]
        round_log = new_round(idx - start_idx + 1, line_num, dialogue)
        speaker = boss.process_one(line_num, dialogue, round_log, quiet=quiet_mode)
        write_label(speaker)
        log_entry(round_log)
        
        # 统计
        batch_tool_calls += len(round_log.get("tool_calls", []))
        labeler = round_log.get("agents", {}).get("Labeler", {})
        batch_tokens += labeler.get("total_tokens", 0)
        
        # 进度显示
        done = idx - start_idx + 1
        elapsed = time.time() - start_time
        avg_sec = elapsed / done
        remaining_sec = avg_sec * (total_rounds - done)
        
        # 一行进度，不刷屏
        if quiet_mode:
            progress_bar = f"[{'█' * (done * 20 // total_rounds):20s}]"
            line = (f"\r  {done:>4d}/{total_rounds:<4d} {progress_bar} "
                    f"L{line_num:<5d} → {speaker:<6s} "
                    f"⚙{batch_tool_calls:>3d} "
                    f"TOK{batch_tokens:>8d} "
                    f"⏱{avg_sec:.0f}s/轮 "
                    f"已过{fmt_duration(elapsed)} "
                    f"预计剩余{fmt_duration(remaining_sec)}")
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            print(f"  [{done}/{total_rounds}] L{line_num} → {speaker:<6s} | "
                  f"⚙{batch_tool_calls:>3d} ⏱{avg_sec:.0f}s/avg "
                  f"已过{fmt_duration(elapsed)} 预计剩余{fmt_duration(remaining_sec)}")
    
    # 安静模式结束，换行收尾
    if quiet_mode:
        print()
    
    print(f"\n{'='*60}")
    print(f"  标注完成")
    print(f"  本次标注: {end_idx - start_idx} 条")
    print(f"  工具调用次数: {boss.labeler.tool_call_count}")
    print(f"  长期记忆压缩: {boss.long_mem.compression_count} 次")
    print(f"  日志已保存: {LOG_PATH}")
    print(f"{'='*60}")
    
    if args.validate:
        validate()


if __name__ == "__main__":
    main()