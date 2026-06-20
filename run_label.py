# -*- coding: utf-8 -*-
"""
多 Agent 小说对话角色标注系统

架构：
  Python 脚本（主控）→ Boss Agent → Labeler Agent（+ 工具调用）
                                     → ShortMem Agent
                                     → LongMem Agent
  Python 脚本 ← 拿到标注结果 → write_label.py 写入

原则：
  1. Python 控制流程，Agent 只读不写
  2. 每次只标注 1 条对话
  3. Agent 没有 write 工具，只能通过返回结果让 Python 写入
  4. 完整记录每个 Agent 的输入输出和 Token 消耗
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import re
import json
import requests
import argparse
from datetime import datetime

OLLAMA_BASE_URL = "http://172.31.102.237:11434"
OLLAMA_MODEL = "qwen3:32b"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOVEL_PATH = os.path.join(SCRIPT_DIR, "novel.txt")
LABELED_PATH = os.path.join(SCRIPT_DIR, "labeled.txt")
ANSWERS_PATH = os.path.join(SCRIPT_DIR, "answers.txt")
LOG_PATH = os.path.join(SCRIPT_DIR, "label_log.jsonl")


# ============================================================
# 日志记录
# ============================================================

def log_entry(entry):
    """追加一条日志到 label_log.jsonl"""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def new_round(round_idx, line_num, dialogue):
    """创建新一轮标注的日志骨架"""
    return {
        "round": round_idx,
        "dialogue_line": line_num,
        "dialogue_text": dialogue,
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "tool_calls": [],
        "result": None,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }


def log_agent(round_log, agent_name, role, messages, response_text, pec, ec, tool_calls=None):
    """记录一个 Agent 的完整调用"""
    entry = {
        "agent": agent_name,
        "role": role,
        "input_messages": messages,
        "input_summary": messages[-1]["content"][:200] if messages else "",
        "response": response_text,
        "prompt_eval_count": pec,
        "eval_count": ec,
        "total_tokens": pec + ec,
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    
    round_log["agents"][agent_name] = entry
    round_log["total_input_tokens"] += pec
    round_log["total_output_tokens"] += ec


# ============================================================
# 工具函数
# ============================================================

def read_novel_lines(start, count):
    """读取 novel.txt 的指定行范围，返回带行号的文本"""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = []
    for i in range(start - 1, min(start - 1 + count, len(lines))):
        result.append(f"{i+1}: {lines[i].rstrip()}")
    return "\n".join(result)


def get_dialogue_list():
    """提取所有对话，返回 [(行号, 对话内容), ...]"""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    dialogues = []
    for line_num, line in enumerate(content.split("\n"), start=1):
        matches = re.findall(r"「([^」]+)」", line)
        for dialogue in matches:
            dialogues.append((line_num, dialogue))
    return dialogues


def get_labeled_count():
    """读取已标注数量"""
    if not os.path.exists(LABELED_PATH):
        return 0
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        return len(f.readlines())


def write_label(name):
    """Python 脚本写入一条标注结果（Agent 不写）"""
    with open(LABELED_PATH, "a", encoding="utf-8") as f:
        f.write(name + "\n")


def call_ollama(messages, tools=None, label=""):
    """调用 Ollama API，返回 (response_text, prompt_tokens, eval_tokens, tool_calls)"""
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
# 工具定义（给 Labeler 用）
# ============================================================

TOOL_READ_NOVEL = {
    "type": "function",
    "function": {
        "name": "read_novel_lines",
        "description": "读取 novel.txt 的指定行范围。start 是起始行号(1-based)，count 是读取行数。",
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
# ShortMem Agent
# ============================================================

class ShortMemAgent:
    """短期记忆 Agent - 维护最近 N 轮的标注摘要"""
    
    def __init__(self, max_rounds=5):
        self.max_rounds = max_rounds
        self.history = []  # [(line_num, speaker, summary), ...]
    
    def update(self, line_num, speaker, summary):
        self.history.append((line_num, speaker, summary))
        if len(self.history) > self.max_rounds:
            self.history.pop(0)
    
    def get_summary(self):
        if not self.history:
            return "（暂无历史标注记录）"
        lines = ["[短期记忆 - 最近标注]"]
        for line_num, speaker, summary in self.history:
            lines.append(f"  #{line_num}: {speaker} | {summary}")
        return "\n".join(lines)


# ============================================================
# LongMem Agent
# ============================================================

class LongMemAgent:
    """长期记忆 Agent - 渐进压缩短期记忆"""
    
    def __init__(self, compress_every=5):
        self.compress_every = compress_every
        self.compression_count = 0
        self.long_memory = ""
        self.pending_summaries = []
    
    def add_summary(self, line_num, speaker, summary):
        self.pending_summaries.append(f"#{line_num}: {speaker} | {summary}")
    
    def should_compress(self):
        return len(self.pending_summaries) >= self.compress_every
    
    def compress(self, round_log):
        """压缩短期记忆为长期记忆，并记录日志"""
        if not self.pending_summaries:
            return
        
        history_text = "\n".join(self.pending_summaries)
        
        messages = [
            {
                "role": "system",
                "content": """你是一个小说分析记忆压缩助手。你的任务是将最近的标注摘要压缩为长期记忆。

保留内容：
- 角色关系图谱（谁是谁、什么关系）
- 当前主线情节（正在去哪里、目的是什么）
- 关键设定（角色身份、特殊设定）
- 角色别名映射（赫萝 = 贤狼赫萝 = 女孩）

丢弃内容：
- 具体对话内容
- 交易细节
- 路人名字
- 已过时的临时信息

输出格式：
角色：
- 角色名：身份描述

主线：
- 当前进展

别名映射：
- A = B = C"""
            },
            {
                "role": "user",
                "content": f"以下是最近的标注摘要：\n\n{history_text}\n\n请压缩为长期记忆。"
            }
        ]
        
        text, pec, ec, _ = call_ollama(messages, label="LongMem")
        
        # 记录 LongMem 日志
        log_agent(round_log, "LongMem", "compressor", messages, text, pec, ec)
        
        self.long_memory = text
        self.pending_summaries = []
        self.compression_count += 1
    
    def get_memory(self):
        if not self.long_memory:
            return "（暂无长期记忆，标注尚未开始或数据量不足）"
        return f"[长期记忆]\n{self.long_memory}"


# ============================================================
# Labeler Agent
# ============================================================

class LabelerAgent:
    """标注 Agent - 分析对话，识别说话人，可调用工具"""
    
    def __init__(self):
        self.tool_call_count = 0
    
    def label(self, line_num, dialogue, context_lines, short_mem_text, long_mem_text, round_log):
        """
        标注一条对话的说话人
        返回: (speaker, summary, pec, ec)
        """
        system_prompt = """你是一个小说对话角色标注助手。你的任务是判断指定对话的说话人。

规则：
1. 仔细阅读给定的对话和上下文
2. 判断「」内的对话是谁说的
3. 如果上下文不足以确定，调用 read_novel_lines 工具搜索更多线索
4. 角色名字要具体准确（如"赫萝"而非"女孩"）
5. 如果一句话有多个可接受答案，用"|"分隔（如"赫萝|贤狼赫萝"）
6. 非对话内容（内心独白、拟声词）标注为"非人物发声"
7. 不要编造任何名字！

输出格式（严格遵循）：
<answer>说话人名字</answer>
<reason>判断依据，包括关键线索所在行号</reason>
<summary>一句话摘要，格式：说话人 | 做了什么/说了什么</summary>"""
        
        user_content = f"""请标注以下对话的说话人：

第{line_num}行：「{dialogue}」

[上下文片段]
{context_lines}

{short_mem_text}

{long_mem_text}"""
        
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
                # 记录最后一次 Labeler 调用（无工具调用的最终回答）
                log_agent(round_log, f"Labeler", "labeler", messages, text, pec, ec)
                break
            
            # 记录带工具调用的 Labeler 调用
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
                "messages_snapshot": messages[-1]["content"][:100],
                "response_before_tool": text[:200],
                "tool_calls": tc_details,
                "prompt_eval_count": pec,
                "eval_count": ec,
            })
            
            # 执行工具
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
                    print(f"    🔧 工具调用 #{self.tool_call_count}: read_novel_lines({s}-{s+c-1}) → {len(result)} 字符")
                    messages.append({"role": "tool", "content": result})
        else:
            print(f"    ⚠️ 达到最大工具调用次数 {max_tool_rounds}")
        
        # 记录工具调用日志
        round_log["tool_calls"] = tool_call_log
        
        # 解析输出
        speaker = self._parse_answer(text)
        summary = self._parse_summary(text)
        
        return speaker, summary, total_pec, total_ec
    
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


# ============================================================
# Boss（由 Python 脚本扮演）
# ============================================================

class Boss:
    """Boss - Python 脚本实现，协调各 Agent"""
    
    def __init__(self, context_range=20, short_mem_rounds=5, long_mem_compress_every=5):
        self.context_range = context_range
        self.labeler = LabelerAgent()
        self.short_mem = ShortMemAgent(max_rounds=short_mem_rounds)
        self.long_mem = LongMemAgent(compress_every=long_mem_compress_every)
        self.total_tokens = 0
        self.round_count = 0
    
    def process_one(self, line_num, dialogue, round_log):
        """处理一条对话的标注"""
        self.round_count += 1
        
        print(f"\n{'='*60}")
        print(f"  第 {self.round_count} 轮标注")
        print(f"  对话：第{line_num}行「{dialogue}」")
        print(f"{'='*60}")
        
        # 1. 获取上下文片段
        ctx_start = max(1, line_num - self.context_range)
        ctx_count = self.context_range * 2 + 1
        context = read_novel_lines(ctx_start, ctx_count)
        
        # 2. 获取记忆
        short_mem_text = self.short_mem.get_summary()
        long_mem_text = self.long_mem.get_memory()
        
        # 记录 Boss 发给 Labeler 的任务
        round_log["boss_task"] = {
            "dialogue_line": line_num,
            "dialogue_text": dialogue,
            "context_range": f"{ctx_start}-{ctx_start + ctx_count - 1}",
            "short_mem": short_mem_text,
            "long_mem": long_mem_text,
        }
        
        # 3. 调用 Labeler
        speaker, summary, pec, ec = self.labeler.label(
            line_num, dialogue, context, short_mem_text, long_mem_text, round_log
        )
        self.total_tokens += pec + ec
        
        print(f"  📝 标注结果: {speaker}")
        print(f"  📊 Labeler Token: 输入 {pec} + 输出 {ec} = {pec + ec}")
        print(f"  📈 累计消耗: {self.total_tokens} tokens")
        
        # 4. 更新记忆
        self.short_mem.update(line_num, speaker, summary)
        self.long_mem.add_summary(line_num, speaker, summary)
        
        # 5. 检查是否需要压缩长期记忆
        if self.long_mem.should_compress():
            print(f"  🔄 触发长期记忆压缩...")
            self.long_mem.compress(round_log)
            print(f"  📝 长期记忆已更新")
        
        round_log["result"] = {
            "speaker": speaker,
            "summary": summary,
        }
        
        return speaker


# ============================================================
# 验证脚本
# ============================================================

def validate():
    """对比 labeled.txt 和 answers.txt，生成验证报告"""
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
        if label in acceptable:
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
    parser = argparse.ArgumentParser(description="多 Agent 小说对话角色标注")
    parser.add_argument("--start", type=int, default=0, help="从第几条对话开始标注（默认0=从上次继续）")
    parser.add_argument("--count", type=int, default=1, help="标注几条对话（默认1）")
    parser.add_argument("--context-range", type=int, default=20, help="上下文片段大小（默认±20行）")
    parser.add_argument("--short-mem", type=int, default=5, help="短期记忆保留轮数（默认5）")
    parser.add_argument("--long-mem-every", type=int, default=5, help="长期记忆压缩频率（默认每5轮）")
    parser.add_argument("--validate", action="store_true", help="标注完成后执行验证")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  多 Agent 小说对话角色标注系统")
    print(f"  模型: {OLLAMA_MODEL}")
    print(f"  服务器: {OLLAMA_BASE_URL}")
    print(f"  上下文范围: ±{args.context_range}行")
    print(f"  短期记忆: {args.short_mem}轮")
    print(f"  长期记忆压缩: 每{args.long_mem_every}轮")
    print(f"  日志文件: {LOG_PATH}")
    print("=" * 60)
    
    # 清空日志
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    
    # 获取所有对话
    dialogues = get_dialogue_list()
    labeled_count = get_labeled_count()
    
    print(f"\n  小说总对话数: {len(dialogues)}")
    print(f"  已标注: {labeled_count}")
    print(f"  未标注: {len(dialogues) - labeled_count}")
    
    # 确定起始位置
    start_idx = args.start if args.start > 0 else labeled_count
    end_idx = min(start_idx + args.count, len(dialogues))
    
    if start_idx >= len(dialogues):
        print("\n  所有对话已标注完毕！")
        return
    
    print(f"  本次标注: 第 {start_idx+1} 到 {end_idx} 条（共 {end_idx - start_idx} 条）")
    
    # 初始化 Boss
    boss = Boss(
        context_range=args.context_range,
        short_mem_rounds=args.short_mem,
        long_mem_compress_every=args.long_mem_every
    )
    
    # 逐条标注
    for idx in range(start_idx, end_idx):
        line_num, dialogue = dialogues[idx]
        
        # 创建本轮日志
        round_log = new_round(idx - start_idx + 1, line_num, dialogue)
        
        # Boss 处理
        speaker = boss.process_one(line_num, dialogue, round_log)
        
        # Python 脚本写入（Agent 不写）
        write_label(speaker)
        print(f"  ✅ 已写入: {speaker}")
        
        # 写入日志
        log_entry(round_log)
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"  标注完成")
    print(f"  本次标注: {end_idx - start_idx} 条")
    print(f"  总 Token 消耗: {boss.total_tokens}")
    print(f"  工具调用次数: {boss.labeler.tool_call_count}")
    print(f"  长期记忆压缩: {boss.long_mem.compression_count} 次")
    print(f"  日志已保存: {LOG_PATH}")
    print(f"{'='*60}")
    
    # 验证
    if args.validate:
        validate()


if __name__ == "__main__":
    main()
