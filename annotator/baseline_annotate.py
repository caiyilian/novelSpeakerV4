# -*- coding: utf-8 -*-
"""Strong-model baseline: read the whole volume at once and label every quote.

This is the minimal "one agent + full text + one shot" configuration described in
the hand-over plan. It never touches project code; all output goes to
tmp/deepseek_probe/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

from probe_dsv4flash import chat, usage_of, RESULTS, BASE_URL  # noqa: E402

ROOT = HERE.parents[0]

VOLUME_DIRS = {
    1: ROOT / "data",
    **{n: ROOT / "data" / f"volume{n}" for n in range(2, 6)},
}

SYSTEM_V1 = (
    "你是一名严谨的小说对话说话人标注员。"
    "你会拿到一整卷小说的全部正文（带原始行号），以及该卷所有「」引号对话的编号清单。"
    "你的任务是为每一处引号对话判断其说话人。"
    "要求：\n"
    "1. 说话人有明确姓名时优先输出姓名；没有姓名时使用原文中出现的稳定泛称（如「村民」「骑士」）。\n"
    "2. 内心独白、环境声、物体声等非人物发声统一输出「非人物发声」。\n"
    "3. 同一行若有多个引号，按出现顺序分别标注。\n"
    "4. 必须完全依据原文证据，不要臆造原文没有的名字。\n"
    "5. 同一个角色在全卷中必须使用同一个标签，不要一会儿用泛称一会儿用实名。"
)

SYSTEM_V2 = (
    "你是一名严谨的小说对话说话人标注员，正在为一整卷小说标注每一处「」引号对话的说话人。"
    "你会拿到全卷正文（行首数字为原书行号）与全部待标注引语的编号清单。\n"
    "【判断顺序】\n"
    "1. 先读紧跟在引号后的叙述句，它通常直接点明说话人。\n"
    "2. 没有的话，向前回溯最近的发声动作。注意“某某说”“某某答道”修饰的是紧邻的引号。\n"
    "3. 一段里只有两人对话时交替发言是常态，但叙述会打断交替，不要机械套用 A-B-A。\n"
    "【非人物发声】以下一律输出「非人物发声」，不要因为附近出现人名就归给他：\n"
    "- 内心独白（人物心里想的话）\n"
    "- 环境声、动物声、物体声、拟声词\n"
    "- 叙述者转述、回忆或书信中被引用的语句\n"
    "- 找不到明确发声者的引语\n"
    "【标签规范】\n"
    "- 原文已给出姓名时必须输出姓名，不要降级成职业或外貌泛称。\n"
    "- 没有姓名的人物使用原文出现过的泛称，并在全卷保持同一个标签。\n"
    "- 不要把泛称人物改写成具体姓名，除非原文明确把两者等同。\n"
    "【输出】每行一条 `编号|说话人`，必须输出全部条目，不要解释、不要代码块。"
)

SYSTEM = SYSTEM_V1
PROMPT_VERSIONS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}

USER_TMPL = """下面是第{volume}卷的全部正文，行首数字是原书行号：

<<<正文开始>>>
{body}
<<<正文结束>>>

该卷共 {count} 处引号对话，编号如下（D 编号 -> 原书行号）：
{roster}

请为每一处引号对话标注说话人。
输出格式：每行一条，`编号|说话人`，不要输出任何其他文字、不要用代码块。
必须输出全部 {count} 条，顺序与编号顺序一致。
示例：
D0001|村民
D0002|罗伦斯
"""


def build_inputs(volume: int):
    novel_path = VOLUME_DIRS[volume] / "novel.txt"
    lines = novel_path.read_text(encoding="utf-8").splitlines()

    body = "\n".join(f"{i}: {line}" for i, line in enumerate(lines, 1))

    roster_items = []
    for line_no, line in enumerate(lines, 1):
        for _ in re.finditer(r"「[^」]*」", line):
            roster_items.append(line_no)

    roster = "\n".join(
        f"D{idx:04d} -> 第{ln}行" for idx, ln in enumerate(roster_items, 1)
    )
    return len(lines), body, roster, roster_items


def read_answers(path: pathlib.Path):
    from analyze_common import read_answers as ra  # noqa
    return ra(path)


def parse_output(text, expected_count):
    got = {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        m = re.match(r"^(D?\d+)\s*[|｜:：\t]\s*(.+)$", line)
        if not m:
            continue
        key = m.group(1)
        if key.upper().startswith("D"):
            key = key[1:]
        try:
            idx = int(key)
        except ValueError:
            continue
        speaker = m.group(2).strip().strip("。，,、\"'` ")
        got[idx] = speaker
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=1)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--thinking", default="none",
                    help="'none' disables reasoning; 'default' keeps it")
    ap.add_argument("--prompt-version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--tag", default="",
                    help="suffix for the output directory")
    args = ap.parse_args()

    system_prompt = PROMPT_VERSIONS[args.prompt_version]

    total_lines, body, roster, roster_items = build_inputs(args.volume)
    count = len(roster_items)

    user = USER_TMPL.format(volume=args.volume, body=body, count=count, roster=roster)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    extra = {}
    if args.thinking == "none":
        extra["reasoning_effort"] = "none"

    print(f"[baseline] volume {args.volume}: {total_lines} lines, {count} quotes, "
          f"prompt {len(user)} chars", flush=True)

    import urllib.request
    import urllib.error
    from probe_dsv4flash import OPENER, API_KEY

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    payload.update(extra)
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST",
    )
    started = time.time()
    with OPENER.open(req, timeout=3600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - started

    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    finished = choice.get("finish_reason")

    parsed = parse_output(content, count)

    out_dir = RESULTS / (f"baseline_vol{args.volume}" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(exist_ok=True)
    (out_dir / "prompt.txt").write_text(user, encoding="utf-8")
    (out_dir / "system.txt").write_text(system_prompt, encoding="utf-8")
    (out_dir / "raw_output.txt").write_text(content, encoding="utf-8")

    report = {
        "volume": args.volume,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "declared_model": data.get("model"),
        "thinking": args.thinking,
        "prompt_chars": len(user),
        "expected_quotes": count,
        "parsed_quotes": len(parsed),
        "finish_reason": finished,
        "latency_s": round(elapsed, 2),
        "usage": usage_of(data),
        "missing": count - len(parsed),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[baseline] wrote {out_dir}")


if __name__ == "__main__":
    main()
