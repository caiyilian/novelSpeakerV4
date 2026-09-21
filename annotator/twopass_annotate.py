# -*- coding: utf-8 -*-
"""Two-pass prototype: build a whole-volume character ledger, then attribute
quotes with the ledger as the candidate set.

Motivation (see 重构准备二 §3.4): the heaviest residual error block is entity
handling — alias folding, generic-label fragmentation, and swapping between the
two leads. One flat call has no shared notion of "who exists in this book".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_dsv4flash import BASE_URL, API_KEY, OPENER, RESULTS, usage_of  # noqa: E402

ROOT = HERE.parents[0]
VOLUME_DIRS = {1: ROOT / "data", **{n: ROOT / "data" / f"volume{n}" for n in range(2, 6)}}
QUOTE_RE = re.compile(r"「[^」]*」")

LEDGER_SYSTEM = (
    "你是小说角色信息抽取专家。你会拿到一整卷小说的全文（行首为原书行号）。"
    "请抽取出这一卷中**所有会说话或被提到的人物实体**，并为每个实体归并别名。\n"
    "要求：\n"
    "1. `canonical` 是全书对该角色**最常用的称呼**（原文出现过的最短常用形式）。\n"
    "2. `full_name` 是原文给出的完整姓名（若有），没有就留空。\n"
    "3. `aliases` 列出原文中所有指向同一角色的其他称呼，包括：全名、简称、"
    "职业/身份泛称（如「旅行商人」）、外貌泛称（如「女孩」）、以及明确等同的称呼。\n"
    "4. **只归并原文明确能证明是同一人的称呼**。不能仅凭性别或职业相似就合并。\n"
    "5. 没有姓名、只用泛称称呼的人物（如「村民」「骑士」「老板」）也必须列入，"
    "canonical 就用那个泛称。\n"
    "6. 同一角色在全书必须只出现一次，不要拆成多条。\n"
    "7. 为每个角色给出 `evidence_line`：最能证明其身份或别名关系的原书行号。\n"
    "8. 不要列入动物、组织、地点、物品。\n\n"
    "输出**合法 JSON**，格式：\n"
    '{"characters":[{"canonical":"罗伦斯","full_name":"克拉福•罗伦斯",'
    '"aliases":["旅行商人","商人"],"first_line":29,"evidence_line":222,'
    '"note":"男主角，旅行商人"}]}\n'
    "只输出 JSON，不要任何其他文字。"
)

ATTR_SYSTEM = (
    "你是小说对话说话人标注员。你会拿到一整卷小说全文（行首为原书行号）、"
    "一份**本卷角色账本**、以及全部待标注引语的编号清单。\n"
    "【核心规则】\n"
    "1. 说话人**必须**从角色账本的 canonical 名称中选一个。账本里没有的名字不得使用。\n"
    "2. 引号若**不是**某个角色当场说出口的话（例如：被引用的词句或说法、"
    "神态姿态描写、单独强调某个词、环境声与动物声、叙述中随手加引号的短语），"
    "统一输出 `非对话引语`。\n"
    "3. 内心独白（原文写「某某心想：『…』」并明确归因于某人）→ 输出那个人。\n"
    "4. 判断顺序：先看引号后紧跟的叙述句；没有就向前回溯最近的发声动作；"
    "一段里只有两人对话时交替是常态，但叙述会打断交替。\n"
    "5. 附近的叙述里出现某个人名，不等于该引语就是他说的。\n"
    "6. 同一角色全卷必须用同一个标签，不要一会儿用泛称一会儿用实名。\n"
    "输出格式：每行一条 `编号|说话人`，必须输出全部条目，不要解释、不要代码块。"
)


def call(model, messages, max_tokens, reasoning="none", timeout=3600):
    payload = {"model": model, "messages": messages,
               "temperature": 0, "max_tokens": max_tokens}
    if reasoning == "none":
        payload["reasoning_effort"] = "none"
    else:
        payload["reasoning_effort"] = reasoning
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build(vol):
    vdir = VOLUME_DIRS[vol]
    lines = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    body = "\n".join(f"{i}: {l}" for i, l in enumerate(lines, 1))
    roster, quote_lines = [], []
    for ln, line in enumerate(lines, 1):
        for _ in QUOTE_RE.finditer(line):
            quote_lines.append(ln)
    roster = "\n".join(f"D{i:04d} -> 第{ln}行" for i, ln in enumerate(quote_lines, 1))
    return body, roster, quote_lines


def parse_labels(text):
    got = {}
    for line in text.splitlines():
        m = re.match(r"^D?(\d+)\s*[|｜:：\t]\s*(.+)$", line.strip())
        if m:
            try:
                got[int(m.group(1))] = m.group(2).strip()
            except ValueError:
                continue
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=5)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--ledger-model", default=None)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--tag", default="twopass")
    args = ap.parse_args()
    ledger_model = args.ledger_model or args.model

    body, roster, quote_lines = build(args.volume)
    count = len(quote_lines)
    print(f"[twopass] vol{args.volume}: {count} quotes", flush=True)

    # ---- pass 1: ledger ----
    t0 = time.time()
    led = call(ledger_model, [
        {"role": "system", "content": LEDGER_SYSTEM},
        {"role": "user", "content": f"第{args.volume}卷全文：\n\n{body}\n\n请输出角色账本 JSON。"},
    ], max_tokens=args.max_tokens, reasoning=args.reasoning)
    ledger_s = time.time() - t0
    ledger_raw = led["choices"][0]["message"].get("content") or ""
    cleaned = re.sub(r"^```(?:json)?|```$", "", ledger_raw.strip(), flags=re.M).strip()
    start = cleaned.find("{")
    ledger = json.loads(cleaned[start:]) if start >= 0 else {"characters": []}
    chars = ledger.get("characters", [])
    print(f"  ledger: {len(chars)} entities, {ledger_s:.1f}s "
          f"({led['choices'][0].get('finish_reason')})", flush=True)

    # ---- pass 2: attribution ----
    ledger_text = json.dumps(ledger, ensure_ascii=False, indent=1)
    user = (
        f"第{args.volume}卷全文：\n\n<<<正文>>>\n{body}\n<<<正文结束>>>\n\n"
        f"本卷角色账本：\n{ledger_text}\n\n"
        f"待标注引语共 {count} 处：\n{roster}\n\n"
        f"请逐条输出 `编号|说话人`，必须输出全部 {count} 条。"
    )
    t1 = time.time()
    attr = call(args.model, [
        {"role": "system", "content": ATTR_SYSTEM},
        {"role": "user", "content": user},
    ], max_tokens=args.max_tokens, reasoning=args.reasoning)
    attr_s = time.time() - t1
    attr_raw = attr["choices"][0]["message"].get("content") or ""
    parsed = parse_labels(attr_raw)
    print(f"  attribution: {len(parsed)}/{count} parsed, {attr_s:.1f}s "
          f"({attr['choices'][0].get('finish_reason')})", flush=True)

    out_dir = RESULTS / f"baseline_vol{args.volume}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_output.txt").write_text(attr_raw, encoding="utf-8")
    (out_dir / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "volume": args.volume, "model": args.model, "ledger_model": ledger_model,
        "prompt_version": "twopass", "thinking": args.reasoning,
        "expected_quotes": count, "parsed_quotes": len(parsed),
        "finish_reason": attr["choices"][0].get("finish_reason"),
        "latency_s": round(ledger_s + attr_s, 2),
        "ledger_latency_s": round(ledger_s, 2),
        "attribution_latency_s": round(attr_s, 2),
        "ledger_entities": len(chars),
        "usage_total": {
            "prompt": (led.get("usage") or {}).get("prompt_tokens", 0)
                      + (attr.get("usage") or {}).get("prompt_tokens", 0),
            "completion": (led.get("usage") or {}).get("completion_tokens", 0)
                          + (attr.get("usage") or {}).get("completion_tokens", 0),
        },
        "missing": count - len(parsed),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
