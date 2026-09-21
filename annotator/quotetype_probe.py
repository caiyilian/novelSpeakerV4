# -*- coding: utf-8 -*-
"""Is "quote used as speech or not" learnable as its own task?

Splits the question the single-shot baseline silently never answers: instead of
asking "who speaks here", ask only "is this quote a real utterance or a
non-speech quotation". If the model can do this in isolation, the fix is task
decomposition rather than a better label.
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

import urllib.request  # noqa: E402

from probe_dsv4flash import BASE_URL, API_KEY, OPENER, RESULTS, usage_of  # noqa: E402

ROOT = HERE.parents[0]
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"

MARKER_RE = re.compile(r"【([^】]+)】")
QUOTE_RE = re.compile(r"「[^」]*」")

SYSTEM_FEWSHOT = (
    "你是一名小说文本分析员。你会拿到小说全卷正文（行首为原书行号）与若干处引号的编号。\n"
    "请判断**每一处引号在原文中的实际用法**，只在这两种之间选一个：\n"
    "A = 实际发言：这个引号代表某个角色当场说出口的话。\n"
    "B = 非实际发言：这个引号不是角色当场说出口的话。\n\n"
    "以下都是 B（非实际发言）的实例，请注意它们长什么样：\n"
    "· 引用词语或说法：罗伦斯当初是基于商人特有的小气想法，也就是「不喜欢拖着空荡荡的货台旅行」。\n"
    "· 引号内是一个名词短语而非句子：「咱的名字是赫萝」这句话让他愣住了。\n"
    "· 描写神态姿态：赫萝用棉被盖住整个身子，一副「不关我事」的模样。\n"
    "· 单独强调某个词：赫萝加重语气说了「又」字。\n"
    "· 引号内是比喻或俗语：只好像「被狐狸盯上的鸡」一样发抖。\n"
    "· 内心独白：罗伦斯心想「这下可麻烦了」。\n"
    "· 他人话语的转述：「你可别后悔。」他当时是这么说的。\n"
    "· 环境声、动物声：远处传来「汪」的一声。\n\n"
    "以下都是 A（实际发言）的实例：\n"
    "· 叙述直接标明发声：赫萝打了一个哈欠说：「呵……啊呼。」\n"
    "· 两人对话中的一句：「汝果然是个笨蛋。」\n"
    "· 引号后紧跟发声动作：「唔。」罗伦斯含糊地应了一声。\n\n"
    "判断要点：\n"
    "1. 引号内若不是一个可以独立说出口的句子，多半是 B。\n"
    "2. 引号所修饰的是「词句本身」「神态」「概念」而不是「谁说的话」，就是 B。\n"
    "3. 附近出现人名不等于 A。\n"
    "4. 只有长句、口语化、带语气词、且上下文有发声动作或对话交替时，才判 A。\n"
    "输出格式：每行一条，`编号 A` 或 `编号 B`，不要输出任何其他内容。"
)


def build(vol):
    vdir = BACKUP / f"volume{vol}"
    answers = (vdir / "answers.txt").read_text(encoding="utf-8").splitlines()
    lines = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()

    items = []
    for line_no, ans_line in enumerate(answers, 1):
        for m in MARKER_RE.finditer(ans_line):
            label = m.group(1).strip()
            items.append({"line": line_no, "label": label,
                          "expect_b": label in ("非人物发声", "旁白")})

    body = "\n".join(f"{i}: {l}" for i, l in enumerate(lines, 1))
    roster = []
    covered = set()
    for idx, it in enumerate(items, 1):
        roster.append(f"Q{idx:04d} 在第{it['line']}行")
        covered.add(it["line"])
    return body, "\n".join(roster), items, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--prompt", default="fewshot", choices=["base", "fewshot"])
    ap.add_argument("--reasoning", default="none",
                    help="reasoning_effort: none/low/medium/high")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    system_prompt = SYSTEM if args.prompt == "base" else SYSTEM_FEWSHOT
    body, roster, items, _ = build(args.volume)
    user = (
        f"下面是第{args.volume}卷的正文：\n\n<<<正文>>>\n{body}\n<<<正文结束>>>\n\n"
        f"需要判断的引号共 {len(items)} 处：\n{roster}\n\n"
        f"请逐条输出 `编号 A` 或 `编号 B`，必须输出全部 {len(items)} 条。"
    )
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user}],
        "temperature": 0, "max_tokens": args.max_tokens,
    }
    if args.reasoning == "none":
        payload["reasoning_effort"] = "none"
    else:
        payload["reasoning_effort"] = args.reasoning
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    started = time.time()
    with OPENER.open(req, timeout=3600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - started
    content = data["choices"][0]["message"].get("content") or ""

    got = {}
    for line in content.splitlines():
        m = re.match(r"^[DQ]?(\d+)\s*[|\s:：\t]\s*([AB])\s*$", line.strip(), re.I)
        if m:
            got[int(m.group(1))] = m.group(2).upper()

    tp = fp = tn = fn = 0
    misses = []
    for idx, it in enumerate(items, 1):
        g = got.get(idx)
        want_b = it["expect_b"]
        if g == "B" and want_b:
            tp += 1
        elif g == "B" and not want_b:
            fp += 1
            if len(misses) < 10:
                misses.append({"i": idx, "line": it["line"], "label": it["label"],
                               "pred": "B"})
        elif g == "A" and not want_b:
            tn += 1
        elif g == "A" and want_b:
            fn += 1
            if len(misses) < 20:
                misses.append({"i": idx, "line": it["line"], "label": it["label"],
                               "pred": "A"})
        else:
            fn += 1

    out_dir = RESULTS / (f"quotetype_vol{args.volume}" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(exist_ok=True)
    (out_dir / "raw_output.txt").write_text(content, encoding="utf-8")
    report = {
        "volume": args.volume,
        "prompt": args.prompt,
        "reasoning": args.reasoning,
        "quotes": len(items),
        "ground_truth_B": sum(1 for i in items if i["expect_b"]),
        "parsed": len(got),
        "latency_s": round(elapsed, 2),
        "usage": usage_of(data),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "confusion": {"B_predicted_B": tp, "B_predicted_A": fn,
                      "A_predicted_B": fp, "A_predicted_A": tn},
        "recall_B": round(tp * 100.0 / max(1, tp + fn), 2),
        "precision_B": round(tp * 100.0 / max(1, tp + fp), 2),
        "overall": round((tp + tn) * 100.0 / len(items), 2),
        "false_positives": misses[:10],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
