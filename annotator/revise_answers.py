# -*- coding: utf-8 -*-
"""Revise the ground-truth answers.txt using the "attributable" criterion.

The original answers label the non-person category with a *quote-type* notion
(神态字幕 / 被引用词句 / 强调词 → 非人物发声), and hedge with compound labels
like `罗伦斯|非人物发声`. This script re-judges every answer entry that contains
"非人物发声" using the *attributable* criterion (same as np_refine_agent):

  - if the quote can be attached to a subject (某人 + 动作/神态/心理/言说) -> that subject
  - else -> 无人称引语

It rewrites those entries to a SINGLE label (subject or 无人称引语), and leaves
everything else untouched. Original answers are backed up separately.

Uses deepseek-v4-flash (only allowed model), high reasoning.
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
sys.path.insert(0, str(HERE.parents[0] / "src"))

import cont_agent as ca  # noqa: E402

ROOT = HERE.parents[0]
RES = HERE / "results"
MARKER_RE = re.compile(r"【([^】]+)】")

NP_SYSTEM = (
    "你是小说引语的标注口径修订员。下面给你若干处引语，它们的参考答案用了「非人物发声」"
    "或「某人|非人物发声」这种不确定写法（这是标注者当时的对冲，口径没定下来）。\n"
    "请用统一的「能否归属到某个主体」判据，给出每条引语的**单一确定答案**：\n"
    "- 若上下文有「某人 + 动作/神态/心理/言说」能挂住这句引语"
    "（即这句是某人说/想/表现的内容） → 归该主体（说话人，用账本规范名或原文稳定泛称）\n"
    "- 若纯环境声/动物声/物体声/无人挂住的引用（如商品名、被谈论的词句） → 标 `无人称引语`\n"
    "【输出铁律】每条**只输出一行**：`D编号|说话人(或「无人称引语」)|一句依据`。"
    "必须输出全部条目，不要解释、不要代码块、不要复述引语。"
)


def call(messages, max_tokens=4096, timeout=900):
    payload = {"model": "deepseek-v4-flash", "messages": messages,
               "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "high"}
    req = urllib.request.Request(
        ca.BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + ca.API_KEY},
        method="POST")
    with ca.OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cluster(idxs, gap=3, max_size=10):
    if not idxs:
        return []
    groups, cur = [], [idxs[0]]
    for i in idxs[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    out = []
    for g in groups:
        for k in range(0, len(g), max_size):
            out.append(g[k:k + max_size])
    return out


def parse_verdicts(text, names):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^D?(\d{1,5})\s*[|｜:：\t]\s*(.+)$", line.strip())
        if not m:
            continue
        sp = ca.strict_parse(m.group(2), names)
        if sp:
            out[int(m.group(1))] = sp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true",
                    help="only report, do not write answers.txt")
    args = ap.parse_args()

    vdir = ca.VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    ans_path = vdir / "answers.txt"
    ans_lines = ans_path.read_text(encoding="utf-8").splitlines()
    ledger = json.loads((RES / f"baseline_vol{args.volume}_twopass" / "ledger.json")
                        .read_text(encoding="utf-8"))
    names = ca.ledger_names(ledger)

    dialogue = []
    for ln, line in enumerate(novel, 1):
        for m in re.finditer(r"「[^」]*」", line):
            dialogue.append((ln, m.group(0)))

    # entries that contain 非人物发声, with their position in answers (1-based D id)
    np_idx = []
    for i, a in enumerate(ca.read_answers(ans_path), 1):
        if "非人物" in "|".join(a["parts"]):
            np_idx.append(i)

    clusters = cluster(np_idx)
    print(f"vol{args.volume}: 含「非人物发声」的答案条目 {len(np_idx)} 条 → 簇 {len(clusters)} 个",
          flush=True)

    verdict = {}
    t0 = time.time()
    for ci, cl in enumerate(clusters, 1):
        lines = [dialogue[i - 1][0] for i in cl]
        lo = max(1, min(lines) - args.radius)
        hi = min(len(novel), max(lines) + args.radius)
        ctx = "\n".join(f"{k}: {novel[k-1]}" for k in range(lo, hi + 1))
        items = "\n".join(
            f"D{i:04d} 第{dialogue[i-1][0]}行：「{dialogue[i-1][1]}」"
            for i in cl)
        user = (f"【角色账本】\n{json.dumps(ledger, ensure_ascii=False)}\n\n"
                f"【待修订条目】\n{items}\n\n【原文上下文】\n{ctx}\n\n"
                f"请逐条给出单一确定答案，输出 `D编号|说话人(或无人称引语)|依据`，共 {len(cl)} 条。")
        try:
            data = call([{"role": "system", "content": NP_SYSTEM},
                         {"role": "user", "content": user}])
            content = data["choices"][0]["message"].get("content") or ""
            v = parse_verdicts(content, names)
            for i in cl:
                if i in v:
                    verdict[i] = v[i]
        except Exception as exc:  # noqa: BLE001
            print(f"  [cluster {ci}] {exc}"[:120], flush=True)
        if ci % 20 == 0:
            print(f"  ...{ci}/{len(clusters)} 簇 ({time.time()-t0:.0f}s, "
                  f"已修订 {len(verdict)})", flush=True)

    # rewrite each 【...】 that contains 非人物 with the single label (in order)
    new_lines = []
    d_id = 0
    changed = 0
    for line in ans_lines:
        def repl(m):
            nonlocal d_id, changed
            d_id += 1
            inner = m.group(1)
            if "非人物发声" not in inner:
                return m.group(0)
            if d_id in verdict:
                new_label = verdict[d_id]
                # normalize the non-person label to 无人称引语
                if new_label == "非人物发声":
                    new_label = "无人称引语"
                changed += 1
                return f"【{new_label}】"
            return m.group(0)
        new_lines.append(MARKER_RE.sub(repl, line))
    new_text = "\n".join(new_lines)

    total_np = len(np_idx)
    print(f"vol{args.volume}: 修订 {changed}/{total_np} 条（其余 {total_np-changed} 条未改）",
          flush=True)

    # persist
    out_dir = RES / "answers_revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"vol{args.volume}_verdicts.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.dry_run:
        ans_path.write_text(new_text, encoding="utf-8")
        print(f"vol{args.volume}: 已写回 answers.txt", flush=True)
    else:
        print(f"vol{args.volume}: dry-run，未写回", flush=True)


if __name__ == "__main__":
    main()
