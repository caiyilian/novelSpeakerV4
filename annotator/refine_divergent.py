# -*- coding: utf-8 -*-
"""Refine divergent quotes with reasoning enabled.

Cont and evledger disagree on ~731 quotes; among them are the ~203 both-wrong.
Re-label the divergent quotes with reasoning_effort=high + richer context, in
the hope that explicit thinking resolves the hard alternation/identity cases.

Locating "divergent" only compares two outputs (no ground truth), so this is a
legitimate refinement step.
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
from probe_dsv4flash import BASE_URL, API_KEY, OPENER  # noqa: E402
from twopass_annotate import VOLUME_DIRS  # noqa: E402
import run_label as rl  # noqa: E402

ROOT = HERE.parents[0]
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
MARKER_RE = re.compile(r"【([^】]+)】")

SYSTEM = (
    "你是小说对话说话人标注员。判断给定一处引语是谁说的。\n"
    "你会看到引语前后的原文（行首为原书行号），以及本卷角色账本、前后已标注结果。\n"
    "请仔细推敲：\n"
    "1. 引语**紧邻**的发声动作是谁（谁「说/道/问/答」）；\n"
    "2. 若两人快速对话，看上一句是谁说的，据此交替；但注意叙述会打断交替，不要机械套 A-B-A；\n"
    "3. 附近出现的人名不等于说话人，要看清发声动词的主语；\n"
    "4. 引语若不是人当场说的话（被引用的说法、神态字幕、强调词、环境声/拟声、内心独白），"
    "分别标注：神态字幕/引用/强调/环境声 → `无人称引语`；内心独白 → 归该主体。\n"
    "只输出一行：`说话人|证据行号|理由`。"
)


def call(messages, reasoning="high", timeout=300):
    payload = {"model": "deepseek-v4-flash", "messages": messages,
               "temperature": 0, "max_tokens": 2048, "reasoning_effort": reasoning}
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse(p):
    g = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^D?(\d+)\s*[|｜:：\t]\s*(.+)$", l.strip())
        if m:
            try:
                g[int(m.group(1))] = m.group(2).strip()
            except ValueError:
                continue
    return g


def read_answers(path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MARKER_RE.finditer(line):
            raw = m.group(1).strip()
            items.append({"parts": {p.strip() for p in raw.split("|") if p.strip()},
                          "line": line_no})
    return items


def semantic(parts, label, identities):
    p = {x.strip() for x in label.split("|") if x.strip()}
    p = {"非人物发声" if x == "无人称引语" else x for x in p}
    if parts & p:
        return True
    return rl._validation_lenient_match(parts, p, verified_identities=identities)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--radius", type=int, default=15)
    args = ap.parse_args()

    vdir = VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    answers = read_answers(vdir / "answers.txt")
    identities = rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    ct = parse(HERE / "results" / f"baseline_vol{args.volume}_contfull" / "raw_output.txt")
    ev = parse(HERE / "results" / f"baseline_vol{args.volume}_evledger" / "raw_output.txt")
    ledger = json.loads((HERE / "results" / f"baseline_vol{args.volume}_twopass"
                         / "ledger.json").read_text(encoding="utf-8"))
    ledger_text = json.dumps(ledger, ensure_ascii=False, indent=1)

    # divergent = different labels (not counting 无人称引语 vs 非人物发声 alias)
    def norm(x):
        return {"无人称引语": "非人物发声"}.get(x, x)

    divergent = [i for i in range(1, len(answers) + 1)
                 if norm(ct.get(i, "")) != norm(ev.get(i, ""))]
    print(f"[refine] vol{args.volume}: 分歧 {len(divergent)} 条", flush=True)

    qlines = []
    for ln, l in enumerate(novel, 1):
        for _ in re.finditer(r"「[^」]*」", l):
            qlines.append(ln)

    refined = dict(ct)
    t0 = time.time()
    for i in divergent:
        ln = qlines[i - 1]
        lo = max(1, ln - args.radius)
        hi = min(len(novel), ln + args.radius)
        ctx = "\n".join(f"{k}: {novel[k-1]}" for k in range(lo, hi + 1))
        prev = [f"D{j}: {ct.get(j, '')}" for j in range(max(1, i - 5), i)]
        nxt = [f"D{j}: {ct.get(j, '')}" for j in range(i + 1, min(len(answers), i + 5))]
        user = (f"原文上下文：\n{ctx}\n\n本卷角色账本：\n{ledger_text}\n\n"
                f"前后已标注：\n" + "\n".join(prev + [f'D{i}: ???'] + nxt) +
                f"\n\n请判断 D{i}（第 {ln} 行）的引语是谁说的，只输出一行 `说话人|证据行号|理由`。")
        try:
            data = call([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}])
            content = data["choices"][0]["message"].get("content") or ""
            parts = [p.strip() for p in re.split(r"[|｜]", content.strip().splitlines()[0])]
            sp = parts[0] if parts else ""
            if sp and not re.match(r"^D?\d+$", sp):
                refined[i] = sp
        except Exception as exc:  # noqa: BLE001
            print(f"  D{i} failed: {exc}"[:100], flush=True)
        if i % 100 == 0:
            print(f"  ...D{i} ({time.time()-t0:.0f}s)", flush=True)

    before = sum(1 for i, a in enumerate(answers, 1)
                 if semantic(a["parts"], ct.get(i, ""), identities))
    after = sum(1 for i, a in enumerate(answers, 1)
                if semantic(a["parts"], refined.get(i, ""), identities))
    out_dir = HERE / "results" / f"baseline_vol{args.volume}_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_output.txt").write_text(
        "\n".join(f"D{i:04d}|{refined.get(i, '')}" for i in range(1, len(answers) + 1)),
        encoding="utf-8")
    print(f"vol{args.volume}: cont {before}/{len(answers)}={before*100/len(answers):.2f}%  "
          f"-> refined {after}/{len(answers)}={after*100/len(answers):.2f}%  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
