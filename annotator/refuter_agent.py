# -*- coding: utf-8 -*-
"""Scheme 4 (docs/discussion/20260921_0936): adversarial Refuter arbitration.

0936's distinctive arbitration mechanism is the *Refuter*: instead of asking the
model to "pick the right one" (scheme 2's arbiter), it asks it to "find a
line-level refutation of one candidate". 0936 argues this is a more focused task
than open-ended selection. It uses a WIDER window (±30 lines) than the arbiter.

Same scaffolding as scheme 2: pure-Python divergence detection on cont vs
evledger, cluster, then a high-reasoning refuter sees both candidates + wide
context + ledger. If it can refute A (line-level evidence A is wrong) it picks B;
if it refutes B it picks A; otherwise it conservatively keeps A (cont).
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
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
RES = HERE / "results"

REFUTER_SYSTEM = (
    "你是小说对话说话人标注的对抗裁决员。对每条引语，给你两个候选答案（A、B）。\n"
    "你的任务**不是选一个**，而是**找出哪个候选与原文矛盾**（能找到证明它错的行号）。\n"
    "【规则】\n"
    "- 若候选A与原文矛盾（能指出证明A错的行号）、候选B不矛盾 → 最终选 B\n"
    "- 若候选B与原文矛盾、候选A不矛盾 → 最终选 A\n"
    "- 若都矛盾或都不矛盾 → 保守选 A（保留默认通道）\n"
    "【答案形式】说话人用账本里的规范名，或原文稳定泛称，或 `无人称引语`。\n"
    "【输出铁律】每条**只输出一行**：`D编号|最终说话人|反证行号(无则填「无」)|一句依据`。"
    "必须输出全部条目，不要解释、不要代码块、不要复述引语。"
)


def ref_call(messages, max_tokens=4096, timeout=900):
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


def parse(path):
    g = {}
    if not path.exists():
        return g
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^D?(\d+)\s*[|｜:：\t]\s*(.+)$", line.strip())
        if m:
            g[int(m.group(1))] = m.group(2).strip()
    return g


def norm_label(x):
    return "非人物发声" if x == "无人称引语" else x


def cluster(idxs, gap, max_size=10):
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
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--radius", type=int, default=30)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--max-clusters", type=int, default=0)
    args = ap.parse_args()

    vdir = ca.VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    answers = ca.read_answers(vdir / "answers.txt")
    identities = ca.rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    total = len(answers)

    dialogue = []
    for ln, line in enumerate(novel, 1):
        for m in re.finditer(r"「[^」]*」", line):
            dialogue.append((ln, m.group(0)))

    cont = parse(RES / f"baseline_vol{args.volume}_contfull" / "raw_output.txt")
    ev = parse(RES / f"baseline_vol{args.volume}_evledger" / "raw_output.txt")
    ledger = json.loads((RES / f"baseline_vol{args.volume}_twopass" / "ledger.json")
                        .read_text(encoding="utf-8"))
    names = ca.ledger_names(ledger)

    div = [i for i in range(1, total + 1)
           if norm_label(cont.get(i, "")) != norm_label(ev.get(i, ""))]
    clusters = cluster(div, args.gap)
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
    print(f"vol{args.volume}: 分歧 {len(div)} 条 → 簇 {len(clusters)} 个", flush=True)

    verdict = {}
    t0 = time.time()
    for ci, cl in enumerate(clusters, 1):
        lines = [dialogue[i - 1][0] for i in cl]
        lo = max(1, min(lines) - args.radius)
        hi = min(len(novel), max(lines) + args.radius)
        ctx = "\n".join(f"{k}: {novel[k-1]}" for k in range(lo, hi + 1))
        items = "\n".join(
            f"D{i:04d} 第{dialogue[i-1][0]}行：「{dialogue[i-1][1]}」  "
            f"候选A={cont.get(i, '?')}  候选B={ev.get(i, '?')}"
            for i in cl)
        user = (f"【角色账本】\n{json.dumps(ledger, ensure_ascii=False)}\n\n"
                f"【待裁决条目】\n{items}\n\n【原文上下文】\n{ctx}\n\n"
                f"请逐条对抗裁决，输出 `D编号|最终说话人|反证行号(无则填无)|依据`，共 {len(cl)} 条。")
        try:
            data = ref_call([{"role": "system", "content": REFUTER_SYSTEM},
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
                  f"已裁决 {len(verdict)})", flush=True)

    results = {i: (verdict.get(i) or cont.get(i, "")) for i in range(1, total + 1)}
    ok_cont = sum(1 for j, a in enumerate(answers, 1)
                  if ca.semantic(a["parts"], cont.get(j, ""), identities))
    ok_ref = sum(1 for j, a in enumerate(answers, 1)
                 if ca.semantic(a["parts"], results.get(j, ""), identities))
    print(f"vol{args.volume}: cont {ok_cont}/{total}={round(ok_cont*100/total,2)}%  →  "
          f"对抗裁决 {ok_ref}/{total}={round(ok_ref*100/total,2)}%  "
          f"({len(verdict)} 条被裁决, 耗时 {time.time()-t0:.0f}s)", flush=True)

    out = RES / f"baseline_vol{args.volume}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_output.txt").write_text(
        "\n".join(f"D{j:04d}|{results.get(j,'')}" for j in range(1, total + 1)),
        encoding="utf-8")
    (out / "verdicts.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
