# -*- coding: utf-8 -*-
"""Generate File A: the error list for one volume, for independent review.

Two sections:
  A1 — model-vs-answer disagreements (46 for vol1)
  A2 — multi-label answers that need resolution (the 'name over occupation'
       criterion applied), i.e. entries where the answer itself hedges.

No answer-side editing, no model calls. Output is Markdown for human/LLM review.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_label as rl  # noqa: E402

spec = importlib.util.spec_from_file_location("sf", ROOT / "annotator" / "score_fixed.py")
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)

QUOTE_RE = re.compile(r"「[^」]*」")
M = re.compile(r"【([^】]+)】")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--context", type=int, default=4)
    args = ap.parse_args()

    v = args.volume
    novel = (sf.VOL[v] / "novel.txt").read_text(encoding="utf-8").splitlines()
    ident = rl._load_verified_validation_identities(
        str(sf.FROZEN / f"volume{v}" / "evidence_vault.json"))
    out = sf.parse_output(sf.RES / f"baseline_vol{v}_{args.tag}" / "raw_output.txt")
    led = json.loads((sf.RES / f"baseline_vol{v}_twopass" / "ledger.json")
                     .read_text(encoding="utf-8"))
    canon = {c["canonical"] for c in led.get("characters", []) if c.get("canonical")}

    dlg = []
    for ln, line in enumerate(novel, 1):
        for m in QUOTE_RE.finditer(line):
            dlg.append((ln, m.group(0)))

    raw = []
    for line in (sf.VOL[v] / "answers.txt").read_text(encoding="utf-8").splitlines():
        for m in M.finditer(line):
            raw.append([p.strip() for p in m.group(1).strip().split("|") if p.strip()])

    def ctx(ln):
        lo = max(1, ln - args.context)
        hi = min(len(novel), ln + args.context)
        block = []
        for k in range(lo, hi + 1):
            mark = " >>>" if k == ln else "    "
            block.append(f"{k:>5}{mark} {novel[k-1]}")
        return "\n".join(block)

    errs = [i for i, a in enumerate(raw, 1)
            if not sf.match(set(a), out.get(i, ""), ident)]
    multi = [i for i, a in enumerate(raw, 1) if len(a) > 1]
    # A2: multi-label items that are NOT already in A1 (avoid duplicates)
    auto_res = {}
    need_dec = []
    for i in multi:
        if i in errs:
            continue          # already listed in A1
        hit = [x for x in raw[i - 1] if x in canon]
        if len(hit) == 1:
            auto_res[i] = hit[0]
        else:
            need_dec.append(i)

    L = []
    L.append(f"# 文件A：第 {v} 卷 待核清单")
    L.append("")
    L.append(f"- 卷：第 {v} 卷，共 {len(raw)} 条引语")
    L.append(f"- 评分器：`annotator/score_fixed.py`（双侧归一）")
    L.append(f"- 模型输出：方案7（`baseline_vol{v}_final`）")
    L.append(f"- **A1 模型与答案不一致：{len(errs)} 条**")
    L.append(f"- **A2 答案多标签需裁定：{len(need_dec)} 条**（另有 {len(auto_res)} 条可依账本规则自动消解，仅统计）")
    L.append("")
    L.append("> 用途：交给独立审阅者（人或模型）判断**哪些是答案错了**，并给出依据。")
    L.append("> 注意：不一致 ≠ 答案错，也可能是模型错，或只是写法不同。")
    L.append("")
    L.append("**标签裁定规则**：有名字的用名字，没有名字的用职业。")
    L.append("")
    L.append(f"本卷账本规范名（canonical）：`{'`、`'.join(sorted(canon))}`")
    L.append("")

    L.append("---")
    L.append("")
    L.append(f"## A1 模型与答案不一致（{len(errs)} 条）")
    L.append("")
    for i in errs:
        ln, qt = dlg[i - 1]
        a_str = " / ".join(raw[i - 1])
        o_str = out.get(i, "") or "(空)"
        L.append(f"### A1-{i:04d}（第 {ln} 行）")
        L.append("")
        L.append(f"- 引语：{qt}")
        L.append(f"- 参考答案：`{a_str}`")
        L.append(f"- 模型输出：`{o_str}`")
        L.append("")
        L.append("```")
        L.append(ctx(ln))
        L.append("```")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"## A2 答案多标签需裁定（{len(need_dec)} 条）")
    L.append("")
    for i in need_dec:
        ln, qt = dlg[i - 1]
        a = raw[i - 1]
        L.append(f"### A2-{i:04d}（第 {ln} 行）")
        L.append("")
        L.append(f"- 引语：{qt}")
        L.append(f"- 答案多标签：`{' / '.join(a)}`")
        L.append(f"- 模型输出：`{out.get(i, '') or '(空)'}`")
        L.append("")
        L.append("```")
        L.append(ctx(ln))
        L.append("```")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"## A3 可自动消解的多标签（{len(auto_res)} 条，仅统计，不需审阅）")
    L.append("")
    L.append("这些条目的多标签集合里**恰好只有一个账本规范名**，其余都是职业/别名，")
    L.append("按「有名字用名字」规则可直接消解为该规范名。")
    L.append("")

    text = "\n".join(L)
    p = ROOT / "docs" / f"审阅_vol{v}_文件A_待核清单.md"
    p.write_text(text, encoding="utf-8")
    print(f"已写出：{p}")
    print(f"  A1 {len(errs)} 条 + A2 {len(need_dec)} 条（自动消解 {len(auto_res)} 条另计），{len(text)} 字符")

    # also dump structured JSON for the review webpage
    items = []
    for i in errs:
        ln, qt = dlg[i - 1]
        items.append({"kind": "A1", "d": i, "line": ln, "quote": qt,
                      "answer": raw[i - 1], "model": out.get(i, ""),
                      "context": ctx(ln)})
    for i in need_dec:
        ln, qt = dlg[i - 1]
        items.append({"kind": "A2", "d": i, "line": ln, "quote": qt,
                      "answer": raw[i - 1], "model": out.get(i, ""),
                      "context": ctx(ln)})
    jp = ROOT / "docs" / f"审阅_vol{v}_文件A_待核清单.json"
    jp.write_text(json.dumps({"volume": v, "canon": sorted(canon), "items": items},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结构化数据：{jp}")


if __name__ == "__main__":
    main()
