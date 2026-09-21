# -*- coding: utf-8 -*-
"""Scheme 5 (docs/discussion/2026-09-21_09-46): precise-contract arbitration.

09-46's core claim: the current system lacks a reliable *task-location contract*
(quote id = line only, no same-line ordinal / span) and drops evidence, so the
reviewer "re-guesses from scratch". This scheme upgrades the arbiter with:

1. Precise quote location via `quote_occurrence` (span + same-line ordinal +
   scope_before/scope_after), removing same-line multi-quote ambiguity.
2. Evidence enforcement: the arbiter MUST output an evidence line number that
   falls inside the novel; a verdict without a valid evidence line is rejected
   (keep the default cont label). This is the "结构化提交 + 证据强制" idea.

Divergence detection and clustering are identical to scheme 2 (cont vs evledger).
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
from quote_occurrence import build_quote_occurrences  # noqa: E402

ROOT = HERE.parents[0]
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
RES = HERE / "results"

ARB_SYSTEM = (
    "你是小说对话说话人标注的裁决员。下面给你若干条引语，每条有两个候选答案"
    "（来自两个独立标注通道）。你要独立阅读原文，判断每条引语真正的说话人。\n"
    "【原则】以原文为准，不要因为某个候选出现得多就选它；两个候选都可能不对。\n"
    "【证据铁律】每条裁决**必须**给出证据行号（能支持你判断的原文行，行首数字即行号）。"
    "证据行号必须在给出的原文范围内；找不到证据就保守保留候选A。\n"
    "【答案形式】说话人用账本里的规范名，或原文稳定泛称，或 `无人称引语`。\n"
    "【输出铁律】每条**只输出一行**：`D编号|最终说话人|证据行号|一句依据`。"
    "必须输出全部条目，不要解释、不要代码块、不要复述引语。"
)


def arb_call(messages, max_tokens=4096, timeout=900):
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


def parse_verdicts(text, names, total_lines):
    """Parse `D编号|说话人|证据行号|依据`. Reject if evidence line is invalid.

    Strict-ish: speaker must exactly match a known name (not substring), or be a
    short generic / 无人称引语. Evidence line must be an int in [1, total_lines].
    """
    out = {}
    for line in text.splitlines():
        m = re.match(r"^D?(\d{1,5})\s*[|｜:：\t]\s*(.+)$", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        parts = [p.strip() for p in re.split(r"[|｜]", m.group(2))]
        if len(parts) < 3:
            continue  # must have speaker + evidence line + reason
        speaker = parts[0]
        ev_line = parts[1]
        # evidence line must be a valid line number inside the novel
        if not re.match(r"^\d{1,5}$", ev_line):
            continue
        if not (1 <= int(ev_line) <= total_lines):
            continue
        # strict speaker: exact match only (fix 09-46's "n in cand" substring bug)
        if speaker == "无人称引语":
            sp = speaker
        elif speaker in names:
            sp = speaker
        elif 1 <= len(speaker) <= 12 and not any(x in speaker for x in "：，。？!、的了吗行"):
            sp = speaker
        else:
            continue
        out[idx] = sp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--tag", default="arb5")
    ap.add_argument("--radius", type=int, default=15)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--max-clusters", type=int, default=0)
    args = ap.parse_args()

    vdir = ca.VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    answers = ca.read_answers(vdir / "answers.txt")
    identities = ca.rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    total = len(answers)
    total_lines = len(novel)

    records = build_quote_occurrences(novel)  # precise span + ordinal + scope
    rec = {r["dialogue_index"] + 1: r for r in records}  # 1-based D id

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
        lines = [rec[i]["line"] for i in cl if i in rec]
        lo = max(1, min(lines) - args.radius)
        hi = min(total_lines, max(lines) + args.radius)
        ctx = "\n".join(f"{k}: {novel[k-1]}" for k in range(lo, hi + 1))
        items = "\n".join(
            f"D{i:04d} 第{rec[i]['line']}行第{rec[i]['same_line_ordinal']+1}处"
            f"（前「{rec[i]['scope_before']}」后「{rec[i]['scope_after']}」）："
            f"「{rec[i]['text']}」  候选A={cont.get(i, '?')}  候选B={ev.get(i, '?')}"
            for i in cl if i in rec)
        user = (f"【角色账本】\n{json.dumps(ledger, ensure_ascii=False)}\n\n"
                f"【待裁决条目】\n{items}\n\n【原文上下文】\n{ctx}\n\n"
                f"请逐条裁决，输出 `D编号|最终说话人|证据行号|依据`，共 {len(cl)} 条。")
        try:
            data = arb_call([{"role": "system", "content": ARB_SYSTEM},
                             {"role": "user", "content": user}])
            content = data["choices"][0]["message"].get("content") or ""
            v = parse_verdicts(content, names, total_lines)
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
    ok_arb = sum(1 for j, a in enumerate(answers, 1)
                 if ca.semantic(a["parts"], results.get(j, ""), identities))
    print(f"vol{args.volume}: cont {ok_cont}/{total}={round(ok_cont*100/total,2)}%  →  "
          f"精确契约仲裁 {ok_arb}/{total}={round(ok_arb*100/total,2)}%  "
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
