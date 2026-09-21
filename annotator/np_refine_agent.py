# -*- coding: utf-8 -*-
"""Scheme 6 (docs/discussion/Agent编排重构方案_2026-09-21_1015): attributable-nonperson.

1015's central finding: the largest error block (236 quotes, 3.37pp) is
"无人称引语 over-prediction". Root cause: cont_agent's prompt uses a *quote-type*
criterion (神态字幕 / 被引用词句 / 强调词 → 无人称引语), but the ground truth
uses an *attributable* criterion: if a subject can be attached to the quote
(神态/心理/言说动作 + a person), it belongs to that person.

This scheme post-processes ONLY the quotes cont labeled 无人称引语 / 非人物发声:
re-judge each with the "attributable" criterion. Everything else is untouched.
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

NP_SYSTEM = (
    "你是小说引语的二次确认员。下面是若干处被初步判为「无人称引语」（非人物发声）的引语。\n"
    "初步判定用了「引号类型」判据（神态字幕/被引用词句/单独强调词 → 无人称引语），可能过判。\n"
    "正确的判据是「能否归属到某个主体」：\n"
    "- 若上下文有「某人 + 动作/神态/心理/言说」能挂住这句引语（即这句是某人说/想/表现的内容）"
    " → 归该主体（说话人，用账本规范名或原文稳定泛称）\n"
    "- 若纯环境声/动物声/物体声/无人挂住的引用 → 保持「无人称引语」\n"
    "【输出铁律】每条**只输出一行**：`D编号|说话人(或「无人称引语」)|一句依据`。"
    "必须输出全部条目，不要解释、不要代码块、不要复述引语。"
)


def np_call(messages, max_tokens=4096, timeout=900):
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
    ap.add_argument("--tag", default="npfix")
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--gap", type=int, default=3)
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
    ledger = json.loads((RES / f"baseline_vol{args.volume}_twopass" / "ledger.json")
                        .read_text(encoding="utf-8"))
    names = ca.ledger_names(ledger)

    # only re-judge the quotes cont labeled as non-person
    np_idx = [i for i in range(1, total + 1)
              if norm_label(cont.get(i, "")) == "非人物发声"]
    clusters = cluster(np_idx, args.gap)
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
    print(f"vol{args.volume}: 无人称引语条目 {len(np_idx)} 条 → 簇 {len(clusters)} 个", flush=True)

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
                f"【待二次确认的引语】\n{items}\n\n【原文上下文】\n{ctx}\n\n"
                f"请逐条用「可归属」判据确认，输出 `D编号|说话人(或无人称引语)|依据`，共 {len(cl)} 条。")
        try:
            data = np_call([{"role": "system", "content": NP_SYSTEM},
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
                  f"已改判 {len(verdict)})", flush=True)

    results = {i: (verdict.get(i) or cont.get(i, "")) for i in range(1, total + 1)}
    ok_cont = sum(1 for j, a in enumerate(answers, 1)
                  if ca.semantic(a["parts"], cont.get(j, ""), identities))
    ok_np = sum(1 for j, a in enumerate(answers, 1)
                if ca.semantic(a["parts"], results.get(j, ""), identities))
    print(f"vol{args.volume}: cont {ok_cont}/{total}={round(ok_cont*100/total,2)}%  →  "
          f"判据修正 {ok_np}/{total}={round(ok_np*100/total,2)}%  "
          f"({len(verdict)} 条被改判, 耗时 {time.time()-t0:.0f}s)", flush=True)

    out = RES / f"baseline_vol{args.volume}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_output.txt").write_text(
        "\n".join(f"D{j:04d}|{results.get(j,'')}" for j in range(1, total + 1)),
        encoding="utf-8")
    (out / "verdicts.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
