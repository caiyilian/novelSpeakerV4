# -*- coding: utf-8 -*-
"""Identity resolution over the ACTUAL output vocabulary (borrowed from CowAgent's
Deep Dream idea: distill a noisy working set into a clean canonical identity map).

Instead of trusting the one-shot ledger's alias list (which was noisy), take the
distinct labels the attribution actually produced, and ask the model to group
them into canonical identities. Then normalize the output through that map.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request
from collections import Counter

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
    "你是小说角色身份消解助手。下面是一份标注系统输出的「说话人标签列表」（带出现次数），"
    "同一个角色可能被写成多种不同形式（全名/简称/错字/不同泛称）。\n"
    "请把这些标签**归并成 canonical 身份**，输出 JSON：\n"
    '{"identities":[{"canonical":"叶克伯","variants":["叶克柏","叶克柏行长","塔兰铁诺行长"]}]}\n'
    "规则：\n"
    "1. canonical 取该角色最规范、最常用、无错字的写法。\n"
    "2. 只有明确是同一人的标签才归并；不同角色不要合并。\n"
    "3. 明显的同音/形近错字（柏/伯、塔兰铁诺/塔兰铁落）归为同一个人的 variants。\n"
    "4. 完全无归属把握的标签，单独列一个 canonical（就是它自己），不要强行合并。\n"
    "5. 「无人称引语」不要动。\n"
    "只输出 JSON，不要其他文字。"
)


def call(messages, timeout=600):
    payload = {"model": "deepseek-v4-flash", "messages": messages,
               "temperature": 0, "max_tokens": 4096, "reasoning_effort": "none"}
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_answers(path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MARKER_RE.finditer(line):
            raw = m.group(1).strip()
            items.append({"parts": {p.strip() for p in raw.split("|") if p.strip()}})
    return items


def parse_speaker(text):
    got = {}
    for line in text.splitlines():
        m = re.match(r"^D?(\d+)\s*[|｜:：\t]\s*(.+)$", line.strip())
        if m:
            try:
                got[int(m.group(1))] = m.group(2).strip()
            except ValueError:
                continue
    return got


def semantic(parts, label, identities):
    p = {x.strip() for x in label.split("|") if x.strip()}
    if parts & p:
        return True
    return rl._validation_lenient_match(parts, p, verified_identities=identities)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--base", default="evledger")
    args = ap.parse_args()

    vdir = VOLUME_DIRS[args.volume]
    answers = read_answers(vdir / "answers.txt")
    identities = rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    src = parse_speaker((HERE / "results" / f"baseline_vol{args.volume}_{args.base}"
                         / "raw_output.txt").read_text(encoding="utf-8"))

    vocab = Counter(src.values())
    vocab_txt = "\n".join(f"{k}\t{v}" for k, v in vocab.most_common())

    data = call([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"标签列表：\n{vocab_txt}\n\n请输出归并后的 JSON。"}])
    content = data["choices"][0]["message"].get("content") or ""
    c = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    s = c.find("{")
    idmap = json.loads(c[s:]) if s >= 0 else {"identities": []}

    variant2canon = {}
    for id_ in idmap.get("identities", []):
        can = id_.get("canonical", "")
        for v in id_.get("variants", []):
            variant2canon[v] = can

    out = {i: variant2canon.get(lbl, lbl) for i, lbl in src.items()}
    before = sum(1 for i, a in enumerate(answers, 1)
                 if semantic(a["parts"], src.get(i, ""), identities))
    after = sum(1 for i, a in enumerate(answers, 1)
                if semantic(a["parts"], out.get(i, ""), identities))

    d = HERE / "results" / f"baseline_vol{args.volume}_idres"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_output.txt").write_text(
        "\n".join(f"D{i:04d}|{out.get(i, '')}" for i in range(1, len(answers) + 1)),
        encoding="utf-8")
    (d / "idmap.json").write_text(json.dumps(idmap, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(f"vol{args.volume}: {before*100/len(answers):.2f}% -> {after*100/len(answers):.2f}% "
          f"(vocab={len(vocab)}, identities={len(idmap.get('identities', []))}, "
          f"mapped={len(variant2canon)})", flush=True)


if __name__ == "__main__":
    main()
