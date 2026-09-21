# -*- coding: utf-8 -*-
"""Post-hoc alias normalization: rewrite each output label to its ledger's
canonical form. Targets the "model copies surface-form variants / typos instead
of the canonical name" error class (e.g. 叶克柏 -> 叶克伯), plus generic-label
splintering.

Conservative: only normalize through UNIQUE, non-generic alias links, and never
touch labels already in the ledger's canonical set. Purely mechanical, no model
calls, no ground-truth leakage.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_label as rl  # noqa: E402

BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
PROBE = ROOT / "annotator" / "results"
MARKER_RE = re.compile(r"【([^】]+)】")
VOLUME_DIRS = {1: ROOT / "data", **{n: ROOT / "data" / f"volume{n}" for n in range(2, 6)}}

# overly-generic alias words that must never be used to rewrite (shared across people)
GENERIC = {"商人", "男子", "老板", "手下", "家伙", "姑娘", "女人", "老人", "年轻人",
           "女孩", "骑士", "村民", "农夫", "士兵", "卫兵", "客人", "店员", "旅人",
           "女孩", "少年", "少女", "女性", "男性", "人物"}


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


def build_alias_map(ledger):
    # alias -> list of canonicals that claim it
    alias_to = {}
    canonicals = set()
    for c in ledger.get("characters", []):
        can = c.get("canonical", "").strip()
        full = c.get("full_name", "").strip()
        if can:
            canonicals.add(can)
        for a in [can, full] + list(c.get("aliases", [])):
            a = a.strip()
            if not a or a in GENERIC:
                continue
            alias_to.setdefault(a, []).append(can)
    # keep only unique, non-generic mappings
    mapping = {}
    for alias, cans in alias_to.items():
        if alias in canonicals:
            continue  # already canonical, don't touch
        if len(set(cans)) == 1 and cans[0] and alias != cans[0]:
            mapping[alias] = cans[0]
    return mapping


def main():
    summary = {}
    for vol in range(1, 6):
        vdir = VOLUME_DIRS[vol]
        answers = read_answers(vdir / "answers.txt")
        identities = rl._load_verified_validation_identities(
            str(BACKUP / f"volume{vol}" / "evidence_vault.json"))
        ledger = json.loads((PROBE / f"baseline_vol{vol}_twopass"
                             / "ledger.json").read_text(encoding="utf-8"))
        mapping = build_alias_map(ledger)

        src = parse_speaker((PROBE / f"baseline_vol{vol}_evledger"
                             / "raw_output.txt").read_text(encoding="utf-8"))
        rewrites = 0
        out = {}
        for i, label in src.items():
            norm = mapping.get(label, label)
            if norm != label:
                rewrites += 1
            out[i] = norm

        before = sum(1 for i, a in enumerate(answers, 1)
                     if semantic(a["parts"], src.get(i, ""), identities))
        after = sum(1 for i, a in enumerate(answers, 1)
                    if semantic(a["parts"], out.get(i, ""), identities))
        summary[str(vol)] = {
            "before": round(before * 100.0 / len(answers), 2),
            "after": round(after * 100.0 / len(answers), 2),
            "delta": round((after - before) * 100.0 / len(answers), 2),
            "rewrites": rewrites,
            "alias_map_size": len(mapping),
        }
        d = PROBE / f"baseline_vol{vol}_norm"
        d.mkdir(parents=True, exist_ok=True)
        (d / "raw_output.txt").write_text(
            "\n".join(f"D{i:04d}|{out.get(i, '')}" for i in range(1, len(answers) + 1)),
            encoding="utf-8")
        print(f"vol{vol}: {before*100/len(answers):.2f}% -> {after*100/len(answers):.2f}% "
              f"(rewrites={rewrites}, map={len(mapping)})", flush=True)

    (ROOT / "tmp" / "refactor_prep" / "alias_normalization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
