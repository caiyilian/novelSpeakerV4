# -*- coding: utf-8 -*-
"""Corrected scorer: normalizes 无人称引语 <-> 非人物发声 on BOTH sides.

Bug being fixed: cont_agent.semantic maps only the OUTPUT's 无人称引语 to
非人物发声 but leaves the ANSWER side alone, and run_label.NON_PERSON_ALIASES
does not contain 无人称引语. So answer=无人称引语 & output=无人称引语 scored
WRONG. This script normalizes both sides to a single internal enum.

Usage: python score_fixed.py [--evidence current|frozen]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "src"))

import run_label as rl  # noqa: E402

MARKER_RE = re.compile(r"【([^】]+)】")
VOL = {1: ROOT / "data", **{n: ROOT / "data" / f"volume{n}" for n in range(2, 6)}}
FROZEN = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
RES = HERE / "results"

# single internal enum for the non-person category
NP_CANON = "非人物发声"
NP_EQUIV = {"无人称引语", "非人物发声", "非人物"}


def canon(x: str) -> str:
    """Map every non-person spelling to ONE canonical label."""
    x = x.strip()
    if x in NP_EQUIV:
        return NP_CANON
    return x


def read_answers(v):
    out = []
    for line in (VOL[v] / "answers.txt").read_text(encoding="utf-8").splitlines():
        for m in MARKER_RE.finditer(line):
            out.append({canon(p) for p in m.group(1).strip().split("|") if p.strip()})
    return out


def parse_output(path):
    g = {}
    if not path.exists():
        return g
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^D?(\d+)\s*[|｜:：\t]\s*(.+)$", line.strip())
        if m:
            g[int(m.group(1))] = m.group(2).strip()
    return g


def match(parts, label, identities):
    """Both sides canonicalized BEFORE matching."""
    p = {canon(x) for x in label.split("|") if x.strip()}
    if not p:
        return False
    if parts & p:
        return True
    return rl._validation_lenient_match(parts, p, verified_identities=identities)[0]


def score_one(tag, evidence="frozen", verbose=True):
    tot = ok = 0
    per = {}
    for v in range(1, 6):
        ans = read_answers(v)
        if evidence == "frozen":
            ident = rl._load_verified_validation_identities(
                str(FROZEN / f"volume{v}" / "evidence_vault.json"))
        else:
            ident = rl._load_verified_validation_identities(
                str(VOL[v] / "evidence_vault.json"))
        out = parse_output(RES / f"baseline_vol{v}_{tag}" / "raw_output.txt")
        o = sum(1 for i, a in enumerate(ans, 1) if match(a, out.get(i, ""), ident))
        tot += len(ans)
        ok += o
        per[v] = (o, len(ans))
    if verbose:
        print(f"{tag:<12}" + "".join(f"{per[v][0]:>6}" for v in range(1, 6))
              + f" | {ok}/{tot} = {round(ok*100/tot,2)}%")
    return ok, tot, per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="frozen", choices=["frozen", "current"])
    args = ap.parse_args()
    print(f"评分器：双侧归一（无人称引语<->非人物发声）；证据库={args.evidence}")
    print(f"{'方案':<12}{'v1':>6}{'v2':>6}{'v3':>6}{'v4':>6}{'v5':>6} | {'五卷':>9}")
    for tag in ["contfull", "evledger", "scene", "arb", "arb3", "ref", "arb5",
                "npfix", "final"]:
        score_one(tag, args.evidence)
