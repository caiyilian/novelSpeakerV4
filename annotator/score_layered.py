# -*- coding: utf-8 -*-
"""Score any candidate run against the layered evaluation set.

Why: aggregate accuracy hides direction. A change that gains 2.7pp on volume 1
and loses 2.2pp on volume 3 shows up as +0.4pp and looks worthless, while a
change that fixes the hard core and breaks easy items looks identical to one
that does nothing. This reports the four strata separately.

Usage:
    python score_layered.py <run_dir> [--labels <labeled.txt>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_label as rl  # noqa: E402

BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
OUT = ROOT / "tmp" / "refactor_prep"
MARKER_RE = re.compile(r"【([^】]+)】")
VOLUME_DIRS = {1: ROOT / "data", **{n: ROOT / "data" / f"volume{n}" for n in range(2, 6)}}


def read_answers(path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MARKER_RE.finditer(line):
            raw = m.group(1).strip()
            items.append({"parts": {p.strip() for p in raw.split("|") if p.strip()},
                          "line": line_no})
    return items


def read_labels(path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_out(text):
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


def load_strata():
    """The four strata, keyed (vol, index) -> stratum name."""
    mapping = {}
    baseline_out = {}
    for vol in range(1, 6):
        path = OUT / f"layered_{'protect_old_right_baseline_wrong'}.json"
        pass
    for name in ("protect_old_right_baseline_wrong", "baseline_fixed",
                 "both_wrong", "both_right"):
        rows = json.loads((OUT / f"layered_{name}.json").read_text(encoding="utf-8"))
        for r in rows:
            mapping[(r["vol"], r["i"])] = name
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="directory holding baseline_vol*/raw_output.txt")
    ap.add_argument("--suffix", default=None,
                    help="run suffix, e.g. _evidence; resolved under "
                         "tmp/deepseek_probe/results/baseline_vol{N}<suffix>. "
                         "Default (omitted) = the plain one-shot baseline.")
    ap.add_argument("--labels", default=None,
                    help="score a plain labeled.txt tree instead")
    args = ap.parse_args()

    results_root = ROOT / "annotator" / "results"

    def resolve(vol):
        if args.labels:
            base = pathlib.Path(args.labels)
            sub = base / f"volume{vol}"
            return read_labels((sub if sub.exists() else base) / "labeled.txt")
        if args.run_dir and args.run_dir != "__old__":
            d = pathlib.Path(args.run_dir) / f"baseline_vol{vol}"
        elif args.suffix is not None:
            d = results_root / f"baseline_vol{vol}{args.suffix}"
        else:
            d = results_root / f"baseline_vol{vol}"
        if not (d / "raw_output.txt").exists():
            raise SystemExit(f"missing {d/'raw_output.txt'}")
        got = parse_out((d / "raw_output.txt").read_text(encoding="utf-8"))
        n = max(got) if got else 0
        return [got.get(i, "") for i in range(1, n + 1)]

    name = args.labels or (args.run_dir if args.run_dir else
                           f"run{args.suffix if args.suffix is not None else ''}")
    strata = load_strata()
    agg = {}
    per_vol = {}
    for vol in range(1, 6):
        vdir = VOLUME_DIRS[vol]
        answers = read_answers(vdir / "answers.txt")
        identities = rl._load_verified_validation_identities(
            str(BACKUP / f"volume{vol}" / "evidence_vault.json"))
        labels = resolve(vol)

        vol_ok = 0
        for i, a in enumerate(answers, 1):
            ok = semantic(a["parts"], labels[i - 1] if i <= len(labels) else "",
                          identities)
            vol_ok += int(ok)
            s = strata.get((vol, i))
            if s:
                d = agg.setdefault(s, {"total": 0, "correct": 0})
                d["total"] += 1
                d["correct"] += int(ok)
        per_vol[vol] = round(vol_ok * 100.0 / len(answers), 2)

    total_all = sum(v["total"] for v in agg.values())
    ok_all = sum(v["correct"] for v in agg.values())
    print(f"### {name}")
    print(f"整体: {ok_all}/{total_all} = {round(ok_all * 100.0 / total_all, 2)}%")
    print("逐卷:", " ".join(f"v{v}:{per_vol[v]}" for v in sorted(per_vol)))
    print(f"{'分层':<32}{'总数':>7}{'正确':>7}{'准确率':>9}")
    for key, label in (("both_right", "两边都对(回归检查)"),
                       ("protect_old_right_baseline_wrong", "保护样本(旧对基线错)"),
                       ("baseline_fixed", "基线已反超"),
                       ("both_wrong", "真难点")):
        d = agg.get(key)
        if not d:
            continue
        print(f"{label:<32}{d['total']:>7}{d['correct']:>7}"
              f"{round(d['correct'] * 100.0 / d['total'], 2):>8}%")
    print()


if __name__ == "__main__":
    main()
