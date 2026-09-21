# -*- coding: utf-8 -*-
"""Build the layered evaluation set that every future candidate must be scored on.

Cross-tabulates the last complete old-system output against the new single-shot
baseline, so regressions and wins are visible per item rather than in aggregate.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_label as rl  # noqa: E402

BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
PROBE = ROOT / "annotator" / "results"
OUT = ROOT / "tmp" / "refactor_prep"
OUT.mkdir(parents=True, exist_ok=True)

MARKER_RE = re.compile(r"【([^】]+)】")
VOLUME_DIRS = {1: ROOT / "data", **{n: ROOT / "data" / f"volume{n}" for n in range(2, 6)}}
QUOTE_RE = re.compile(r"「[^」]*」")


def read_answers(path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MARKER_RE.finditer(line):
            raw = m.group(1).strip()
            items.append({"raw": raw,
                          "parts": {p.strip() for p in raw.split("|") if p.strip()},
                          "line": line_no})
    return items


def read_labels(path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_baseline(text):
    got = {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        m = re.match(r"^(D?\d+)\s*[|｜:：\t]\s*(.+)$", line)
        if not m:
            continue
        key = m.group(1).lstrip("Dd")
        try:
            got[int(key)] = m.group(2).strip()
        except ValueError:
            continue
    return got


def semantic(answer_parts, label, identities):
    parts = {p.strip() for p in label.split("|") if p.strip()}
    if answer_parts & parts:
        return True
    return rl._validation_lenient_match(answer_parts, parts,
                                        verified_identities=identities)[0]


def category(answer_parts, label):
    def kind(values):
        norm = {rl._normalize_validation_label(v) for v in values if v}
        if not norm:
            return "empty"
        if rl.NON_PERSON_LABEL in norm:
            return "nonperson"
        if all(rl._is_generic_validation_label(v) for v in norm):
            return "generic"
        return "named"
    table = {
        ("named", "named"): "实名与实名互换",
        ("named", "generic"): "实名降级为泛称",
        ("generic", "named"): "泛称误判为实名",
        ("generic", "generic"): "泛称与泛称不一致",
        ("named", "nonperson"): "人物误判为非人物发声",
        ("generic", "nonperson"): "人物误判为非人物发声",
        ("nonperson", "named"): "非人物发声误判为人物",
        ("nonperson", "generic"): "非人物发声误判为人物",
        ("nonperson", "nonperson"): "非人物标签不一致",
    }
    return table.get((kind(answer_parts), kind({label})), "other")


def main():
    strata = defaultdict(list)
    cross = Counter()
    per_volume = {}

    for vol in range(1, 6):
        vdir = VOLUME_DIRS[vol]
        answers = read_answers(vdir / "answers.txt")
        old = read_labels(BACKUP / f"volume{vol}" / "labeled.txt")
        base = parse_baseline(
            (PROBE / f"baseline_vol{vol}" / "raw_output.txt").read_text(encoding="utf-8"))
        identities = rl._load_verified_validation_identities(
            str(BACKUP / f"volume{vol}" / "evidence_vault.json"))
        novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()

        counts = Counter()
        for i, ans in enumerate(answers, 1):
            old_ok = semantic(ans["parts"], old[i - 1], identities) if i <= len(old) else False
            base_ok = semantic(ans["parts"], base.get(i, ""), identities)
            counts[("old_ok" if old_ok else "old_bad",
                    "base_ok" if base_ok else "base_bad")] += 1
            cross[("old_ok" if old_ok else "old_bad",
                   "base_ok" if base_ok else "base_bad")] += 1

            row = {
                "vol": vol, "i": i, "line": ans["line"], "expected": ans["raw"],
                "old": old[i - 1] if i <= len(old) else "",
                "baseline": base.get(i, ""),
                "old_correct": old_ok, "baseline_correct": base_ok,
            }
            line_text = novel[ans["line"] - 1] if ans["line"] - 1 < len(novel) else ""
            row["quote"] = (QUOTE_RE.findall(line_text) or [""])[
                0 if ans["line"] == 0 else 0]

            if old_ok and not base_ok:
                row["category"] = category(ans["parts"], base.get(i, ""))
                strata["protect_old_right_baseline_wrong"].append(row)
            elif not old_ok and base_ok:
                strata["baseline_fixed"].append(row)
            elif not old_ok and not base_ok:
                row["category"] = category(ans["parts"], base.get(i, ""))
                strata["both_wrong"].append(row)
            else:
                strata["both_right"].append(row)

        per_volume[vol] = {f"{k[0]}__{k[1]}": v for k, v in counts.items()}

    summary = {
        "source": {
            "old_system": str(BACKUP.relative_to(ROOT)),
            "baseline": "tmp/deepseek_probe/results/baseline_vol{1..5}",
        },
        "cross_table": {f"{k[0]}__{k[1]}": v for k, v in cross.items()},
        "per_volume": {str(k): v for k, v in per_volume.items()},
        "strata_sizes": {k: len(v) for k, v in strata.items()},
        "protect_set_category_breakdown": dict(Counter(
            r["category"] for r in strata["protect_old_right_baseline_wrong"]).most_common()),
        "both_wrong_category_breakdown": dict(Counter(
            r["category"] for r in strata["both_wrong"]).most_common()),
    }

    (OUT / "layered_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in strata.items():
        (OUT / f"layered_{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # compact human-readable protect list
    lines = ["# 保护样本：旧系统对、基线与新方案都必须守住", ""]
    for r in strata["protect_old_right_baseline_wrong"][:60]:
        lines.append(f"- 第{r['vol']}卷 D{r['i']:04d} L{r['line']} | 期望 {r['expected']}"
                     f" | 旧 {r['old']} | 基线 {r['baseline']} | {r.get('category','')}")
    (OUT / "protect_set.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
