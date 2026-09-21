# -*- coding: utf-8 -*-
"""Score a baseline_annotate run against the ground truth with the project scorer."""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_label as rl  # noqa: E402
from baseline_annotate import parse_output, build_inputs, VOLUME_DIRS  # noqa: E402

MARKER_RE = re.compile(r"【([^】]+)】")


def read_answers(path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MARKER_RE.finditer(line):
            raw = m.group(1).strip()
            items.append({"raw": raw,
                          "parts": {p.strip() for p in raw.split("|") if p.strip()},
                          "line": line_no})
    return items


def speaker_kind(values):
    norm = {rl._normalize_validation_label(v) for v in values if v}
    if not norm:
        return "empty"
    if rl.NON_PERSON_LABEL in norm:
        return "nonperson"
    if all(rl._is_generic_validation_label(v) for v in norm):
        return "generic"
    return "named"


CATEGORY = {
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


def main():
    volume = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE / "results" / f"baseline_vol{volume}"

    raw = (run_dir / "raw_output.txt").read_text(encoding="utf-8")
    parsed = parse_output(raw, 0)

    vdir = VOLUME_DIRS[volume]
    answers = read_answers(vdir / "answers.txt")
    identities = rl._load_verified_validation_identities(
        str(vdir / "evidence_vault.json"))

    rows = []
    strict_correct = semantic_correct = 0
    missing = []
    for idx, ans in enumerate(answers, 1):
        got = parsed.get(idx)
        if got is None:
            missing.append(idx)
            got = ""
        parts = {p.strip() for p in got.split("|") if p.strip()}
        strict = bool(ans["parts"] & parts)
        if strict:
            semantic, reason = True, "exact"
        else:
            semantic, reason = rl._validation_lenient_match(
                ans["parts"], parts, verified_identities=identities)
        strict_correct += int(strict)
        semantic_correct += int(semantic)
        rows.append({"i": idx, "line": ans["line"], "expected": ans["raw"], "got": got,
                     "strict": strict, "semantic": semantic, "reason": reason,
                     "category": "" if semantic else CATEGORY.get(
                         (speaker_kind(ans["parts"]), speaker_kind(parts)), "other")})

    total = len(answers)
    cats = Counter(r["category"] for r in rows if not r["semantic"])
    confusions = Counter((r["expected"], r["got"]) for r in rows if not r["semantic"])
    windows = Counter()
    win_tot = Counter()
    for r in rows:
        w = (r["i"] - 1) // 100
        win_tot[w] += 1
        windows[w] += int(r["semantic"])

    report = {
        "volume": volume,
        "run_dir": str(run_dir),
        "total": total,
        "parsed": len(parsed),
        "missing": len(missing),
        "strict_correct": strict_correct,
        "strict_accuracy": round(strict_correct * 100.0 / total, 2),
        "semantic_correct": semantic_correct,
        "semantic_accuracy": round(semantic_correct * 100.0 / total, 2),
        "errors": total - semantic_correct,
        "max_errors_allowed_98": total - int(total * 0.98),
        "error_categories": dict(cats.most_common()),
        "top_confusions": [{"expected": k[0], "got": k[1], "count": v}
                           for k, v in confusions.most_common(15)],
        "windows": [{"window": f"D{w*100+1}-D{min((w+1)*100, total)}",
                     "accuracy": round(windows[w] * 100.0 / win_tot[w], 2),
                     "correct": windows[w], "total": win_tot[w]}
                    for w in sorted(win_tot)],
    }
    (run_dir / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    errs = [r for r in rows if not r["semantic"]]
    (run_dir / "errors.json").write_text(
        json.dumps(errs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
