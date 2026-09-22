# -*- coding: utf-8 -*-
"""Scheme 1 (docs/discussion/..._0931): natural scene-block annotation.

Each natural scene (split at strong narrative boundaries, target ~10-40 quotes)
is annotated in its own multi-turn conversation over the FULL scene text. This
gives the scene-level global view that the baseline ±8-line windows lack, while
still tracking turn alternation inside the scene. A character ledger (pre-loaded
from twopass) plus prior-scene anchors carry identity state across scenes, so no
large-window compression is needed.

Differs from cont_agent.py (baseline) in ONE axis: context = full scene instead
of ±8 lines, and the conversation is per-scene instead of one full-volume roll.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

import cont_agent as ca  # noqa: E402  (call / strict_parse / run_tool / read_answers / semantic / ledger_names)
from scene_sequence import build_scene_segments  # noqa: E402

ROOT = HERE.parents[0]
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
MARKER_RE = re.compile(r"【([^】]+)】")

SCENE_SYSTEM = (
    "你是小说对话说话人标注员，一次处理一个完整场景（连续对话段落）。\n"
    "我会先给你本场景的完整原文（行首是原书行号），再按编号逐条请你标注其中每处引语的说话人。\n"
    "【身份追踪】若某角色此刻未公布姓名（只知「女孩」「商人」），但后文揭示了真名，"
    "则这里也标真名（参考账本回填）。全文未揭示就用原文稳定泛称。\n"
    "【无人称引语】仅当这句引语**无法归属到任何具体人物**时才标：纯环境声/动物声/物体声、"
    "被当作语言材料谈论的词句（商品名/说法/概念）、无明确主体的引用。\n"
    "若上下文有「某人 + 动作/神态/心理/言说」能挂住这句引语（包括神态字幕式引号、"
    "单独强调的词），就归该主体，**不要**标无人称引语。\n"
    "短语气词/应答词/省略号仍是发言；内心独白归该主体。"
    "【输出铁律】每条**只输出一行**：`说话人|证据行号|理由`。"
    "绝不要复述引语、不要写「第X行是」、不要多行。"
)


def build_dialogue_list(novel):
    dialogue = []
    for ln, line in enumerate(novel, 1):
        for m in re.finditer(r"「[^」]*」", line):
            dialogue.append((ln, m.group(0)))
    return dialogue


def scene_text(novel, raw_start, raw_end):
    return "\n".join(f"{k}: {novel[k-1]}" for k in range(raw_start, raw_end + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--tag", default="scene")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="max quotes per scene block (0931: 10-40)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    vdir = ca.VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    answers = ca.read_answers(vdir / "answers.txt")
    identities = ca.rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    total = len(answers)

    dialogue_list = build_dialogue_list(novel)
    segments = build_scene_segments(
        novel, dialogue_list,
        max_turns=args.max_turns, max_raw_span=240, hard_gap_lines=16, raw_padding=4)

    # ledger: pre-load twopass static ledger (same as cont_agent baseline)
    ledger = {"characters": []}
    static = HERE / "results" / f"baseline_vol{args.volume}_twopass" / "ledger.json"
    if static.exists():
        try:
            ledger = json.loads(static.read_text(encoding="utf-8"))
        except Exception:
            pass
    names = ca.ledger_names(ledger)

    results = {}
    prior_anchors = []  # last 3 labeled (D, speaker) from previous scene
    done_scenes = set()
    ckpt = HERE / "results" / f"scene_vol{args.volume}.ckpt.json"
    if args.resume and ckpt.exists():
        d = json.loads(ckpt.read_text(encoding="utf-8"))
        results = {int(k): v for k, v in d.get("results", {}).items()}
        prior_anchors = d.get("prior_anchors", [])
        done_scenes = set(d.get("done_scenes", []))
        print(f"[resume] 已有 {len(results)} 条，已完成场景 {len(done_scenes)}", flush=True)

    t0 = time.time()
    for seg in segments:
        if seg["scene_id"] in done_scenes:
            continue
        start_idx = seg["start_index"]          # 0-based dialogue index
        end_idx = seg["end_index"]              # 0-based exclusive
        d_lo, d_hi = start_idx + 1, end_idx     # 1-based D ids
        raw_start, raw_end = seg["raw_start"], seg["raw_end"]

        messages = [{"role": "system", "content": SCENE_SYSTEM}]
        if ledger.get("characters"):
            messages.append({"role": "user", "content":
                             "【本卷角色账本】\n" + json.dumps(ledger, ensure_ascii=False)})
        if prior_anchors:
            messages.append({"role": "user", "content":
                             "【上一场景最后标注】\n" +
                             "\n".join(f"D{i}: {s}" for i, s in prior_anchors)})
        messages.append({"role": "user", "content":
                         "【本场景完整原文】\n" + scene_text(novel, raw_start, raw_end)})

        for i in range(d_lo, d_hi + 1):
            ln = dialogue_list[i - 1][0]
            messages.append({"role": "user", "content":
                             f"标注 D{i:04d}（第 {ln} 行）。只输出一行 `说话人|证据行号|理由`。"})
            sp, _, _ = ca.label_one(novel, messages, names)
            results[i] = sp
            if sp and sp not in names:
                names.add(sp)

        prior_anchors = [(j, results[j]) for j in range(max(1, d_hi - 2), d_hi + 1)
                         if j in results]
        done_scenes.add(seg["scene_id"])

        if len(done_scenes) % 10 == 0:
            ckpt.write_text(json.dumps({"results": results, "prior_anchors": prior_anchors,
                                        "done_scenes": sorted(done_scenes)},
                                       ensure_ascii=False), encoding="utf-8")
            print(f"  [{len(done_scenes)}/{len(segments)} 场景] D{d_hi} "
                  f"({time.time()-t0:.0f}s, 账本 {len(ledger.get('characters', []))})", flush=True)

    ok = sum(1 for j, a in enumerate(answers, 1)
             if ca.semantic(a["parts"], results.get(j, ""), identities))
    print(f"vol{args.volume}: 场景块 {ok}/{total} = {round(ok*100/total, 2)}%  "
          f"耗时 {time.time()-t0:.0f}s", flush=True)

    out_dir = HERE / "results" / f"baseline_vol{args.volume}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_output.txt").write_text(
        "\n".join(f"D{j:04d}|{results.get(j, '')}" for j in range(1, total + 1)),
        encoding="utf-8")
    (out_dir / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
