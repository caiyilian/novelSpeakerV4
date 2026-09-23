# -*- coding: utf-8 -*-
"""Scheme 7 (combined): two-layer arbitration.

Layer 1 = scheme 2's arbiter (cont vs evledger divergence, ±15 lines) -> `arb`.
Layer 2 = ONLY for entries where layer-1 (arb) and the scene-block channel disagree,
re-arbitrate with the FULL SCENE as context (borrowing scheme 1's scene-level view).

Why this design:
  - Scheme 2 (94.61/93.65%) is the best single scheme, but its arbiter sees only
    ±15 lines, so it misses "real-name swaps" (fast dialogue alternation).
  - Scheme 1 (scene block) has the scene-level view that fixes those swaps: of the
    102 entries where arb is wrong but scene is right, 86 (84%) are real-name swaps,
    concentrated in v2 (fast dialogue).
  - Naive combination fails (scheme 3: three channels at ±15 lines, the third
    channel's noise 201 > its signal 102, net -99).
  - This scheme narrows the scope (only the 415 arb-vs-scene divergences) AND gives
    layer 2 the scene-level view.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

import cont_agent as ca  # noqa: E402
from scene_sequence import build_scene_segments  # noqa: E402

ROOT = HERE.parents[0]
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
RES = HERE / "results"

ARB_SYSTEM = (
    "你是小说对话说话人标注的最终裁决员。下面给你一个**完整场景**的原文，以及该场景内若干条引语的"
    "两个候选答案（来自两个独立通道）。你要用**整个场景的对话交替结构**判断每条引语真正的说话人。\n"
    "【关键】盯住场景内的对话交替顺序：谁问了、谁答了、谁在接话。快速对话处最容易把两人对调。\n"
    "【答案形式】说话人用账本里的规范名，或原文里的稳定泛称；仅当这句引语**无法归属到任何具体人物**"
    "时才填 `无人称引语`。\n"
    "【输出铁律】每条**只输出一行**：`D编号|最终说话人|一句依据`。"
    "必须输出全部条目，不要解释、不要代码块、不要复述引语。"
)


def call(messages, max_tokens=4096, timeout=900):
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
    ap.add_argument("--tag", default="final")
    ap.add_argument("--max-scenes", type=int, default=0)
    args = ap.parse_args()

    vdir = ca.VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    answers = ca.read_answers(vdir / "answers.txt")
    identities = ca.rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    total = len(answers)
    ledger = json.loads((RES / f"baseline_vol{args.volume}_twopass" / "ledger.json")
                        .read_text(encoding="utf-8"))
    names = ca.ledger_names(ledger)

    dialogue = []
    for ln, line in enumerate(novel, 1):
        for m in re.finditer(r"「[^」]*」", line):
            dialogue.append((ln, m.group(0)))
    segments = build_scene_segments(novel, dialogue, max_turns=40, max_raw_span=240,
                                    hard_gap_lines=16, raw_padding=4)

    arb = parse(RES / f"baseline_vol{args.volume}_arb" / "raw_output.txt")
    scene = parse(RES / f"baseline_vol{args.volume}_scene" / "raw_output.txt")

    div = set(i for i in range(1, total + 1)
              if norm_label(arb.get(i, "")) != norm_label(scene.get(i, "")))

    scene_of = {}
    for si, seg in enumerate(segments):
        for idx in range(seg["start_index"] + 1, seg["end_index"] + 1):
            scene_of[idx] = si
    by_scene = defaultdict(list)
    for i in sorted(div):
        by_scene[scene_of.get(i, -1)].append(i)

    print(f"vol{args.volume}: arb vs scene 分歧 {len(div)} 条 → {len(by_scene)} 个场景", flush=True)

    verdict = {}
    t0 = time.time()
    scene_ids = sorted(k for k in by_scene if k >= 0)
    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]
    for n, si in enumerate(scene_ids, 1):
        items = by_scene[si]
        seg = segments[si]
        scene_text = "\n".join(f"{k}: {novel[k-1]}"
                               for k in range(seg["raw_start"], seg["raw_end"] + 1))
        lst = "\n".join(
            f"D{i:04d} 第{dialogue[i-1][0]}行：「{dialogue[i-1][1]}」  "
            f"候选A={arb.get(i, '?')}  候选B={scene.get(i, '?')}"
            for i in items)
        user = (f"【角色账本】\n{json.dumps(ledger, ensure_ascii=False)}\n\n"
                f"【待裁决条目】\n{lst}\n\n【本场景完整原文】\n{scene_text}\n\n"
                f"请用场景交替结构逐条裁决，输出 `D编号|最终说话人|依据`，共 {len(items)} 条。")
        try:
            data = call([{"role": "system", "content": ARB_SYSTEM},
                         {"role": "user", "content": user}])
            content = data["choices"][0]["message"].get("content") or ""
            v = parse_verdicts(content, names)
            for i in items:
                if i in v:
                    verdict[i] = v[i]
        except Exception as exc:  # noqa: BLE001
            print(f"  [scene {si}] {exc}"[:120], flush=True)
        if n % 10 == 0:
            print(f"  ...{n}/{len(scene_ids)} 场景 ({time.time()-t0:.0f}s, "
                  f"已裁决 {len(verdict)})", flush=True)

    results = {i: (verdict.get(i) or arb.get(i, "")) for i in range(1, total + 1)}
    ok_arb = sum(1 for j, a in enumerate(answers, 1)
                 if ca.semantic(a["parts"], arb.get(j, ""), identities))
    ok_fin = sum(1 for j, a in enumerate(answers, 1)
                 if ca.semantic(a["parts"], results.get(j, ""), identities))
    print(f"vol{args.volume}: arb {ok_arb}/{total}={round(ok_arb*100/total,2)}%  →  "
          f"两层仲裁 {ok_fin}/{total}={round(ok_fin*100/total,2)}%  "
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
