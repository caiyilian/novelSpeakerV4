# -*- coding: utf-8 -*-
"""Continuous multi-turn annotation agent with threshold-based compression.

A single rolling conversation. Each quote is labeled in turn with the full prior
context (alternation tracking fixes 实名互换). When the context grows past a
token budget (e.g. ~60-70% of the model window), it is COMPRESSED — not thrown
away: the history is distilled into a character ledger + recent speaker anchors,
then the conversation continues on top of the distilled state.

Resumable: a checkpoint (results + ledger + anchors) is written every N quotes so
an interrupted run can continue instead of restarting.
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
from probe_dsv4flash import BASE_URL, API_KEY, OPENER  # noqa: E402
from twopass_annotate import VOLUME_DIRS  # noqa: E402
import run_label as rl  # noqa: E402

ROOT = HERE.parents[0]
BACKUP = ROOT / "backup" / "2026-09-01_205124_post_native256k_full_volume_review_vol1-5"
MARKER_RE = re.compile(r"【([^】]+)】")

SYSTEM = (
    "你是小说对话说话人标注员，按编号顺序逐条标注。\n"
    "每条我会给你该引语的原文上下文（行首为原书行号）。\n"
    "【身份追踪】若某角色此刻未公布姓名（只知「女孩」「商人」），但后文揭示了真名，"
    "则这里也标真名（我会在后续给出账本提示，你尽量回填）。全文未揭示就用原文稳定泛称。\n"
    "【无人称引语】仅当这句引语**无法归属到任何具体人物**时才标：纯环境声/动物声/物体声、"
    "被当作语言材料谈论的词句（商品名/说法/概念）、无明确主体的引用。\n"
    "若上下文有「某人 + 动作/神态/心理/言说」能挂住这句引语（包括神态字幕式引号、"
    "单独强调的词），就归该主体，**不要**标无人称引语。\n"
    "短语气词/应答词/省略号仍是发言；内心独白归该主体。"
    "【输出铁律】每条**只输出一行**：`说话人|证据行号|理由`。"
    "绝不要复述引语、不要写「第X行是」、不要多行。"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_novel_lines",
        "description": "读取小说指定行范围（1-based 闭区间），查证远处线索。",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer"},
                "count": {"type": "integer"},
            },
            "required": ["start", "count"],
        },
    },
}]

DISTILL_SYSTEM = (
    "你是角色账本维护助手。把当前账本与最近一批标注合并，输出更新后的账本 JSON：\n"
    '{"characters":[{"canonical":"赫萝","aliases":["女孩","贤狼"],"note":"前面称女孩"}]}\n'
    "只合并别名、记录身份揭示、补充新说话人。只输出 JSON，尽量紧凑。"
)


def call(messages, tools=None, max_tokens=1024, timeout=300):
    payload = {"model": "deepseek-v4-flash", "messages": messages,
               "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "none"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_tool(novel, tc):
    args = json.loads(tc["function"]["arguments"])
    start = int(args.get("start", 1))
    count = min(int(args.get("count", 10)), 60)
    lo = max(1, start)
    hi = min(len(novel), lo + count - 1)
    return "\n".join(f"{k}: {novel[k-1]}" for k in range(lo, hi + 1))


def read_answers(path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MARKER_RE.finditer(line):
            raw = m.group(1).strip()
            items.append({"parts": {p.strip() for p in raw.split("|") if p.strip()}})
    return items


def semantic(parts, label, identities):
    p = {x.strip() for x in label.split("|") if x.strip()}
    p = {"非人物发声" if x == "无人称引语" else x for x in p}
    if parts & p:
        return True
    return rl._validation_lenient_match(parts, p, verified_identities=identities)[0]


PROMPT_WORD_BLACKLIST = {"说话人", "理由", "证据", "依据", "标签", "人物", "角色",
                          "对话", "旁白", "内容", "引语", "编号"}


def strict_parse(content, names):
    first_line = content.strip().splitlines()[0].strip() if content.strip() else ""
    parts = [p.strip() for p in re.split(r"[|｜]", first_line)]
    if parts and parts[0]:
        cand = parts[0]
        # reject IDs / line refs the model sometimes echoes back
        if re.match(r"^D?\d{1,6}$", cand) or re.match(r"^第\s*\d+\s*行", cand):
            return ""
        if cand in PROMPT_WORD_BLACKLIST:
            return ""
        for n in sorted(names, key=len, reverse=True):
            if cand == n or n in cand:
                return n
        if 1 <= len(cand) <= 12 and not any(x in cand for x in "：，。？!、的了吗行"):
            return cand
    return ""


def label_one(novel, messages, names):
    content = ""
    prompt_tokens = 0
    for attempt in range(3):
        try:
            data = call(messages, tools=TOOLS)
        except Exception as exc:  # noqa: BLE001
            if attempt < 2:
                time.sleep(3)
                continue
            return "", f"call failed: {exc}", prompt_tokens
        # the LAST call's prompt_tokens reflects the current context size
        prompt_tokens = (data.get("usage") or {}).get("prompt_tokens", 0)
        msg = data["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if tcs and attempt < 2:
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": tcs})
            for tc in tcs:
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": run_tool(novel, tc)})
            continue
        content = msg.get("content") or ""
        messages.append({"role": "assistant", "content": content})
        sp = strict_parse(content, names)
        if sp:
            return sp, content, prompt_tokens
        if attempt < 2:
            messages.append({"role": "user",
                             "content": "请直接输出姓名（一个词），例如：罗伦斯。不要加任何其他字或标点。"})
    return "", content, prompt_tokens


def distill(ledger, block_results):
    lines = [f"D{i}: {sp}" for i, sp in sorted(block_results.items()) if sp]
    if not lines:
        return ledger
    user = (f"当前账本：\n{json.dumps(ledger, ensure_ascii=False)}\n\n"
            f"最近标注：\n" + "\n".join(lines[:100]) + "\n\n请输出更新后的账本 JSON。")
    try:
        data = call([{"role": "system", "content": DISTILL_SYSTEM},
                     {"role": "user", "content": user}], max_tokens=2048)
        content = data["choices"][0]["message"].get("content") or ""
        c = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        s = c.find("{")
        if s >= 0:
            parsed = json.loads(c[s:])
            if parsed.get("characters"):
                return parsed
    except Exception as exc:  # noqa: BLE001
        print(f"  [distill] {exc}"[:120], flush=True)
    return ledger


def ledger_names(ledger):
    names = {"无人称引语", "非人物发声", "旁白"}
    for c in ledger.get("characters", []):
        for a in [c.get("canonical", ""), c.get("full_name", "")] + list(c.get("aliases", [])):
            if a:
                names.add(a)
    return names


def context_of(novel, ln, radius=8):
    lo = max(1, ln - radius)
    hi = min(len(novel), ln + radius)
    return "\n".join(f"{k}: {novel[k-1]}" for k in range(lo, hi + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--compress-tokens", type=int, default=600000,
                    help="compress when accumulated prompt_tokens exceed this "
                         "(deepseek-v4-flash window is 1M; 60% safety margin)")
    ap.add_argument("--tag", default="cont")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    vdir = VOLUME_DIRS[args.volume]
    novel = (vdir / "novel.txt").read_text(encoding="utf-8").splitlines()
    answers = read_answers(vdir / "answers.txt")
    identities = rl._load_verified_validation_identities(
        str(BACKUP / f"volume{args.volume}" / "evidence_vault.json"))
    end = args.end or len(answers) + 1

    qlines = []
    for ln, l in enumerate(novel, 1):
        for _ in re.finditer(r"「[^」]*」", l):
            qlines.append(ln)

    ckpt = HERE / "results" / f"cont_vol{args.volume}.ckpt.json"
    ledger = {"characters": []}
    results = {}
    recent = []
    cursor = args.start
    if args.resume and ckpt.exists():
        d = json.loads(ckpt.read_text(encoding="utf-8"))
        ledger = d.get("ledger", ledger)
        results = {int(k): v for k, v in d.get("results", {}).items()}
        recent = d.get("recent", [])
        cursor = d.get("cursor", args.start)
        print(f"[resume] 从 D{cursor} 续跑，已有 {len(results)} 条", flush=True)

    # pre-load the static twopass ledger if present (better identity baseline)
    static = HERE / "results" / f"baseline_vol{args.volume}_twopass" / "ledger.json"
    if static.exists() and not ledger.get("characters"):
        try:
            ledger = json.loads(static.read_text(encoding="utf-8"))
        except Exception:
            pass

    names = ledger_names(ledger)
    messages = [{"role": "system", "content": SYSTEM}]
    if ledger.get("characters"):
        messages.append({"role": "user", "content":
                         "【本卷角色账本】\n" + json.dumps(ledger, ensure_ascii=False)})
    if recent:
        messages.append({"role": "user", "content":
                         "【最近已标注】\n" + "\n".join(f"D{i}: {s}" for i, s in recent)})

    t0 = time.time()
    i = cursor
    total_tokens = 0
    while i < end:
        ln = qlines[i - 1]
        messages.append({"role": "user",
                         "content": f"标注 D{i:04d}（第 {ln} 行）。原文上下文：\n{context_of(novel, ln)}"
                                    f"\n只输出一行 `说话人|证据行号|理由`。"})
        sp, _, pt = label_one(novel, messages, names)
        total_tokens = pt  # pt is the current context size (last call's prompt_tokens)
        results[i] = sp
        recent.append((i, sp))
        recent = recent[-20:]
        if sp and sp not in names:
            names.add(sp)

        if i % 50 == 0:
            ckpt.write_text(json.dumps({"ledger": ledger, "results": results,
                                        "recent": recent, "cursor": i + 1},
                                       ensure_ascii=False), encoding="utf-8")
            print(f"  ...D{i} ({time.time()-t0:.0f}s, 账本 {len(ledger.get('characters',[]))}, "
                  f"累计 {total_tokens} tok)", flush=True)

        # token-threshold compression (borrowed from Codex local compaction):
        # keep recent user context + recent anchors, distill the annotation
        # history into the character ledger, replace the rest.
        if total_tokens > args.compress_tokens:
            block = {j: results[j] for j in range(args.start, i + 1)}
            ledger = distill(ledger, block)
            names = ledger_names(ledger)
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content":
                         "【本卷角色账本（压缩后）】\n" + json.dumps(ledger, ensure_ascii=False)},
                        {"role": "user", "content":
                         "【最近已标注】\n" + "\n".join(f"D{j}: {s}" for j, s in recent)}]
            total_tokens = 0
            print(f"  [压缩] D{i} 处压缩上下文（累计 {total_tokens + args.compress_tokens} tok）",
                  flush=True)
        i += 1

    ok = sum(1 for j, a in enumerate(answers, 1)
             if args.start <= j < end and semantic(a["parts"], results.get(j, ""), identities))
    rng = end - args.start
    print(f"vol{args.volume} D{args.start}-D{end-1}: 连续多轮 {ok}/{rng} = "
          f"{round(ok*100/rng,1)}%  耗时 {time.time()-t0:.0f}s", flush=True)

    out_dir = HERE / "results" / f"baseline_vol{args.volume}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_output.txt").write_text(
        "\n".join(f"D{j:04d}|{results.get(j, '')}" for j in range(1, len(answers) + 1)),
        encoding="utf-8")
    (out_dir / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
