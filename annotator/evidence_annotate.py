# -*- coding: utf-8 -*-
"""Variant: force the model to cite the original-text line that supports each
label, in the same single call. Tests whether self-grounding reduces the
lead-character swaps that neither the old system nor the plain baseline fixed.

Output schema: 编号|说话人|证据行号|依据
依据 ∈ 发声动作 / 对话交替 / 心想 / 引用非对话 / 环境声 / 无直接证据
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
from probe_dsv4flash import BASE_URL, API_KEY, OPENER, RESULTS, usage_of  # noqa: E402
from twopass_annotate import build  # noqa: E402

SYSTEM = (
    "你是小说对话说话人标注员。你会拿到一整卷小说全文（行首为原书行号）与全部待标注引语的编号清单。\n"
    "【输出要求】每行输出四个字段，用 `|` 分隔：`编号|说话人|证据行号|依据`\n"
    "· 编号：清单里的编号，必须全部输出。\n"
    "· 说话人：有明确姓名时输出姓名；没有姓名时用原文出现过的稳定泛称；"
    "引号若不是某个角色当场说出口的话（被引用的词句或说法、神态姿态的字幕式引号、"
    "单独强调的词、环境声与拟声、无明确归属的引用），输出 `无人称引语`。\n"
    "· 证据行号：**原书行号**，必须是你判断该说话人所依据的那一行。\n"
    "   - 若依据是「某某说／某某答道」这类发声动作，填那一行。\n"
    "   - 若依据是对话交替，填上一个由另一人发言的行。\n"
    "   - 若依据是内心独白标记（心想／暗自说／脑海里），填那一行。\n"
    "   - 若确实是引用或环境声，填出现元语言标记或声响的那一行。\n"
    "   - 找不到任何依据时填 `0`，依据填 `无直接证据`。\n"
    "· 依据：只能填这六个之一 —— `发声动作` / `对话交替` / `心想` / `引用非对话` / `环境声` / `无直接证据`。\n"
    "【重要】先确定依据行号，再写说话人。依据行里必须真的能支持你的判断；"
    "不要先写答案再补一个行号。\n"
    "【常见错误】两人快速对话时把发言对调。判断时要盯住引语**紧邻**的发声动作，"
    "而不是整段的整体印象。\n"
    "【无人称引语·触发清单】以下任一成立，说话人填 `无人称引语`，依据填 `引用非对话`"
    "（环境声响则填 `环境声`）：\n"
    "  · 引号内容是被引用的词句／说法／俗语／概念，例如「也就是『不喜欢拖着空荡荡的货台旅行』」「所谓『生意』」；\n"
    "  · 无法挂住任何主体的神态／姿态字幕式引号（上下文无具体人物），例如「一副『不关我事』的模样」；\n"
    "  · 单独强调某个词或字，例如「加重语气说了『又』字」；\n"
    "  · 引号被当作语言材料来谈论，主语是「这句话」「这两个字」「这个说法」；\n"
    "  · 环境声、动物声、物体声、拟声词。\n"
    "【无人称引语·反例】以下情况**不要**判无人称引语：\n"
    "  · 短促的语气词／应答词／感叹词／省略号（「唔。」「咦？」「……」）——它们是角色当场发出的，是发言；\n"
    "  · 内心独白（心想／暗自说／脑海里浮现）——归那个主体，依据填 `心想`；\n"
    "  · 神态／姿态字幕式引号若上下文有明确主体（如「赫萝一副『真是拿你没辄』的模样」）——归该主体，依据填 `发声动作`。\n"
    "【铁律】找不到任何发声动作时，先认真考虑是不是 `无人称引语`，而不是硬把引语安给附近某个角色。"
    "附近出现人名不等于有人说话。\n"
    "不要输出解释、不要代码块。"
)

FEWSHOT_NONPERSON = (
    "【无人称引语·正例】下面这些引号都不归属于任何人的发言，说话人一律填 `无人称引语`：\n"
    "1. 莫大的恐惧袭上罗伦斯心头，仿佛突来的冷风「唰唰唰唰唰」窜入身体般。 → 无人称引语（拟声）\n"
    "2. 旅行商人会把前往新城镇挖掘值钱的商品说成「找老婆」，这句话也包含了…… → 无人称引语（被引用的说法）\n"
    "3. 赫萝的尾巴末端轻轻拍打了车板一下，发出「啪唰」一声。 → 无人称引语（声响）\n"
    "4. 赫萝瞪着罗伦斯，一副「别想逃跑」的模样拉住他的衣服。 → 无人称引语（神态字幕）\n"
    "5. 「因为太贪心，所以赔了钱」，这句话完完全全道出事实。 → 无人称引语（被谈论的句子）\n"
    "6. 她的眼神诉说着「你要是回答得不好，我会再踩下去」。 → 无人称引语（神态字幕）\n"
    "注意：这些例子里的引号都不是「谁当场说出口的话」，而是被引用、被形容、或被摹写的声音。\n"
)


def call(messages, max_tokens=65536, reasoning="none", timeout=3600, temperature=0.0):
    payload = {"model": "deepseek-v4-flash", "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if reasoning != "default":
        payload["reasoning_effort"] = reasoning
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
        method="POST")
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_rich(text):
    got = {}
    for line in text.splitlines():
        parts = [p.strip() for p in re.split(r"[|｜]", line.strip())]
        if len(parts) < 2:
            continue
        m = re.match(r"^D?(\d+)$", parts[0])
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        got[idx] = {
            "speaker": parts[1],
            "evidence_line": parts[2] if len(parts) > 2 else "",
            "basis": parts[3] if len(parts) > 3 else "",
        }
    return got


def parse_plain(text):
    got = {}
    for line in text.splitlines():
        parts = [p.strip() for p in re.split(r"[|｜]", line.strip())]
        if len(parts) < 2:
            continue
        m = re.match(r"^D?(\d+)$", parts[0])
        if not m:
            continue
        got[int(m.group(1))] = {"speaker": parts[1], "evidence_line": "", "basis": ""}
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=1)
    ap.add_argument("--tag", default="evidence")
    ap.add_argument("--columns", type=int, default=4, choices=[2, 4],
                    help="4 = require citation (default), 2 = ablation without citation")
    ap.add_argument("--with-ledger", action="store_true",
                    help="run the whole-volume character ledger first and inject it")
    ap.add_argument("--fewshot-nonperson", action="store_true",
                    help="inject in-domain positive examples of 无人称引语")
    ap.add_argument("--hint-file", default=None,
                    help="file with one quote index per line; these are flagged as "
                         "'maybe not speech' and the model must re-verify each")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (0 = greedy)")
    args = ap.parse_args()

    body, roster, quote_lines = build(args.volume)
    count = len(quote_lines)
    if args.columns == 4:
        system = SYSTEM + (FEWSHOT_NONPERSON if args.fewshot_nonperson else "")
        fmt = "编号|说话人|证据行号|依据"
    else:
        system = SYSTEM.split("【输出要求】")[0] + (
            "【输出要求】每行输出两个字段，用 `|` 分隔：`编号|说话人`\n"
            "· 说话人：有明确姓名时输出姓名；没有姓名时用原文出现过的稳定泛称；"
            "引号若不是某个角色当场说出口的话，输出 `无人称引语`。\n"
            "【常见错误】两人快速对话时把发言对调。判断时要盯住引语**紧邻**的发声动作，"
            "而不是整段的整体印象。\n"
            "不要输出解释、不要代码块。")
        fmt = "编号|说话人"

    ledger_text = ""
    ledger_entities = 0
    ledger_s = 0.0
    if args.with_ledger:
        from twopass_annotate import LEDGER_SYSTEM
        t0 = time.time()
        led = call([{"role": "system", "content": LEDGER_SYSTEM},
                    {"role": "user", "content":
                        f"第{args.volume}卷全文：\n\n{body}\n\n请输出角色账本 JSON。"}])
        ledger_s = time.time() - t0
        raw = led["choices"][0]["message"].get("content") or ""
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        start = cleaned.find("{")
        ledger = json.loads(cleaned[start:]) if start >= 0 else {"characters": []}
        ledger_entities = len(ledger.get("characters", []))
        ledger_text = (
            "\n\n本卷角色账本（说话人必须从下面的 canonical 名称中选，"
            "账本里没有的名字不得使用；同一角色全卷用同一个标签）：\n"
            + json.dumps(ledger, ensure_ascii=False, indent=1) + "\n")
        (pathlib.Path("results") / f"baseline_vol{args.volume}_{args.tag}").mkdir(
            parents=True, exist_ok=True)

    user = (f"下面是第{args.volume}卷正文，行首数字是原书行号：\n\n<<<正文>>>\n{body}\n"
            f"<<<正文结束>>>\n{ledger_text}\n待标注引语共 {count} 处：\n{roster}\n\n"
            f"请逐条输出 `{fmt}`，必须输出全部 {count} 条。")

    hint_text = ""
    if args.hint_file:
        hints = []
        for line in pathlib.Path(args.hint_file).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^D?(\d+)$", line.strip())
            if m:
                hints.append(int(m.group(1)))
        hint_text = (
            "\n\n【复核提示】有一个独立的\"引语用途\"判断器，它标记了下面这些编号的引语"
            "**可能不是任何角色当场说出口的话**（可能是：被引用的词句或说法、神态姿态的字幕式引号、"
            "单独强调的词、环境声与拟声、内心独白）：\n"
            + "、".join(f"D{i:04d}" for i in hints) + "\n"
            "请对上面每个编号**单独复核**：\n"
            "  · 若它确实是引用/神态字幕/强调词/环境声（不是发言），说话人填 `无人称引语`，"
            "依据填 `引用非对话` 或 `环境声`；\n"
            "  · 若它是内心独白（心想/暗自/脑海里），说话人填那个主体，依据填 `心想`；\n"
            "  · 若它其实是角色当场说出口的话，就正常归因，**不要因为它在提示清单里就机械判无人称引语**。\n"
        )
        user += hint_text

    started = time.time()
    data = call([{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=args.temperature)
    elapsed = time.time() - started
    content = data["choices"][0]["message"].get("content") or ""
    rich = parse_rich(content) if args.columns == 4 else parse_plain(content)

    out_dir = RESULTS / f"baseline_vol{args.volume}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_output.txt").write_text(
        "\n".join(f"D{i:04d}|{rich[i]['speaker']}" for i in sorted(rich)),
        encoding="utf-8")
    (out_dir / "raw_output_rich.txt").write_text(content, encoding="utf-8")

    # evidence diagnostics: does the cited line actually exist and contain a hint?
    novel = (pathlib.Path(__file__).resolve().parents[1] / "data"
             / ("" if args.volume == 1 else f"volume{args.volume}") / "novel.txt")
    if not novel.exists():
        novel = (pathlib.Path(__file__).resolve().parents[1] / "data"
                 / f"volume{args.volume}" / "novel.txt")
    lines = novel.read_text(encoding="utf-8").splitlines()
    cited_ok = sum(1 for v in rich.values()
                   if v["evidence_line"].isdigit()
                   and 0 < int(v["evidence_line"]) <= len(lines))
    speaker_in_cited = sum(
        1 for v in rich.values()
        if v["evidence_line"].isdigit() and 0 < int(v["evidence_line"]) <= len(lines)
        and v["speaker"] and v["speaker"] in lines[int(v["evidence_line"]) - 1])

    basis_counts = {}
    for v in rich.values():
        basis_counts[v["basis"]] = basis_counts.get(v["basis"], 0) + 1

    report = {
        "volume": args.volume, "model": "deepseek-v4-flash",
        "prompt_version": args.tag, "thinking": "none",
        "expected_quotes": count, "parsed_quotes": len(rich),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "latency_s": round(elapsed + ledger_s, 1), "usage": usage_of(data),
        "ledger_entities": ledger_entities,
        "ledger_latency_s": round(ledger_s, 1),
        "cited_line_valid": cited_ok,
        "speaker_appears_in_cited_line": speaker_in_cited,
        "basis_distribution": basis_counts,
        "missing": count - len(rich),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
