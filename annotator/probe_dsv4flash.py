# -*- coding: utf-8 -*-
"""DeepSeek V4 Flash capability probe via the local opencode model pool.

Read-only with respect to the project: this script only talks to the local
gateway and writes its own result files under tmp/deepseek_probe/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
POOL = pathlib.Path.home() / ".config" / "opencode" / "opencode.jsonc"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def load_pool():
    raw = POOL.read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("//")
    )
    data = json.loads(stripped)
    provider = data["provider"]["sensenova-pool"]
    opts = provider["options"]
    return opts["baseURL"].rstrip("/"), opts["apiKey"], provider["models"]


BASE_URL, API_KEY, MODELS = load_pool()
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def chat(model, messages, tools=None, tool_choice=None, max_tokens=8192,
         temperature=0, timeout=900, response_format=None, retries=2):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if response_format:
        payload["response_format"] = response_format

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    last = None
    for attempt in range(retries + 1):
        started = time.time()
        try:
            with OPENER.open(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            elapsed = time.time() - started
            return json.loads(raw), elapsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            last = f"HTTP {exc.code}: {detail}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(last)


def usage_of(payload):
    u = payload.get("usage") or {}
    return {
        "prompt": u.get("prompt_tokens"),
        "completion": u.get("completion_tokens"),
        "total": u.get("total_tokens"),
        "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }


def stage_basic(model):
    msg = [{"role": "user", "content": "用一句话说明你是谁，并写出 17*23 的结果。"}]
    data, elapsed = chat(model, msg, max_tokens=2048)
    choice = data["choices"][0]
    return {
        "model": model,
        "declared": data.get("model"),
        "latency_s": round(elapsed, 2),
        "usage": usage_of(data),
        "finish_reason": choice.get("finish_reason"),
        "content": (choice["message"].get("content") or "")[:500],
        "reasoning_present": bool(
            choice["message"].get("reasoning_content")
            or choice["message"].get("reasoning")
        ),
    }


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_novel_lines",
        "description": "Read an inclusive line range from the novel. Line numbers are 1-based.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "first line, 1-based"},
                "count": {"type": "integer", "description": "how many lines"},
            },
            "required": ["start", "count"],
        },
    },
}]


def stage_tools(model):
    messages = [
        {"role": "system", "content": "你是一个只能通过工具读原文的标注助手。"},
        {"role": "user", "content": "先读取第 23 到 26 行原文，然后只回答这四行的行号。"},
    ]
    data, elapsed = chat(model, messages, tools=TOOLS, max_tokens=2048)
    message = data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    first = None
    if calls:
        first = {
            "name": calls[0]["function"]["name"],
            "arguments": calls[0]["function"]["arguments"],
        }
    round2 = None
    if calls:
        messages.append({"role": "assistant", "content": message.get("content") or "",
                         "tool_calls": calls})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                         "content": "23:「这是最后一件了吧？」\n24:「嗯，这里确实有……七十件。多谢惠顾。」\n25:「不，我们才要谢谢你呢。」\n26:「不过，我也因此拿到上等的皮草啊，我会再来的。」"})
        data2, elapsed2 = chat(model, messages, tools=TOOLS, max_tokens=2048)
        round2 = {
            "finish_reason": data2["choices"][0].get("finish_reason"),
            "content": (data2["choices"][0]["message"].get("content") or "")[:200],
            "more_tool_calls": len(data2["choices"][0]["message"].get("tool_calls") or []),
        }
    return {
        "model": model,
        "latency_s": round(elapsed, 2),
        "usage": usage_of(data),
        "tool_call_emitted": bool(calls),
        "first_call": first,
        "second_round": round2,
    }


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "speaker_labels",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "integer"},
                            "speaker": {"type": "string"},
                        },
                        "required": ["line", "speaker"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        },
    },
}


def stage_json(model):
    text = (
        "23:「这是最后一件了吧？」\n"
        "24:「嗯，这里确实有……七十件。多谢惠顾。」\n"
        "25:「不，我们才要谢谢你呢。」\n"
        "26:「不过，我也因此拿到上等的皮草啊，我会再来的。」\n"
    )
    messages = [
        {"role": "system", "content": "为每一行对话标注说话人，未知则填「未知」。"},
        {"role": "user", "content": text},
    ]
    out = {"model": model}
    try:
        data, elapsed = chat(model, messages, max_tokens=2048,
                             response_format=SCHEMA)
        content = data["choices"][0]["message"].get("content") or ""
        out.update({
            "mode": "json_schema",
            "latency_s": round(elapsed, 2),
            "usage": usage_of(data),
            "valid_json": None,
            "content": content[:400],
        })
        try:
            parsed = json.loads(content)
            out["valid_json"] = True
            out["parsed"] = parsed
        except Exception:  # noqa: BLE001
            out["valid_json"] = False
    except Exception as exc:  # noqa: BLE001
        out.update({"mode": "json_schema", "error": str(exc)[:300]})

    messages2 = [
        {"role": "system", "content": "只输出 JSON，不要任何解释文字。格式：{\"labels\":[{\"line\":23,\"speaker\":\"村民\"}]}"},
        {"role": "user", "content": text},
    ]
    data2, elapsed2 = chat(model, messages2, max_tokens=2048)
    content2 = data2["choices"][0]["message"].get("content") or ""
    out["prompt_only"] = {
        "latency_s": round(elapsed2, 2),
        "usage": usage_of(data2),
        "content": content2[:400],
    }
    try:
        json.loads(re.sub(r"^```(?:json)?|```$", "", content2.strip(), flags=re.M).strip())
        out["prompt_only"]["valid_json"] = True
    except Exception:  # noqa: BLE001
        out["prompt_only"]["valid_json"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["basic", "tools", "json", "all"])
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    stages = ["basic", "tools", "json"] if args.stage == "all" else [args.stage]
    report = {"base_url": BASE_URL, "models_available": MODELS, "results": {}}
    for name in stages:
        fn = {"basic": stage_basic, "tools": stage_tools, "json": stage_json}[name]
        print(f"[probe] {name} ...", flush=True)
        try:
            report["results"][name] = fn(args.model)
        except Exception as exc:  # noqa: BLE001
            report["results"][name] = {"error": str(exc)[:500]}
            print(f"  FAILED: {exc}"[:300], flush=True)

    out = RESULTS / "capability.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=2)[:4000])
    print(f"\n[probe] written to {out}")


if __name__ == "__main__":
    sys.exit(main())
