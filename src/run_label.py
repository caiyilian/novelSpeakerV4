# -*- coding: utf-8 -*-
"""
Multi-agent novel dialogue speaker annotation system v4

Architecture:
  Python orchestrator -> Labeler (tool calling, English prompt)
                             -> ShortMem (recent N rounds)
                             -> CharacterState (validated, clean)

Key design principles:
  1. Python controls flow - Agents are read-only
  2. One dialogue per round - quality over speed
  3. English prompt - LLM reasons better in English
  4. Strict character name validation - no polluted state
  5. Token budget tracking - prevent context overflow
  6. Active backward search for unnamed characters (girl -> real name)
"""
import sys, io

import sys
import os
import re
import json
import ast
import requests
import argparse
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from dialogue_ensemble import (
    build_dialogue_packet,
    citations_are_local,
    format_dialogue_packet,
    get_tag as get_ensemble_tag,
    parse_line_citations,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Read Ollama config
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "ip_config")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:32b"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OLLAMA_BASE_URL="):
                OLLAMA_BASE_URL = line.split("=", 1)[1]
            elif line.startswith("OLLAMA_MODEL="):
                OLLAMA_MODEL = line.split("=", 1)[1]

DATA_DIR = os.path.join(ROOT_DIR, "data")
NOVEL_PATH = os.path.join(DATA_DIR, "novel.txt")
LABELED_PATH = os.path.join(DATA_DIR, "labeled.txt")
ANSWERS_PATH = os.path.join(DATA_DIR, "answers.txt")
LOG_PATH = os.path.join(DATA_DIR, "label_log.jsonl")
TEMP_LOG_PATH = os.path.join(DATA_DIR, "label_log.temp.jsonl")
STATE_PATH = os.path.join(DATA_DIR, "character_state.json")
VAULT_PATH = os.path.join(DATA_DIR, "evidence_vault.json")

# Token budget: stop searching when cumulative tokens exceed this
TOKEN_BUDGET = 18000
MAX_TOOL_ROUNDS = 10
MODEL_PROVIDER = "ollama"
API_MODEL_FILTER = ""
API_MODEL_PRIORITY = []
API_CONTEXT_LIMIT = 40000
API_MAX_OUTPUT_TOKENS = 2048
API_RETRIES = 3
API_RETRY_DELAY = 5.0
API_CALL_TRACE = []
CURRENT_ROUND_TRACE = None
API_ROUND_ROBIN_CURSOR = {}
SENSENOVA_MODEL = "sensenova-6.7-flash-lite"
SENSENOVA_KEYS_PATH = os.path.join(ROOT_DIR, "config", "other_sensenova_apikeys")
DECISION_MODE = os.environ.get("NOVEL_DECISION_MODE", "quality").strip().lower()


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DIALOGUE_BLOCK_RADIUS = _env_int("NOVEL_DIALOGUE_BLOCK_RADIUS", 4)
QUALITY_REQUEST_TIMEOUT = _env_int("NOVEL_QUALITY_REQUEST_TIMEOUT", 120)
QUALITY_SCENE_RADIUS = _env_int("NOVEL_QUALITY_SCENE_RADIUS", 12)
QUALITY_AUDIT_RETRIES = max(1, _env_int("NOVEL_QUALITY_AUDIT_RETRIES", 2))


def _resolve_workspace_path(path_value):
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(ROOT_DIR) / path
    return str(path)


def configure_paths(data_dir=None, novel_path=None, answers_path=None,
                    labeled_path=None, log_path=None, state_path=None, vault_path=None):
    """Configure per-run input/output paths so parallel volumes never share state."""
    global DATA_DIR, NOVEL_PATH, LABELED_PATH, ANSWERS_PATH, LOG_PATH, TEMP_LOG_PATH, STATE_PATH, VAULT_PATH

    if data_dir:
        DATA_DIR = _resolve_workspace_path(data_dir)
    else:
        DATA_DIR = os.path.join(ROOT_DIR, "data")

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    NOVEL_PATH = _resolve_workspace_path(novel_path) if novel_path else os.path.join(DATA_DIR, "novel.txt")
    ANSWERS_PATH = _resolve_workspace_path(answers_path) if answers_path else os.path.join(DATA_DIR, "answers.txt")
    LABELED_PATH = _resolve_workspace_path(labeled_path) if labeled_path else os.path.join(DATA_DIR, "labeled.txt")
    LOG_PATH = _resolve_workspace_path(log_path) if log_path else os.path.join(DATA_DIR, "label_log.jsonl")
    log_base, log_ext = os.path.splitext(LOG_PATH)
    TEMP_LOG_PATH = f"{log_base}.temp{log_ext or '.jsonl'}"
    STATE_PATH = _resolve_workspace_path(state_path) if state_path else os.path.join(DATA_DIR, "character_state.json")
    VAULT_PATH = _resolve_workspace_path(vault_path) if vault_path else os.path.join(DATA_DIR, "evidence_vault.json")

    for output_path in (LABELED_PATH, LOG_PATH, TEMP_LOG_PATH, STATE_PATH, VAULT_PATH):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)


class ModelCallError(RuntimeError):
    pass


class ApiModel:
    def __init__(self, name, model, base_url, api_key="", min_interval=1.5,
                 tool_capable=True, display_model=None, use_env_proxy=True,
                 round_robin_group=""):
        self.name = name
        self.model = model
        self.display_model = display_model or model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.min_interval = min_interval
        self.tool_capable = tool_capable
        self.use_env_proxy = use_env_proxy
        self.last_call_at = 0.0
        self.cooldown_until = 0.0
        self.last_error = ""
        self.disabled = False
        self.round_robin_group = round_robin_group

    @property
    def label(self):
        return f"{self.name}/{self.display_model}"

    @property
    def chat_url(self):
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1") or self.base_url.endswith("/paas/v4"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def wait_for_slot(self):
        delay = self.min_interval - (time.time() - self.last_call_at)
        if delay > 0:
            time.sleep(delay)

    def mark_failure(self, message, cooldown=30):
        self.last_error = message[:200]
        self.cooldown_until = time.time() + cooldown
        if "HTTP 401" in message or "HTTP 403" in message:
            self.disabled = True

    def available(self, needs_tools):
        if self.disabled:
            return False
        if needs_tools and not self.tool_capable:
            return False
        return time.time() >= self.cooldown_until


API_MODELS = []


def _load_opencode_provider(provider_name):
    path = Path.home() / ".config" / "opencode" / "opencode.jsonc"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        lines = [line for line in raw.splitlines() if not line.strip().startswith("//")]
        data = json.loads("\n".join(lines))
        return data.get("provider", {}).get(provider_name)
    except Exception:
        return None


def _extract_api_key_from_python(path, var_name="API_KEY"):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    pattern = rf'{re.escape(var_name)}\s*=\s*["\']([^"\']+)["\']'
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _load_zhipu_key():
    for env_name in ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "BIGMODEL_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value
    key_file = os.environ.get("ZHIPUAI_API_KEY_FILE")
    if key_file:
        key = _extract_api_key_from_python(key_file)
        if key:
            return key
        try:
            return Path(key_file).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    free_api_root = os.environ.get("FREE_API_ROOT") or str(Path(ROOT_DIR).parent / "free-api")
    return _extract_api_key_from_python(Path(free_api_root) / "tests" / "test_zhipu.py")


def _read_key_file(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_agnes_key():
    value = os.environ.get("AGNES_API_KEY")
    if value:
        return value.strip()
    key_file = os.environ.get("AGNES_API_KEY_FILE")
    if key_file:
        return _read_key_file(key_file)
    local_key_file = Path(ROOT_DIR) / "config" / "agnes_api_key"
    if local_key_file.exists():
        return _read_key_file(local_key_file)
    provider = _load_opencode_provider("agnes")
    if provider:
        return (provider.get("options", {}).get("apiKey") or "").strip()
    return ""


def _load_sensenova_keys():
    key_file = os.environ.get("SENSENOVA_API_KEYS_FILE") or SENSENOVA_KEYS_PATH
    try:
        lines = Path(key_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    keys = []
    seen = set()
    for line in lines:
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item in seen:
            continue
        seen.add(item)
        keys.append(item)
    return keys


def _suffix_name(index):
    return str(index + 1)


def _append_sensenova_models(models):
    provider = _load_opencode_provider("sense-nova")
    if not provider:
        return

    opts = provider.get("options", {}) or {}
    configured_models = provider.get("models", {}) or {}
    base_url = opts.get("baseURL", "") or opts.get("baseUrl", "") or opts.get("base_url", "")
    if SENSENOVA_MODEL not in configured_models or not base_url:
        return

    model_conf = configured_models.get(SENSENOVA_MODEL) or {}
    actual_model = model_conf.get("id") or SENSENOVA_MODEL
    use_env_proxy = os.environ.get("SENSENOVA_USE_ENV_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}
    file_keys = _load_sensenova_keys()
    if file_keys:
        for index, api_key in enumerate(file_keys):
            suffix = _suffix_name(index)
            models.append(ApiModel(
                f"sense-nova-{suffix}",
                actual_model,
                base_url,
                api_key,
                min_interval=1.5,
                tool_capable=True,
                display_model=f"{SENSENOVA_MODEL}-{suffix}",
                use_env_proxy=use_env_proxy,
                round_robin_group="sense-nova",
            ))
        return

    api_key = opts.get("apiKey", "") or opts.get("api_key", "")
    if api_key:
        models.append(ApiModel("sense-nova", actual_model, base_url, api_key,
                               min_interval=1.5, tool_capable=True,
                               display_model=SENSENOVA_MODEL,
                               use_env_proxy=use_env_proxy))


def _model_matches_priority(model, priority_item):
    item = (priority_item or "").strip().lower()
    if not item:
        return False
    fields = (model.label.lower(), model.model.lower(), model.name.lower())
    if item in fields:
        return True
    return any(item in field for field in fields)


def _apply_api_priority(models):
    if not API_MODEL_PRIORITY:
        return models

    ranked = []
    for original_index, model in enumerate(models):
        rank = len(API_MODEL_PRIORITY)
        for priority_index, priority_item in enumerate(API_MODEL_PRIORITY):
            if _model_matches_priority(model, priority_item):
                rank = priority_index
                break
        ranked.append((rank, original_index, model))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [model for _, _, model in ranked]


def _api_model_iteration_order():
    """Return API models with same-provider key pools rotated per request."""
    if not API_MODELS:
        return []

    ordered = []
    emitted_groups = set()
    for model in API_MODELS:
        group = model.round_robin_group
        if not group:
            ordered.append(model)
            continue
        if group in emitted_groups:
            continue
        emitted_groups.add(group)
        group_models = [m for m in API_MODELS if m.round_robin_group == group]
        if len(group_models) <= 1:
            ordered.extend(group_models)
            continue
        cursor = API_ROUND_ROBIN_CURSOR.get(group, 0) % len(group_models)
        API_ROUND_ROBIN_CURSOR[group] = (cursor + 1) % len(group_models)
        ordered.extend(group_models[cursor:] + group_models[:cursor])
    return ordered


def _append_opencode_model(models, provider_name, model_name, min_interval=2.0):
    provider = _load_opencode_provider(provider_name)
    if not provider:
        return
    opts = provider.get("options", {}) or {}
    configured_models = provider.get("models", {}) or {}
    base_url = opts.get("baseURL", "") or opts.get("baseUrl", "") or opts.get("base_url", "")
    api_key = opts.get("apiKey", "") or opts.get("api_key", "")
    if model_name in configured_models and base_url:
        model_conf = configured_models.get(model_name) or {}
        actual_model = model_conf.get("id") or model_name
        models.append(ApiModel(provider_name, actual_model, base_url, api_key,
                               min_interval=min_interval, tool_capable=True,
                               display_model=model_name))


def _build_api_models():
    models = []

    # Priority is intentional, not random. Keep low-context models such as
    # GLM-4V-Flash (16K) out of this pool because experiments target 40K.
    _append_sensenova_models(models)

    longcat_provider = _load_opencode_provider("longcat")
    if longcat_provider:
        opts = longcat_provider.get("options", {})
        configured_models = longcat_provider.get("models", {})
        if "LongCat-2.0" in configured_models:
            models.append(ApiModel("longcat", "LongCat-2.0",
                                   opts.get("baseURL", ""), opts.get("apiKey", ""),
                                   min_interval=2.0, tool_capable=True))

    agnes_key = _load_agnes_key()
    if agnes_key:
        models.append(ApiModel("agnes", "agnes-2.0-flash", "https://apihub.agnes-ai.com/v1",
                               agnes_key, min_interval=1.5, tool_capable=True))

    zhipu_key = _load_zhipu_key()
    if zhipu_key:
        base = "https://open.bigmodel.cn/api/paas/v4"
        models.extend([
            ApiModel("zhipu", "GLM-4-Flash-250414", base, zhipu_key, min_interval=1.5, tool_capable=True),
            ApiModel("zhipu", "GLM-4-Flash", base, zhipu_key, min_interval=1.5, tool_capable=True),
            ApiModel("zhipu", "GLM-4.6V-Flash", base, zhipu_key, min_interval=3.0, tool_capable=True),
        ])

    # Extra free DeepSeek-v4-flash endpoints from opencode/forwarded LAN providers.
    # They are added after the original pool so the default order stays stable.
    # Use --api-priority to move them to the front for dedicated multi-volume runs.
    _append_opencode_model(models, "atomcode-proxy", "deepseek-v4-flash", min_interval=2.0)
    _append_opencode_model(models, "lan-237", "deepseek-v4-flash-free-237", min_interval=2.0)
    _append_opencode_model(models, "lan-162", "deepseek-v4-flash-free-162", min_interval=2.0)
    _append_opencode_model(models, "lan-171", "deepseek-v4-flash-free-171", min_interval=2.0)
    _append_opencode_model(models, "lan-189", "deepseek-v4-flash-free-189", min_interval=2.0)

    if API_MODEL_FILTER:
        lowered = API_MODEL_FILTER.lower()
        exact_models = [
            model for model in models
            if lowered == model.label.lower() or lowered == model.model.lower() or lowered == model.name.lower()
        ]
        if exact_models:
            models = exact_models
        else:
            models = [
                model for model in models
                if lowered in model.label.lower() or lowered in model.model.lower() or lowered in model.name.lower()
            ]
    models = _apply_api_priority(models)
    return models


def _message_content(message):
    return (message.get("content") or message.get("reasoning") or
            message.get("reasoning_content") or "")


def _estimate_message_tokens(messages):
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(content)
        for tc in msg.get("tool_calls") or []:
            total += len(json.dumps(tc, ensure_ascii=False))
    return total


def _openai_messages(messages):
    converted = []
    pending_tool_ids = []
    for msg in messages:
        role = msg.get("role")
        item = {"role": role, "content": msg.get("content") or ""}
        if role == "assistant" and msg.get("tool_calls"):
            item["tool_calls"] = msg["tool_calls"]
            pending_tool_ids = [tc.get("id") for tc in msg["tool_calls"] if tc.get("id")]
        elif role == "tool":
            if pending_tool_ids:
                item["tool_call_id"] = pending_tool_ids.pop(0)
            else:
                item["tool_call_id"] = "tool_call_0"
        converted.append(item)
    return converted


def _api_chat(model, messages, tools=None, tool_choice="auto", request_timeout=300):
    estimated_tokens = _estimate_message_tokens(messages)
    if estimated_tokens + API_MAX_OUTPUT_TOKENS > API_CONTEXT_LIMIT:
        raise ModelCallError(
            f"context budget exceeded: estimated {estimated_tokens + API_MAX_OUTPUT_TOKENS} > {API_CONTEXT_LIMIT}"
        )
    model.wait_for_slot()
    headers = {"Content-Type": "application/json"}
    if model.api_key:
        headers["Authorization"] = model.api_key if model.api_key.lower().startswith("bearer ") else f"Bearer {model.api_key}"
    payload = {
        "model": model.model,
        "messages": _openai_messages(messages),
        "temperature": 0,
        "max_tokens": API_MAX_OUTPUT_TOKENS,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    with requests.Session() as session:
        session.trust_env = model.use_env_proxy
        resp = session.post(model.chat_url, headers=headers, json=payload, timeout=request_timeout)
    model.last_call_at = time.time()
    if resp.status_code == 429:
        model.mark_failure("rate limited", cooldown=120)
        raise ModelCallError(f"{model.label}: HTTP 429 rate limited")
    if resp.status_code >= 500:
        model.mark_failure(f"HTTP {resp.status_code}", cooldown=60)
        raise ModelCallError(f"{model.label}: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        model.mark_failure(f"HTTP {resp.status_code}", cooldown=300)
        raise ModelCallError(f"{model.label}: HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = _message_content(msg)
    provider_fields = msg.get("provider_specific_fields") or {}
    tool_calls = msg.get("tool_calls") or provider_fields.get("tool_calls") or []
    usage = data.get("usage") or {}
    pec = usage.get("prompt_tokens", 0)
    ec = usage.get("completion_tokens", 0)
    return text, pec, ec, tool_calls


def _health_check_model(model, needs_tools):
    messages = [{"role": "user", "content": "只回复 OK"}]
    _api_chat(model, messages)
    if not needs_tools:
        return True
    test_tool = {
        "type": "function",
        "function": {
            "name": "read_novel_lines",
            "description": "Read lines from novel text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "count": {"type": "integer"},
                },
                "required": ["start", "count"],
            },
        },
    }
    tool_messages = [
        {"role": "system", "content": "You are a tool-using assistant. Call tools when asked."},
        {"role": "user", "content": "Call the tool read_novel_lines with start=1 and count=1. Do not answer in text."},
    ]
    _, _, _, tool_calls = _api_chat(model, tool_messages, tools=[test_tool], tool_choice="auto")
    if not tool_calls:
        raise ModelCallError(f"{model.label}: chat works but tool calls are unsupported")
    return True


def init_api_fallback(health_check="all"):
    global API_MODELS, API_ROUND_ROBIN_CURSOR
    API_MODELS = _build_api_models()
    API_ROUND_ROBIN_CURSOR = {}
    if not API_MODELS:
        raise ModelCallError("No API fallback models configured. Set ZHIPUAI_API_KEY or configure opencode providers.")

    if health_check == "none":
        print("  API fallback health check: skipped")
        print("  API fallback order:")
        for model in API_MODELS:
            rr = f" rr={model.round_robin_group}" if model.round_robin_group else ""
            print(f"    PEND {model.label}{rr} (lazy check on first use)")
        return

    print("  API fallback health check:")
    usable = []
    models_to_check = API_MODELS[:1] if health_check == "first" else API_MODELS
    checked = set()
    for model in models_to_check:
        checked.add(model.label)
        needs_tools = model.tool_capable
        try:
            _health_check_model(model, needs_tools=needs_tools)
            if needs_tools:
                usable.append(model)
                rr = f" rr={model.round_robin_group}" if model.round_robin_group else ""
                print(f"    OK   {model.label}{rr} (chat + tools)")
            else:
                rr = f" rr={model.round_robin_group}" if model.round_robin_group else ""
                print(f"    CHAT {model.label}{rr} (chat only, skipped for tool rounds)")
        except Exception as exc:
            model.mark_failure(str(exc), cooldown=300)
            print(f"    FAIL {model.label} ({str(exc)[:100]})")
    if health_check == "first":
        for model in API_MODELS:
            if model.label not in checked:
                rr = f" rr={model.round_robin_group}" if model.round_robin_group else ""
                print(f"    PEND {model.label}{rr} (lazy fallback)")
    if not usable and health_check == "all":
        raise ModelCallError("All tool-capable API fallback models failed health checks.")


def _expects_tool_call(messages, tools):
    if not tools:
        return False
    if any(msg.get("role") == "tool" for msg in messages):
        return False
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content") or ""
            break
    forced_markers = (
        "MUST call read_novel_lines",
        "Call the tool read_novel_lines",
        "Call read_novel_lines",
    )
    return any(marker in last_user for marker in forced_markers)


def _coerce_text_tool_call(text, tools):
    if not text or not tools:
        return []
    tool_names = {tool.get("function", {}).get("name") for tool in tools}
    if "read_novel_lines" in tool_names:
        match = re.search(r"read_novel_lines\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if match:
            args = {"start": int(match.group(1)), "count": int(match.group(2))}
            return [{
                "id": "text_tool_read_novel_lines",
                "type": "function",
                "function": {"name": "read_novel_lines", "arguments": json.dumps(args, ensure_ascii=False)},
            }]
    if "search_novel" in tool_names:
        match = re.search(r"search_novel\s*\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*(\d+))?\s*\)", text)
        if match:
            args = {"keyword": match.group(1)}
            if match.group(2):
                args["context_lines"] = int(match.group(2))
            return [{
                "id": "text_tool_search_novel",
                "type": "function",
                "function": {"name": "search_novel", "arguments": json.dumps(args, ensure_ascii=False)},
            }]
    return []


def _tool_names(tools):
    return {tool.get("function", {}).get("name") for tool in (tools or [])}


def _infer_target_line_from_messages(messages):
    for msg in reversed(messages or []):
        content = msg.get("content") or ""
        match = re.search(r"\bLine\s+(\d+)\s*[:：]", content)
        if match:
            return int(match.group(1))
        match = re.search(r"\bL(\d+)\b", content)
        if match:
            return int(match.group(1))
    return None


def _fallback_required_tool_call(messages, tools):
    """Recover when a tool-capable model writes intent text instead of a required tool call."""
    if "read_novel_lines" not in _tool_names(tools):
        return []
    target_line = _infer_target_line_from_messages(messages)
    if not target_line:
        return []
    start = max(1, target_line - 10)
    args = {"start": start, "count": 25}
    return [{
        "id": "recovered_required_read_novel_lines",
        "type": "function",
        "function": {"name": "read_novel_lines", "arguments": json.dumps(args, ensure_ascii=False)},
    }]


def _should_retry_model_error(exc):
    text = str(exc)
    fatal_markers = (
        "context budget exceeded",
        "HTTP 400",
        "HTTP 401",
        "HTTP 403",
        "invalid api key",
        "unauthorized",
        "forbidden",
    )
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in fatal_markers):
        return False
    if "HTTP 429" in text:
        return False
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    retry_markers = (
        "Read timed out",
        "Max retries exceeded",
        "Connection aborted",
        "RemoteDisconnected",
        "HTTP 500",
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
    )
    return any(marker in text for marker in retry_markers)


def _should_failover_to_next_model(exc, model):
    """Avoid retrying one stalled account when a round-robin pool has peers."""
    if not getattr(model, "round_robin_group", ""):
        return False
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("read timed out", "connection aborted", "max retries exceeded"))


def _cooldown_for_model_error(exc):
    text = str(exc)
    lowered = text.lower()
    if any(marker in lowered for marker in ("http 401", "http 403", "invalid api key", "unauthorized", "forbidden")):
        return 3600
    if any(marker in lowered for marker in ("http 429", "rate limit", "too many requests", "quota", "insufficient")):
        return 300
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return 60
    if any(marker in text for marker in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")):
        return 60
    return 30


def _sleep_before_retry(attempt):
    if API_RETRY_DELAY <= 0:
        return
    delay = min(API_RETRY_DELAY * (2 ** max(0, attempt - 1)), 60.0)
    time.sleep(delay)


def _extract_int_arg(text, name, default=None):
    match = re.search(rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*["\']?(-?\d+)', text)
    if match:
        return int(match.group(1))
    return default


def _extract_str_arg(text, name, default=""):
    quoted = re.search(rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*["\']([^"\']+)["\']', text)
    if quoted:
        return quoted.group(1)
    bare = re.search(rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*([^,}}\]\s]+)', text)
    if bare:
        return bare.group(1).strip()
    return default


def _repair_missing_commas(text):
    value = r'(?:"[^"]*"|\'[^\']*\'|-?\d+(?:\.\d+)?|true|false|null)'
    key_ahead = r'(?=["\']?[A-Za-z_][\w-]*["\']?\s*:)'
    return re.sub(rf'({value})\s+{key_ahead}', r'\1, ', text)


def parse_tool_arguments(function_name, raw_args):
    """Parse tool arguments defensively; model tool JSON is sometimes malformed."""
    if isinstance(raw_args, dict):
        return raw_args, ""
    if raw_args is None:
        return {}, "missing tool arguments"
    if not isinstance(raw_args, str):
        return {}, f"unsupported argument type: {type(raw_args).__name__}"

    text = raw_args.strip()
    if not text:
        return {}, "empty tool arguments"

    attempts = [text, _repair_missing_commas(text)]
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "" if candidate == text else "repaired malformed JSON arguments"
        except json.JSONDecodeError:
            pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed, "parsed Python-style tool arguments"
    except (ValueError, SyntaxError):
        pass

    if function_name == "read_novel_lines":
        start = _extract_int_arg(text, "start")
        count = _extract_int_arg(text, "count")
        if start is None or count is None:
            nums = [int(n) for n in re.findall(r"-?\d+", text)]
            if start is None and nums:
                start = nums[0]
            if count is None and len(nums) >= 2:
                count = nums[1]
        if start is not None:
            args = {"start": max(1, int(start)), "count": max(1, min(int(count or 10), 200))}
            return args, "recovered read_novel_lines arguments from malformed JSON"

    if function_name == "search_novel":
        keyword = _extract_str_arg(text, "keyword")
        context_lines = _extract_int_arg(text, "context_lines", 2)
        if keyword:
            return {"keyword": keyword, "context_lines": max(0, min(int(context_lines or 2), 5))}, \
                "recovered search_novel arguments from malformed JSON"

    return {}, f"failed to parse tool arguments: {text[:120]}"


def normalize_tool_call(tc, func_args):
    normalized = dict(tc)
    func = dict(normalized.get("function", {}) or {})
    func["arguments"] = json.dumps(func_args or {}, ensure_ascii=False)
    normalized["function"] = func
    return normalized


def compact_text(text, max_chars, keep="tail"):
    text = text or ""
    if len(text) <= max_chars:
        return text
    marker = f"\n...(truncated {len(text) - max_chars} chars)...\n"
    keep_chars = max(0, max_chars - len(marker))
    if keep == "head":
        return text[:keep_chars] + marker
    if keep == "middle":
        left = keep_chars // 2
        right = keep_chars - left
        return text[:left] + marker + text[-right:]
    return marker + text[-keep_chars:]


def call_api_fallback(messages, tools=None, label="", request_timeout=None, tool_choice="auto"):
    global API_CALL_TRACE
    needs_tools = bool(tools)
    expects_tool = _expects_tool_call(messages, tools)
    effective_timeout = max(1, int(request_timeout or 300))
    errors = []
    max_attempts = max(1, int(API_RETRIES or 1))
    for model in _api_model_iteration_order():
        if not model.available(needs_tools):
            continue
        for attempt in range(1, max_attempts + 1):
            try:
                effective_tool_choice = tool_choice
                if expects_tool and model.name != "agnes":
                    effective_tool_choice = {"type": "function", "function": {"name": "read_novel_lines"}}
                temp_log_event(
                    "model_call_start",
                    label=label,
                    provider=model.name,
                    model=model.model,
                    model_label=model.label,
                    round_robin_group=model.round_robin_group,
                    tools_enabled=bool(tools),
                    expects_tool=expects_tool,
                    tool_choice=effective_tool_choice,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    api_context_limit=API_CONTEXT_LIMIT,
                    request_timeout=effective_timeout,
                    **_message_trace_summary(messages),
                )
                text, pec, ec, tool_calls = _api_chat(
                    model,
                    messages,
                    tools=tools,
                    tool_choice=effective_tool_choice,
                    request_timeout=effective_timeout,
                )
                coerced_tool_call = False
                recovered_required_tool_call = False
                if not tool_calls:
                    tool_calls = _coerce_text_tool_call(text, tools)
                    coerced_tool_call = bool(tool_calls)
                if expects_tool and not tool_calls:
                    tool_calls = _fallback_required_tool_call(messages, tools)
                    recovered_required_tool_call = bool(tool_calls)
                    if recovered_required_tool_call:
                        temp_log_event(
                            "model_call_recovered_tool",
                            label=label,
                            provider=model.name,
                            model=model.model,
                            model_label=model.label,
                            round_robin_group=model.round_robin_group,
                            attempt=attempt,
                            reason="synthesized required read_novel_lines call from target line",
                            tool_calls=len(tool_calls),
                            response_len=len(text or ""),
                            response_head=(text or "")[:500],
                        )
                if expects_tool and not tool_calls:
                    temp_log_event(
                        "model_call_rejected",
                        label=label,
                        provider=model.name,
                        model=model.model,
                        model_label=model.label,
                        round_robin_group=model.round_robin_group,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        reason="expected tool call but model returned none",
                        response_len=len(text or ""),
                        response_head=(text or "")[:500],
                    )
                    err = f"{model.label}: no tool call"
                    if attempt < max_attempts:
                        errors.append(f"{err} (attempt {attempt}/{max_attempts})")
                        temp_log_event(
                            "model_call_retry",
                            label=label,
                            provider=model.name,
                            model=model.model,
                            model_label=model.label,
                            round_robin_group=model.round_robin_group,
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            reason="no tool call",
                        )
                        _sleep_before_retry(attempt)
                        continue
                    errors.append(err)
                    break
                temp_log_event(
                    "model_call_success",
                    label=label,
                    provider=model.name,
                    model=model.model,
                    model_label=model.label,
                    round_robin_group=model.round_robin_group,
                    tools_enabled=bool(tools),
                    tool_calls=len(tool_calls),
                    coerced_text_tool_call=coerced_tool_call,
                    recovered_required_tool_call=recovered_required_tool_call,
                    attempt=attempt,
                    prompt_eval_count=pec,
                    eval_count=ec,
                    response_len=len(text or ""),
                    response_head=(text or "")[:500],
                )
                API_CALL_TRACE.append({
                    "label": label,
                    "provider": model.name,
                    "model": model.model,
                    "model_label": model.label,
                    "round_robin_group": model.round_robin_group,
                    "tools_enabled": bool(tools),
                    "tool_choice": effective_tool_choice,
                    "tool_calls": len(tool_calls),
                    "coerced_text_tool_call": coerced_tool_call,
                    "recovered_required_tool_call": recovered_required_tool_call,
                    "attempt": attempt,
                    "prompt_eval_count": pec,
                    "eval_count": ec,
                })
                return text, pec, ec, tool_calls
            except Exception as exc:
                temp_log_event(
                    "model_call_error",
                    label=label,
                    provider=model.name,
                    model=model.model,
                    model_label=model.label,
                    round_robin_group=model.round_robin_group,
                    tools_enabled=bool(tools),
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=type(exc).__name__,
                    error=str(exc)[:1000],
                    **_message_trace_summary(messages),
                )
                err = str(exc)[:160]
                if _should_failover_to_next_model(exc, model):
                    errors.append(err)
                    model.mark_failure(err, cooldown=_cooldown_for_model_error(exc))
                    temp_log_event(
                        "model_call_failover",
                        label=label,
                        provider=model.name,
                        model=model.model,
                        model_label=model.label,
                        round_robin_group=model.round_robin_group,
                        attempt=attempt,
                        reason="transient round-robin member failure; trying next model",
                    )
                    break
                if attempt < max_attempts and _should_retry_model_error(exc):
                    errors.append(f"{err} (attempt {attempt}/{max_attempts})")
                    temp_log_event(
                        "model_call_retry",
                        label=label,
                        provider=model.name,
                        model=model.model,
                        model_label=model.label,
                        round_robin_group=model.round_robin_group,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        reason=err,
                    )
                    _sleep_before_retry(attempt)
                    continue
                errors.append(err)
                model.mark_failure(err, cooldown=_cooldown_for_model_error(exc))
                break
    detail = "; ".join(errors[-5:]) if errors else "no model available outside cooldown"
    temp_log_event(
        "model_call_failed_all",
        label=label,
        tools_enabled=bool(tools),
        errors=errors[-10:],
        detail=detail,
    )
    raise ModelCallError(f"All API fallback models failed for {label or 'request'}: {detail}")

# Ambiguous descriptive labels that may later resolve to a named character.
# Keep this generic: do not add novel-specific character names here.
TEMP_DESCRIPTORS = {
    "女孩", "少年", "老人", "大汉", "年轻人", "男孩", "少女", "妇人", "老妇",
    "男子", "女子", "青年", "孩子", "女人", "男人", "姑娘", "小伙子",
    "老者", "老妇人", "老翁", "对方", "陌生人", "路人", "来客", "访客",
    "男商人", "女商人", "年轻商人", "老商人", "中年商人", "旅行商人", "行商人", "商人",
    "兑换商", "皮草商", "店主", "老板", "酒吧老板", "店员", "服务生",
    "女服务生", "男服务生", "红发女孩", "红发少女", "红发女子", "客人", "男客人", "女客人",
    "房客", "乞丐", "修女", "祭司", "旅人", "工匠", "磨粉匠", "摊贩",
    "摊贩老板", "车夫", "船夫", "守卫", "卫兵", "士兵", "神父",
    "村民", "村人", "镇民", "居民", "市民", "群众", "众人", "人群", "一行人",
    "来人", "那人", "这个人", "那个人", "同伴", "旅伴", "手下", "追兵", "部下",
    "伙计", "职员", "员工", "成员", "商行员工", "商行成员", "商行同伴", "商行手下",
    "中年男子", "中年女人", "中年女子", "年轻男子", "年轻女子", "年轻女孩",
    "金发女孩", "黑发男子", "店老板", "店家", "老板娘", "女主人", "男主人", "主人",
    "佣人", "仆人", "书记员", "官员", "军人", "骑士", "贵族", "领主", "代理人",
    "船员", "水手", "牧羊人", "牧羊女", "教师", "学生", "医师", "医生",
}

# Forward search patterns for name reveal (must be actual name-introduction, not generic "I am X")
# Priority order: longer patterns first to avoid partial matches
NAME_REVEAL_PATTERNS = ["咱的名字是", "我的名字是", "吾乃", "名字是"]
NAME_TERMINATORS = set("「」『』\"'。，！？、，：:；;）)] \t\r\n")


def search_forward_for_name(line_num, descriptor):
    """
    Search forward in novel text for a real name for a temporary descriptor.
    Searches ALL remaining lines (no hard limit), returns first valid match.
    Returns (real_name, reveal_line) or None if not found.
    Only matches when the pattern is clearly introducing a character's name.
    """
    import re
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    end = len(lines)
    for i in range(line_num, end):
        line = lines[i].strip()
        if len(line) < 3:
            continue
        for pattern in NAME_REVEAL_PATTERNS:
            if pattern in line:
                idx = line.index(pattern) + len(pattern)
                # Extract name: take chars until we hit punctuation or non-name chars
                raw_name = ""
                for ch in line[idx:]:
                    if ch in NAME_TERMINATORS:
                        break
                    raw_name += ch
                # Validate: Chinese name is 2-4 chars
                name_match = re.match(r'^([\u4e00-\u9fff]{2,4})', raw_name)
                if name_match:
                    name = name_match.group(1)
                    cleaned = validate_char_name(name)
                    if cleaned and cleaned != descriptor:
                        return cleaned, i + 1
                # If validation failed but pattern matched, continue searching
                continue
    return None, None


# ============================================================
# Logging
# ============================================================

def _preview_text(text, limit=500):
    text = text or ""
    if len(text) <= limit * 2:
        return {"len": len(text), "head": text, "tail": ""}
    return {"len": len(text), "head": text[:limit], "tail": text[-limit:]}


def _message_trace_summary(messages):
    message_items = []
    for idx, msg in enumerate(messages or []):
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        tool_calls_len = sum(len(json.dumps(tc, ensure_ascii=False, default=str)) for tc in tool_calls)
        preview = _preview_text(content, 240)
        message_items.append({
            "idx": idx,
            "role": msg.get("role", ""),
            "content_len": len(content),
            "tool_calls_len": tool_calls_len,
            "total_len": len(content) + tool_calls_len,
            "content_head": preview["head"],
            "content_tail": preview["tail"],
        })
    estimated = _estimate_message_tokens(messages or [])
    return {
        "message_count": len(messages or []),
        "estimated_context": estimated,
        "estimated_with_output": estimated + API_MAX_OUTPUT_TOKENS,
        "messages": message_items,
    }


def _trace_context(round_log=None):
    source = round_log or CURRENT_ROUND_TRACE or {}
    if not source:
        return {}
    return {
        "trace_id": source.get("trace_id"),
        "round": source.get("round"),
        "dialogue_line": source.get("dialogue_line"),
        "dialogue_text": source.get("dialogue_text"),
    }


def temp_log_event(event, round_log=None, **fields):
    """Append best-effort per-round trace data before the formal log is committed."""
    if not TEMP_LOG_PATH:
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
    }
    entry.update({k: v for k, v in _trace_context(round_log).items() if v is not None})
    entry.update(fields)
    try:
        with open(TEMP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
    except Exception:
        # Temp tracing must never break annotation.
        pass


def set_current_round_trace(round_log):
    global CURRENT_ROUND_TRACE
    CURRENT_ROUND_TRACE = round_log


def clear_current_round_trace():
    global CURRENT_ROUND_TRACE
    CURRENT_ROUND_TRACE = None


def trace_tool_result(agent_name, tool_round, function_name, args, result, note=""):
    preview = _preview_text(result, 500)
    temp_log_event(
        "tool_result",
        agent=agent_name,
        tool_round=tool_round,
        function=function_name,
        args=args,
        result_len=preview["len"],
        result_head=preview["head"],
        result_tail=preview["tail"],
        note=note,
    )


def log_entry(entry):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def new_round(round_idx, line_num, dialogue):
    timestamp = datetime.now().isoformat()
    return {
        "trace_id": f"{os.getpid()}-{timestamp}-L{line_num}",
        "round": round_idx,
        "dialogue_line": line_num,
        "dialogue_text": dialogue,
        "timestamp": timestamp,
        "agents": {},
        "tool_calls": [],
        "result": None,
    }


def log_agent(round_log, agent_name, role, input_messages, response_text, pec, ec,
               tool_calls_list=None, total_pec=None, total_ec=None):
    global API_CALL_TRACE
    entry = {
        "agent": agent_name,
        "role": role,
        "input_summary": input_messages[-1]["content"][:200] if input_messages else "",
        "response": response_text,
        "prompt_eval_count": total_pec if total_pec else pec,
        "eval_count": total_ec if total_ec else ec,
        "total_tokens": (total_pec if total_pec else pec) + (total_ec if total_ec else ec),
    }
    if tool_calls_list:
        entry["tool_calls_detail"] = tool_calls_list
    if API_CALL_TRACE:
        entry["model_calls"] = API_CALL_TRACE
        API_CALL_TRACE = []
    round_log["agents"][agent_name] = entry
    temp_log_event("agent_complete", round_log, agent=agent_name, role=role, entry=entry)


# ============================================================
# Utility functions
# ============================================================

SPEECH_ATTRIBUTION_RE = re.compile(
    r"(说(?:道|着|完|了)?|问(?:道)?|回答|答道|喊(?:道)?|叫(?:道)?|开口|"
    r"低语|嘀咕|喃喃|叹(?:道|息)?|笑(?:道)?|补充|继续说|表示)"
)


def read_novel_lines(start, count):
    """Read lines from novel.txt. start is 1-based."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = []
    for i in range(start - 1, min(start - 1 + count, len(lines))):
        result.append(f"{i+1}: {lines[i].rstrip()}")
    return "\n".join(result)


def search_novel(keyword, context_lines=1, max_matches=8, max_chars=12000):
    """Search novel.txt for a keyword. Returns bounded matching lines with context."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    results = []
    total_matches = 0
    context_lines = max(0, min(int(context_lines or 1), 5))
    for i, line in enumerate(lines):
        if keyword in line:
            total_matches += 1
            if len(results) >= max_matches:
                continue
            ctx_start = max(0, i - context_lines)
            ctx_end = min(len(lines), i + context_lines + 1)
            block = []
            for j in range(ctx_start, ctx_end):
                marker = ">>>" if j == i else "   "
                block.append(f"{marker}{j+1}: {lines[j].rstrip()}")
            results.append("\n".join(block))
    if not results:
        return f"(No matches for '{keyword}')"
    output = "\n---\n".join(results)
    if len(output) > max_chars:
        output = output[:max_chars] + "\n...(search result truncated; use a narrower keyword or read_novel_lines around a specific line)"
    if total_matches > len(results):
        output += f"\n...(showing first {len(results)} of {total_matches} matches; use a narrower keyword or read_novel_lines around a specific line)"
    return output


def get_narrative_before(line_num, max_lines=5):
    """Extract narrative (non-dialogue) lines before a given line."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    narrative = []
    for i in range(line_num - 2, max(-1, line_num - 2 - max_lines), -1):
        if i < 0:
            break
        text = lines[i].rstrip()
        if "「" in text:
            break
        if text.strip():
            narrative.insert(0, text.strip())
    return " | ".join(narrative) if narrative else ""


def deep_search_identity(temp_name, around_line, search_forward=200, search_backward=100):
    """Search for identity clues for a temporary descriptor near a line."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = max(0, around_line - 1 - search_backward)
    end = min(len(lines), around_line - 1 + search_forward)
    intro_patterns = ["我叫", "咱是", "我是", "名字是", "吾乃", "咱的名字是"]

    results = [f"=== Deep identity search for '{temp_name}' near L{around_line} ==="]
    results.append(f"Range: L{start+1} to L{end} ({end-start} lines)\n")

    intro_matches = []
    for i in range(start, end):
        line_text = lines[i]
        for pat in intro_patterns:
            if pat in line_text:
                intro_matches.append((i + 1, line_text.strip()))

    if intro_matches:
        results.append(f"--- Self-introduction patterns ({len(intro_matches)} matches) ---")
        for ln, text in intro_matches:
            rel = ">" if abs(ln - around_line) < 50 else " "
            results.append(f"  {rel}L{ln}: {text[:100]}")

    temp_matches = []
    for i in range(start, end):
        if temp_name in lines[i]:
            temp_matches.append((i + 1, lines[i].strip()))

    if temp_matches:
        results.append(f"\n--- '{temp_name}' occurrences ({len(temp_matches)} matches) ---")
        for ln, text in temp_matches[:15]:
            results.append(f"  L{ln}: {text[:100]}")
        if len(temp_matches) > 15:
            results.append(f"  ... and {len(temp_matches) - 15} more")

    if not intro_matches and not temp_matches:
        results.append("(No relevant matches found)")

    return "\n".join(results)


def find_all_references(name, max_results=10):
    """Find all occurrences of a name in the novel with context."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    results = []
    count = 0
    for i, line in enumerate(lines):
        if name in line:
            ctx_start = max(0, i - 1)
            ctx_end = min(len(lines), i + 2)
            block = []
            for j in range(ctx_start, ctx_end):
                marker = ">>>" if j == i else "   "
                block.append(f"{marker}{j+1}: {lines[j].rstrip()}")
            results.append("\n".join(block))
            count += 1
            if count >= max_results:
                break
    return "\n---\n".join(results) if results else f"(No references to '{name}')"


def build_context_index(line_num, context_window=40):
    """Build an abstract map of novel around target line (no full text).
    Forces the Labeler to use read_novel_lines to see actual content."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = max(1, line_num - context_window)
    end = min(len(lines), line_num + context_window)
    result = []
    result.append(f"[Novel Map near L{line_num} (read full text with read_novel_lines)]")
    result.append(f"  -> = target  [D] = has dialogue  [N] = narrative")

    last_type = None
    block_start = None
    block_lines = []

    for i in range(start - 1, end):
        ln = i + 1
        text = lines[i].rstrip()
        has_dialogue = "\u300c" in text
        line_type = "D" if has_dialogue else "N"

        if line_type == "D":
            if last_type == "N" and block_start is not None:
                result.append(f"     L{block_start}-L{ln-1} [N] ({len(block_lines)} lines)")
            marker = "->" if ln == line_num else "  "
            dlg = re.search(r"\u300c([^\u300d]+)\u300d", text)
            dlg_text = dlg.group(1)[:40] + "..." if dlg and len(dlg.group(1)) > 40 else (dlg.group(1) if dlg else "")
            result.append(f"  {marker}L{ln} [D] \u300c{dlg_text}\u300d")
            last_type = "D"
            block_start = None
            block_lines = []
        else:
            if last_type != "N":
                if block_start is not None:
                    result.append(f"     L{block_start}-L{ln-1} [N] ({len(block_lines)} lines)")
                block_start = ln
                block_lines = []
            block_lines.append(text)
            last_type = "N"

    if block_start is not None and block_lines:
        result.append(f"     L{block_start}-L{min(len(lines), end)} [N] ({len(block_lines)} lines)")

    return "\n".join(result)


def get_dialogue_list():
    """Extract all dialogues from novel as (line_num, text) pairs."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    dialogues = []
    for line_num, line in enumerate(content.split("\n"), start=1):
        matches = re.findall(r"「([^」]+)」", line)
        for dialogue in matches:
            dialogues.append((line_num, dialogue))
    return dialogues


def get_labeled_count():
    if not os.path.exists(LABELED_PATH):
        return 0
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        return len(f.readlines())


def write_label(name):
    with open(LABELED_PATH, "a", encoding="utf-8") as f:
        f.write(name + "\n")


def call_ollama(messages, tools=None, label="", request_timeout=None, tool_choice="auto"):
    if MODEL_PROVIDER == "api-fallback":
        return call_api_fallback(
            messages,
            tools=tools,
            label=label,
            request_timeout=request_timeout,
            tool_choice=tool_choice,
        )

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    if tools:
        payload["tools"] = tools
    temp_log_event(
        "model_call_start",
        label=label,
        provider="ollama",
        model=OLLAMA_MODEL,
        model_label=OLLAMA_MODEL,
        tools_enabled=bool(tools),
        **_message_trace_summary(messages),
    )
    try:
        resp = requests.post(url, json=payload, timeout=max(1, int(request_timeout or 300)))
        data = resp.json()
        pec = data.get("prompt_eval_count", 0)
        ec = data.get("eval_count", 0)
        text = data.get("message", {}).get("content", "")
        tool_calls = data.get("message", {}).get("tool_calls", [])
        temp_log_event(
            "model_call_success",
            label=label,
            provider="ollama",
            model=OLLAMA_MODEL,
            model_label=OLLAMA_MODEL,
            tools_enabled=bool(tools),
            tool_calls=len(tool_calls),
            prompt_eval_count=pec,
            eval_count=ec,
            response_len=len(text or ""),
            response_head=(text or "")[:500],
        )
        return text, pec, ec, tool_calls
    except Exception as exc:
        temp_log_event(
            "model_call_error",
            label=label,
            provider="ollama",
            model=OLLAMA_MODEL,
            model_label=OLLAMA_MODEL,
            tools_enabled=bool(tools),
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
            **_message_trace_summary(messages),
        )
        raise


# ============================================================
# Tool definition
# ============================================================

TOOL_READ_NOVEL = {
    "type": "function",
    "function": {
        "name": "read_novel_lines",
        "description": "Read lines from novel.txt. start is 1-based line number. count is how many lines to read. Returns lines in 'line_number: content' format.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "Starting line number (1-based)"},
                "count": {"type": "integer", "description": "Number of lines to read"}
            },
            "required": ["start", "count"]
        }
    }
}

TOOL_SEARCH_NOVEL = {
    "type": "function",
    "function": {
        "name": "search_novel",
        "description": "Search the novel text for a keyword or character name. Returns matching lines with surrounding context. Use this to find where a character was introduced, where their name appears, or to confirm identity clues.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "The keyword or character name to search for"},
                "context_lines": {"type": "integer", "description": "Number of context lines before and after each match (default: 2)"}
            },
            "required": ["keyword"]
        }
    }
}

LABELER_TOOLS = [TOOL_READ_NOVEL, TOOL_SEARCH_NOVEL]

QUALITY_OUTPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit one structured, evidence-cited quality review decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string"},
                "target_speaker": {"type": "string"},
                "preferred_speaker": {"type": "string"},
                "block_assignments": {"type": "string"},
                "quote_type": {
                    "type": "string",
                    "enum": [
                        "direct_speech", "group_speech", "embedded_quote",
                        "thought_or_narration", "sound_or_text", "unclear",
                    ],
                },
                "voice_kind": {
                    "type": "string",
                    "enum": ["person", "group", "non_person", "unclear"],
                },
                "evidence_basis": {
                    "type": "string",
                    "enum": [
                        "explicit_attribution", "speaker_action", "quote_type",
                        "alternation", "inference", "unknown",
                    ],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "reason": {"type": "string"},
            },
        },
    },
}


NON_PERSON_LABEL = "非人物发声"
NON_PERSON_ALIASES = {
    "narr",
    "narrator",
    "non-human",
    "non-person",
    "non_person",
    "nonperson",
    "not-spoken",
    "not_spoken",
    "sound-effect",
    "sound_effect",
    "ambient-sound",
    "ambient_sound",
    "narrative",
    "narration",
    "narrative_voice",
    "旁白",
    "叙述",
    "叙述者",
    "音效",
    "声音",
    "拟声",
    "非人物",
    "非人物发声",
}


# ============================================================
# Character State - clean, validated, no pollution
# ============================================================

def validate_char_name(name):
    """
    Validate and clean a character name.
    Returns cleaned name or None if invalid.
    Rejects: names with description text appended.
    """
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if not name:
        return None
    alias_key = name.strip().lower()
    if alias_key in NON_PERSON_ALIASES:
        return NON_PERSON_LABEL
    # Reject if contains description patterns (model sometimes appends these)
    bad_patterns = [
        "--", "—", "–", "\u2014", "\u2013",
        "根据", "原文", "禁止", "错误示例",
        "Looking at", "context around", "speaker is", "the speaker",
    ]
    for bp in bad_patterns:
        if bp in name:
            return None
    # Reject if too long (>15 chars means it's probably a description)
    if len(name) > 15:
        return None
    # Reject if contains whitespace
    if re.search(r"\s", name):
        return None
    # Reject if contains Chinese parentheses or brackets with content
    if re.search(r"[（(【\[].*[）)】\]]", name):
        return None
    return name


INVALID_SPEAKER_OUTPUTS = {
    "?", "unclear", "unknown", "null", "none", "speaker", "person", "group",
}


def is_valid_final_speaker(name):
    """Reject protocol values and untranslated labels before they reach state or output."""
    cleaned = validate_char_name(name)
    if not cleaned:
        return False
    if cleaned == NON_PERSON_LABEL:
        return True
    if cleaned.strip().lower() in INVALID_SPEAKER_OUTPUTS:
        return False
    if re.search(r"[A-Za-z]", cleaned):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", cleaned))


GENERIC_ROLE_SUFFIXES = (
    "的人", "手下", "成员", "员工", "同伴", "部下", "追兵", "群众", "众人", "人群",
    "男子", "女子", "女孩", "少年", "少女", "老人", "客人", "村民", "镇民", "居民",
    "商人", "店主", "老板", "伙计", "职员", "佣人", "仆人", "士兵", "守卫", "卫兵",
)
SPEAKER_PRONOUNS = {
    "我", "你", "他", "她", "它", "咱", "咱们", "我们", "你们", "他们", "她们",
    "此人", "那人", "这人", "那个人", "这个人", "对方", "自己", "本人",
}
ATTRIBUTION_CLEAN_SPLITS = re.compile(r"[，,。！？；;：:\s]|(?:向|对|朝|冲|跟|和|给|把|被)")
ATTRIBUTION_VERB_RE = re.compile(
    r"(?P<speaker>[\u4e00-\u9fffA-Za-z0-9·•・]{1,15}(?:向|对|朝|冲|跟|和|给)?[\u4e00-\u9fffA-Za-z0-9·•・]{0,8})"
    r"(?P<verb>说(?:道|着|完|了)?|问(?:道)?|回答|答道|喊(?:道)?|叫(?:道)?|开口|"
    r"低语|嘀咕|喃喃|叹(?:道|息)?|笑(?:道)?|补充|继续说|表示|怒吼|大喊|小声说)"
)
QUOTE_TYPE_CAUTION_KEYWORDS = (
    "写着", "刻着", "上面写", "牌子", "文字", "读作", "念作", "书上", "信上", "纸上",
    "传来", "响起", "声音", "音效", "敲门声", "脚步声", "钟声",
    "心想", "心里想", "心中", "脑中", "想着", "自言自语般", "仿佛", "似乎在说", "像是在说",
)


def is_generic_speaker_label(name):
    """Return True for role/appearance/group labels that should not become canonical characters."""
    cleaned = validate_char_name(name) or (name or "").strip()
    if not cleaned or cleaned in {"?", NON_PERSON_LABEL}:
        return False
    if cleaned in TEMP_DESCRIPTORS:
        return True
    # A role followed by an apparent personal name should stay name-like.
    for role in sorted(TEMP_DESCRIPTORS, key=len, reverse=True):
        if cleaned.startswith(role) and len(cleaned) > len(role):
            return False
    if any(cleaned.endswith(suffix) for suffix in GENERIC_ROLE_SUFFIXES):
        return True
    return False


def _clean_attribution_candidate(raw):
    raw = (raw or "").strip(" 　，,。！？；;：:「」『』（）()[]【】")
    if not raw:
        return ""
    parts = [p for p in ATTRIBUTION_CLEAN_SPLITS.split(raw) if p]
    candidate = parts[0] if parts else raw
    candidate = re.sub(r"^(于是|然后|接着|这时|那时|只见|而|但|可是|不过|同时)", "", candidate)
    candidate = re.sub(r"(则|又|也|便|才|却|就|仍|仍然|继续)$", "", candidate)
    candidate = candidate.strip(" 　，,。！？；;：:「」『』（）()[]【】")
    if candidate in SPEAKER_PRONOUNS:
        return ""
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", candidate):
        return ""
    return candidate[:15]


def _extract_attribution_candidates(text):
    candidates = []
    for match in ATTRIBUTION_VERB_RE.finditer(text or ""):
        candidate = _clean_attribution_candidate(match.group("speaker"))
        if candidate:
            candidates.append({"speaker": candidate, "verb": match.group("verb")})
    # Handle quote-first forms: 「...」某人说道。
    for tail in re.findall(r"」([^。！？；;]{0,30})", text or ""):
        for match in ATTRIBUTION_VERB_RE.finditer(tail):
            candidate = _clean_attribution_candidate(match.group("speaker"))
            if candidate:
                candidates.append({"speaker": candidate, "verb": match.group("verb")})
    deduped = []
    seen = set()
    for item in candidates:
        key = (item["speaker"], item["verb"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def analyze_local_evidence(line_num, dialogue, local_structure=None):
    """Build deterministic local evidence hints without using answer labels."""
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_lines = [line.rstrip("\n") for line in f.readlines()]
    if not (1 <= line_num <= len(novel_lines)):
        return {"line_num": line_num, "target_line": "", "attribution_candidates": [], "quote_type_cautions": []}

    start = max(1, line_num - 3)
    end = min(len(novel_lines), line_num + 2)
    local_items = []
    candidates = []
    for ln in range(start, end + 1):
        text = novel_lines[ln - 1].strip()
        local_items.append((ln, text))
        for item in _extract_attribution_candidates(text):
            item = dict(item)
            item["line"] = ln
            item["text"] = text[:120]
            candidates.append(item)

    target_line = novel_lines[line_num - 1].strip()
    cautions = []
    if target_line.count("「") > 1:
        cautions.append("multiple quoted spans on the target line; verify which quoted text is the target")
    if any(keyword in target_line for keyword in QUOTE_TYPE_CAUTION_KEYWORDS):
        cautions.append("target line contains quote-type caution words; verify actual speech vs narration/sound/text")
    if len((dialogue or "").strip()) <= 4 and re.search(r"[咚叮砰啪哗轰咕咔铃吱呀啊嗯唔呜嘿喂]", dialogue or ""):
        cautions.append("very short sound-like fragment; verify whether it is human speech")

    return {
        "line_num": line_num,
        "target_line": target_line,
        "local_start": start,
        "local_end": end,
        "local_lines": local_items,
        "attribution_candidates": candidates[:8],
        "quote_type_cautions": cautions,
        "narrative_between": (local_structure or {}).get("narrative_between", []),
        "no_narrative_break_from_previous": bool((local_structure or {}).get("no_narrative_break_from_previous")),
    }


def format_local_evidence_hint(local_evidence):
    lines = ["[Deterministic local evidence audit - no answer labels]"]
    target_line = local_evidence.get("target_line") or ""
    if target_line:
        lines.append(f"  Target raw line: L{local_evidence.get('line_num')}: {target_line[:180]}")
    candidates = local_evidence.get("attribution_candidates") or []
    if candidates:
        lines.append("  Local attribution candidates detected by regex; verify in raw text before using:")
        for item in candidates[:5]:
            lines.append(f"    L{item.get('line')}: {item.get('speaker')} + {item.get('verb')} | {item.get('text')[:90]}")
    else:
        lines.append("  Local attribution candidates detected by regex: none")
    cautions = local_evidence.get("quote_type_cautions") or []
    if cautions:
        lines.append("  Quote-type cautions:")
        for caution in cautions:
            lines.append(f"    - {caution}")
    lines.append("  Treat alternation as weak evidence. Do not override raw attribution or quote-type evidence with memory order.")
    return "\n".join(lines)


class CharacterState:
    """Clean character state with strict validation."""

    def __init__(self):
        self.state = self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Validate loaded data - clean if corrupted
                if self._is_valid_state(data):
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return {"characters": {}, "alias_map": {}, "speech_order": []}

    def _is_valid_state(self, data):
        if not isinstance(data, dict):
            return False
        if "characters" not in data or "alias_map" not in data:
            return False
        # Check for corrupted entries
        for name in data.get("characters", {}):
            if not validate_char_name(name):
                return False
        return True

    def update_speech_order(self, speaker_name):
        """Track the order of who spoke, for alternating pattern detection."""
        cleaned = validate_char_name(speaker_name)
        if not cleaned:
            return
        real_name = self.resolve_alias(cleaned)
        order = self.state.setdefault("speech_order", [])
        # Keep last 10 entries
        order.append(real_name)
        if len(order) > 10:
            order.pop(0)

    def get_recent_speakers_text(self):
        """Get a hint about who spoke recently - useful for alternating pattern."""
        order = self.state.get("speech_order", [])
        if not order:
            return ""
        unique = []
        for s in order:
            if s not in unique:
                unique.append(s)
            if len(unique) >= 3:
                break
        last = order[-1] if order else ""
        if last:
            text = f"  [Last speaker: {last}]"
            if len(unique) >= 2:
                text += f"\n  [Exchange: {unique[0]} ↔ {unique[1]}]"
            # Check if the last 4 show clear alternation
            if len(order) >= 4:
                recent = order[-4:]
                if recent[0] != recent[1] and recent[2] != recent[3] and recent[0] == recent[2] and recent[1] == recent[3]:
                    text += "\n  [Alternating pattern confirmed]"
            return text
        return ""

    def save(self, round_num=0):
        if round_num > 0 and os.path.exists(STATE_PATH):
            history_dir = os.path.join(ROOT_DIR, "data", "state_history")
            os.makedirs(history_dir, exist_ok=True)
            backup_path = os.path.join(history_dir, f"character_state_round_{round_num:04d}.json")
            with open(STATE_PATH, "r", encoding="utf-8") as src:
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def resolve_alias(self, name):
        """Resolve an alias to the canonical character name."""
        alias_map = self.state.get("alias_map", {})
        if name in alias_map:
            return alias_map[name]
        for char_name, char_info in self.state.get("characters", {}).items():
            if name in char_info.get("aliases", []):
                return char_name
        return name

    def add_character(self, name, aliases=None):
        """Add a new character with validated name."""
        cleaned = validate_char_name(name)
        if not cleaned:
            return
        if cleaned not in self.state["characters"]:
            self.state["characters"][cleaned] = {
                "aliases": [cleaned],
                "first_seen_line": None,
                "last_seen_line": None,
                "recent_lines": [],
            }
        for alias in (aliases or []):
            cleaned_alias = validate_char_name(alias)
            if cleaned_alias and cleaned_alias not in self.state["characters"][cleaned]["aliases"]:
                self.state["characters"][cleaned]["aliases"].append(cleaned_alias)
                self.state["alias_map"][cleaned_alias] = cleaned

    def update_character(self, name, line_num, dialogue_text=""):
        """Update character's last seen line and recent dialogue."""
        cleaned = validate_char_name(name)
        if not cleaned:
            return
        real_name = self.resolve_alias(cleaned)
        if real_name not in self.state["characters"]:
            self.add_character(real_name)
        char = self.state["characters"][real_name]
        if char["first_seen_line"] is None:
            char["first_seen_line"] = line_num
        char["last_seen_line"] = line_num
        if dialogue_text:
            char["recent_lines"].append((line_num, dialogue_text))
            if len(char["recent_lines"]) > 5:
                char["recent_lines"].pop(0)

    def add_alias(self, real_name, alias, is_identity_reveal=False):
        """Add an alias mapping."""
        cleaned_real = validate_char_name(real_name)
        cleaned_alias = validate_char_name(alias)
        if not cleaned_real or not cleaned_alias:
            return
        if is_identity_reveal:
            # If the alias already exists as its own character, merge it into real_name
            if cleaned_alias in self.state["characters"] and cleaned_alias != cleaned_real:
                alias_char = self.state["characters"].pop(cleaned_alias)
                # Transfer recent_lines and first/last_seen
                if cleaned_real not in self.state["characters"]:
                    self.add_character(cleaned_real)
                target = self.state["characters"][cleaned_real]
                target["recent_lines"].extend(alias_char.get("recent_lines", []))
                if len(target["recent_lines"]) > 5:
                    target["recent_lines"] = target["recent_lines"][-5:]
                if alias_char.get("first_seen_line") and (target["first_seen_line"] is None or alias_char["first_seen_line"] < target["first_seen_line"]):
                    target["first_seen_line"] = alias_char["first_seen_line"]
                if alias_char.get("last_seen_line") and (target["last_seen_line"] is None or alias_char["last_seen_line"] > target["last_seen_line"]):
                    target["last_seen_line"] = alias_char["last_seen_line"]
                # Remove old alias_map entries pointing to the aliased character
                for old_alias, mapped in list(self.state.get("alias_map", {}).items()):
                    if mapped == cleaned_alias:
                        self.state["alias_map"][old_alias] = cleaned_real
            if cleaned_real not in self.state["characters"]:
                self.add_character(cleaned_real)
            if cleaned_alias not in self.state["characters"][cleaned_real]["aliases"]:
                self.state["characters"][cleaned_real]["aliases"].append(cleaned_alias)
            self.state["alias_map"][cleaned_alias] = cleaned_real

    def get_active_characters(self, last_n_lines=50, current_line=0):
        """Get characters active within last N lines."""
        active = []
        for name, info in self.state.get("characters", {}).items():
            ls = info.get("last_seen_line")
            if ls and (current_line - ls) <= last_n_lines:
                active.append(name)
        return active

    def get_state_text(self, current_line=0):
        """Generate character state text for Labeler."""
        lines = []
        active = self.get_active_characters(last_n_lines=50, current_line=current_line)
        lines.append("[Character State]")
        for name in active:
            info = self.state["characters"].get(name, {})
            aliases = info.get("aliases", [name])
            recent = info.get("recent_lines", [])
            alias_str = ", ".join(a for a in aliases if a != name)
            if alias_str:
                lines.append(f"  {name} (aka: {alias_str})")
            else:
                lines.append(f"  {name}")
            last_ls = info.get('last_seen_line', '?')
            lines.append(f"    last seen: ~L{last_ls}")
            if recent:
                recent_str = "; ".join([f"L{ln}「{t[:20]}」" for ln, t in recent[-3:]])
                lines.append(f"    recent: {recent_str}")
        if len(self.state.get("characters", {})) > len(active):
            inactive = len(self.state["characters"]) - len(active)
            lines.append(f"  ({inactive} inactive characters)")
        return "\n".join(lines)

    def get_scene_summary(self, current_line=0):
        """Generate brief scene summary."""
        active = self.get_active_characters(last_n_lines=30, current_line=current_line)
        if not active:
            return "[Scene] Initial scene"
        chars_str = ", ".join(active)
        lines_list = [self.state["characters"][c]["last_seen_line"] for c in active
                      if self.state["characters"][c].get("last_seen_line")]
        if lines_list:
            return f"[Scene] L{min(lines_list)}-L{max(lines_list)} | Active: {chars_str}"
        return f"[Scene] Active: {chars_str}"

    def parse_state_update(self, text, line_num, dialogue=""):
        """Parse <state_update> block from Labeler output, with strict validation."""
        match = re.search(r"<state_update>(.*?)</state_update>", text, re.DOTALL)
        if not match:
            return
        body = match.group(1).strip()
        for action_line in body.split("\n"):
            action_line = action_line.strip()
            if not action_line:
                continue

            # UPDATE character: name.field = value
            m = re.match(r"UPDATE\s+character:\s+(\S+)\.(\S+)\s*=\s*(.+)", action_line)
            if m:
                char_name = validate_char_name(m.group(1))
                if not char_name:
                    continue
                field = m.group(2)
                value = m.group(3).strip()
                self.update_character(char_name, line_num, dialogue_text=dialogue)
                if field == "alias":
                    cleaned_val = validate_char_name(value)
                    if cleaned_val:
                        self.add_alias(char_name, cleaned_val)
                continue

            # UPDATE alias: alias_name -> real_name
            m = re.match(r"UPDATE\s+alias:\s+(.+?)\s*->\s*(.+)", action_line)
            if m:
                alias = validate_char_name(m.group(1).strip())
                real = validate_char_name(m.group(2).strip())
                if alias and real:
                    # Check for identity reveal patterns
                    intro_patterns = ["名字是", "我叫", "咱是", "咱的名字是", "吾乃", "我是", "名は"]
                    is_reveal = any(p in dialogue for p in intro_patterns)
                    self.add_alias(real, alias, is_identity_reveal=is_reveal)
                continue

            # NEW character: name
            m = re.match(r"NEW\s+character:\s+(.+)", action_line)
            if m:
                raw_name = m.group(1).strip()
                cleaned = validate_char_name(raw_name)
                if not cleaned:
                    continue
                # Don't create duplicate if name already exists
                if cleaned not in self.state["characters"]:
                    self.add_character(cleaned)
                continue


# ============================================================
# Short-term Memory Agent
# ============================================================

class ShortMemAgent:
    def __init__(self, max_rounds=20):
        self.max_rounds = max_rounds
        self.history = []  # [(line_num, dialogue_text, speaker, reason, narrative_before)]
        self.confirmed_keys = set()
        self.phase_violation = False  # set by Boss when model contradicts alternating pattern

    def update(self, line_num, dialogue_text, speaker, reason="", narrative_before="", confirmed=True):
        self.history.append((line_num, dialogue_text, speaker, reason, narrative_before))
        key = (line_num, dialogue_text)
        if confirmed:
            self.confirmed_keys.add(key)
        if len(self.history) > self.max_rounds:
            removed = self.history.pop(0)
            self.confirmed_keys.discard((removed[0], removed[1]))

    def _confirmed_history(self):
        return [
            entry for entry in self.history
            if (entry[0], entry[1]) in self.confirmed_keys
        ]

    def _rhythm_speaker(self, speaker):
        cleaned = validate_char_name(speaker)
        if not cleaned or cleaned in {NON_PERSON_LABEL, "?"}:
            return None
        if is_generic_speaker_label(cleaned):
            return None
        return cleaned

    def detect_rapid_exchange(self, min_length=3):
        """Detect if last N entries alternate between two speakers."""
        if len(self.history) < min_length:
            return False
        recent = [self._rhythm_speaker(entry[2]) for entry in self.history[-min_length:] if self._rhythm_speaker(entry[2])]
        unique = list(dict.fromkeys(recent))
        if len(unique) != 2:
            return False
        for i in range(1, len(recent)):
            if recent[i] == recent[i-1]:
                return False
        return True

    def _get_exchange_rhythm(self):
        """Detect rapid 2-person exchange pattern and return (text, next_expected)."""
        confirmed = self._confirmed_history()
        if len(confirmed) < 4:
            return None, None
        recent = confirmed[-8:]  # last 8 confirmed anchors max
        speakers = [self._rhythm_speaker(sp) for _, _, sp, _, _ in recent]
        speakers = [sp for sp in speakers if sp]

        # Get the last two distinct speakers (in order of first appearance)
        seen = []
        for sp in speakers:
            if sp not in seen:
                seen.append(sp)
            if len(seen) == 2:
                break
        if len(seen) != 2:
            return None, None  # Not a 2-person exchange

        a, b = seen[0], seen[1]

        # Check if the recent sequence is a rapidly alternating pattern
        # Criteria: at least 4 rounds, both speakers appear at least 2 times
        count_a = speakers.count(a)
        count_b = speakers.count(b)
        if count_a < 2 or count_b < 2:
            return None, None

        # Check how well they alternate (allow one deviation)
        expected = a
        switches = 0
        for sp in speakers:
            if sp == expected:
                switches += 1
                expected = b if expected == a else a
        # If more than half the positions match an alternating pattern, it's likely a dialogue
        if switches >= len(speakers) * 0.6:
            next_expected = b if speakers[-1] == a else a
            text = (f"[Exchange rhythm - last {len(speakers)} rounds]\n"
                    f"  Alternating: {a} \u2194 {b}\n"
                    f"  Current expects: {next_expected} (opposite of previous speaker)\n"
                    f"  NEXT EXPECTED: {next_expected}")
            return text, next_expected
        return None, None

    def get_next_expected(self):
        """Get a weak expectation from confirmed anchors only, or None."""
        _, next_exp = self._get_exchange_rhythm()
        return next_exp

    def get_confirmed_summary(self):
        """Return only independently confirmed anchors for quality-mode agents."""
        confirmed = self._confirmed_history()
        if not confirmed:
            return "(No confirmed dialogue anchors yet)"
        lines = ["[Confirmed dialogue anchors]"]
        for line_num, dialogue, speaker, _, _ in confirmed[-8:]:
            lines.append(f"  L{line_num}「{dialogue[:50]}」-> {speaker}")
        return "\n".join(lines)

    def get_recent_speakers_hint(self):
        """Compact speaker-order hint for the Labeler."""
        if not self.history:
            return ""
        recent = [speaker for _, _, speaker, _, _ in self.history[-8:]]
        unique = []
        for speaker in recent:
            if speaker not in unique:
                unique.append(speaker)
        lines = ["[Recent speaker order]"]
        lines.append("  " + " -> ".join(recent))
        lines.append(f"  Last speaker: {recent[-1]}")
        if len(unique) == 2:
            next_expected = self.get_next_expected()
            pair = f"{unique[0]} <-> {unique[1]}"
            if next_expected:
                lines.append(f"  Weak two-person exchange hint: {pair}; possible next speaker only if no narrative break and no attribution: {next_expected}")
            else:
                lines.append(f"  Possible two-person exchange: {pair}")
        return "\n".join(lines)

    def get_summary(self):
        if not self.history:
            return "(No prior annotations)"
        lines = ["[Recent Dialogues]"]
        for line_num, dlg, speaker, reason, narr in self.history:
            narr_note = f" [before: {narr[:60]}]" if narr else ""
            line = f"  L{line_num}「{dlg[:50]}」-> {speaker}{narr_note}"
            lines.append(line)
        # Add exchange rhythm hint
        rhythm, _ = self._get_exchange_rhythm()
        if rhythm:
            lines.append("")
            lines.append(rhythm)

        # Add phase violation constraint for the next round
        if self.phase_violation:
            next_exp = self.get_next_expected()
            if next_exp:
                lines.append("")
                lines.append(f"[ANCHOR CONSTRAINT] Last round violated the alternating pattern!")
                lines.append(f"The last {min(len(self.history), 8)} rounds form {self._alternating_pair_name()}.")
                lines.append(f"Possible next speaker by weak alternation: {next_exp}")
                lines.append("RULE: Use this only when local raw text has no narrative break and no attribution.")

        # Add rapid exchange warning
        if self.detect_rapid_exchange(4):
            lines.append("")
            lines.append("RAPID EXCHANGE: Last 4 concrete named dialogues alternate between two speakers.")
            lines.append("RULE: Treat alternation as weak; narrative attribution and quote type come first.")
        return "\n".join(lines)

    def _alternating_pair_name(self):
        """Get the two alternating speaker names as a readable string."""
        if len(self.history) < 4:
            return "?"
        speakers = [entry[2] for entry in self.history[-min(len(self.history), 8):]]
        seen = []
        for s in speakers:
            if s not in seen:
                seen.append(s)
            if len(seen) == 2:
                break
        if len(seen) == 2:
            return f"{seen[0]} ↔ {seen[1]} alternation"
        return "alternation"


# ============================================================
# FactCurator Agent - structured character fact maintenance
# ============================================================

class FactCurator:
    """Maintains structured character facts, runs every N rounds."""

    def __init__(self, curator_every=10):
        self.curator_every = curator_every
        self.curation_count = 0
        self.pending_rounds = []
        self.fact_summary = "(No character fact database yet)"

    def add_round(self, line_num, speaker, summary):
        self.pending_rounds.append({"line": line_num, "speaker": speaker, "summary": summary})

    def should_curate(self):
        return len(self.pending_rounds) >= self.curator_every

    def curate(self, short_mem_text, char_state_json, round_log):
        if not self.pending_rounds:
            return
        summaries = "\n".join([f"#{r['line']}: {r['speaker']} | {r['summary']}" for r in self.pending_rounds])
        system_prompt = "You are a character database curator. Examine the current character state and recent annotations. Output ONLY a JSON object: {\"updates\":[...]}. Each update: {\"action\":\"add_character\"|\"add_alias\"|\"none\",\"target\":\"name\",\"detail\":{}}. If nothing new, output {\"updates\":[{\"action\":\"none\"}]}."
        user_content = f"""Current character state:
{json.dumps(char_state_json, ensure_ascii=False, indent=2)[:1500]}

Recent annotations:
{short_mem_text[:1000]}

Summaries:
{summaries[:1000]}

Check for new characters, alias relationships, or character facts."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        text, pec, ec, _ = call_ollama(messages, label="FactCurator")
        log_agent(round_log, "FactCurator", "curator", messages, text, pec, ec)
        # Build fact summary from current character state
        chars = char_state_json.get("characters", {})
        summary_lines = ["[Character Fact Database]"]
        if chars:
            for name, info in chars.items():
                aliases = info.get("aliases", [])
                non_self = [a for a in aliases if a != name]
                alias_str = f" (also: {', '.join(non_self)})" if non_self else ""
                last = info.get("last_seen_line", "?")
                summary_lines.append(f"  {name}{alias_str} (last L{last})")
        else:
            summary_lines.append("  (no characters yet)")
        self.fact_summary = "\n".join(summary_lines)
        self.pending_rounds = []
        self.curation_count += 1

    def get_summary(self):
        return self.fact_summary


# ============================================================
# Labeler Agent
# ============================================================

class LabelerAgent:
    def __init__(self):
        self.tool_call_count = 0

    def label(self, line_num, dialogue, short_mem_text, fact_summary,
               char_state_text, navigation_text, scene_summary, round_log, quiet=False,
               override_force_tool=True, recent_speakers_hint="", structure_hint="", local_evidence_hint=""):
        """
        Label one dialogue's speaker. Returns (speaker, summary, reason, pec, ec).
        """

        system_prompt = """You are a dialogue speaker annotation assistant. Your ONLY task is to identify who speaks a given line of dialogue in a Chinese novel.

========================================
TOOL - Your ONLY source for novel text
========================================

Tool: read_novel_lines(start, count)
- Reads 'count' lines from novel.txt starting at line 'start' (1-based)
- Returns each line as: "line_number: content"
- You can call this multiple times, in any order, for any range
**You do NOT have the novel text. You MUST use read_novel_lines to read it.**

========================================
EVIDENCE HIERARCHY (Priority)
========================================

1. [HIGHEST] Speech verbs naming the speaker in the IMMEDIATE context
   Lines like: "XX说", "XX喊道", "XX开口", "XX回答", "XX问", "XX叹息", "XX回答"
   If you find one within 5 lines of the dialogue → USE IT. You are done.

2. [HIGH] Narrative-position evidence
   The paragraph/sentence structure around the dialogue. E.g.:
   - "XX做了某事，然后说：" → XX is the speaker
   - "XX说道：" → XX is the speaker
   - "XX回答那人说：" → XX is the speaker
   The line IMMEDIATELY before a 「dialogue」 is the most important.

3. [LOW] Alternating dialogue pattern
   Useful only as a tie-breaker when two adjacent character speeches have no narrative break and no explicit attribution.
   A narrative paragraph between two lines BREAKS the alternating pattern.

4. [LOWEST] Character availability / scene presence
   Just because a character was mentioned or recently active does NOT mean they are speaking.

========================================
CRITICAL RULES - Do NOT Violate
========================================

RULE A: Speech verbs over everything
- If you find "说/喊道/问/开口/回答/继续说/低语" naming a character within 5 lines, THAT is the speaker
- Ignore any narrative that describes appearance/thoughts/actions of a different character
- Narrative describing how someone looks/feels does NOT prove they are speaking

RULE B: Read locally FIRST
- Start reading from 5-10 lines BEFORE the dialogue line
- The evidence you need is almost always within 5 lines
- Only expand search range if you find NO speech verb in the immediate context
- Do NOT search 40+ lines away unless local search found nothing

RULE C: Distinguish "speaker" from "mentioned person"
- "被XX的人物" = mentioned, not speaking
- "XX的人物" = describes someone, not speaking
- A line like "XX看着YY" describes the observer (XX), not the speaker
- Look for the pattern: [Narrative about character A + speech verb]「dialogue」→ A is speaker

RULE D: Decide quote type BEFORE speaker identity
- First classify the quoted text: actual character speech, narrator wording, sound effect, title/term, embedded quote, hypothetical "as if saying", or reported speech.
- If nobody in the scene actually speaks the quoted text, use "非人物发声".
- If it is actual character/group speech, continue to identify the speaker.
- Do NOT assign a narrator/sound/title quote to a nearby character just because that character is present.

RULE E: Speaker names - prefer canonical identity when available
- If the character has a specific name anywhere in reliable context/state/evidence, output the name, not a generic role.
- Role or appearance labels (girl, woman, merchant, clerk, guest, craftsman, etc.) are temporary until proven unnamed.
- If a descriptor/role is linked to a named character by self-introduction, narration, repeated alias evidence, or character state, output the canonical name.
- If the character is genuinely unnamed, use the most stable descriptive label from the text. Prefer role/function over vague appearance when both are available.
- Group/collective speech → use the group name, NOT "非人物发声".
- Do NOT make up a name or use the wrong role.

RULE F: Non-person speech - ONLY for these cases
- Ambient sounds, object sounds, sound effects → "非人物发声"
- If the text explicitly says a character made the sound, credit the character
- Collective shouting/group speech is NOT "非人物发声"

RULE G: Alternating dialogue - specific rules
- Alternation is WEAK evidence. It is useful only when there is no narrative break and no explicit attribution.
- If a narrative paragraph appears between two dialogues → the pattern RESETS.
- One character CAN speak multiple consecutive lines. Do not force alternation.
- RULE: If immediate narrative contains a speech verb → that overrides ALL alternating patterns.
- Use [Local dialogue structure] as a map: it tells you whether the previous dialogue is separated by narrative.
- Use [Recent speaker order] only after checking local text; it is a clue, not proof, and may contain earlier mistakes.

RULE H: Report evidence strength honestly
- evidence_basis=explicit_attribution only when local raw text directly attributes the target quote with a speech/thought verb.
- evidence_basis=alternation only when you rely mainly on adjacent dialogue order; this is not high confidence unless raw text also supports it.
- confidence=high only when a nearby line gives direct attribution, verified identity alias, or clear quote-type evidence.

========================================
WORKFLOW
========================================

1. CALL read_novel_lines for lines around the target (±15 lines)
   Read lines 10-15 BEFORE the dialogue first. Look for speech verbs naming the speaker.

2. If you find a speech verb within 5 lines → CONFIRM and output answer immediately
   Do NOT search further. You have your answer.

3. If no speech verb found → expand search range (±30 lines)
   Check for:
   a. Who spoke last (from auxiliary info and text)
   b. Is there an alternating pattern?
   c. Did the scene change?

4. If using a descriptor/role instead of a name → verify whether it has a named identity.
   Search nearby context, character state, evidence summaries, and forward/backward text for introductions or alias links.
   Look for patterns such as "我叫XX", "我是XX", "名字是XX", "吾乃XX", or narration that says the descriptor is named XX.
   If found → use the revealed name. If not found → keep the stable descriptor.

5. Output with <answer>, <quote_type>, <evidence_basis>, <confidence>, <reason>, and <summary>. Optionally add <discovery> if you find new character info.

========================================
OUTPUT FORMAT
========================================

<answer>speaker_name</answer>
- One single speaker name, or "非人物发声"
- Do NOT use "|" to separate multiple names
- If unsure, use a descriptive label (what the novel calls them). This triggers an automated search.

<quote_type>direct_speech|group_speech|embedded_quote|thought_or_narration|sound_or_text|unclear</quote_type>
<evidence_basis>explicit_attribution|speaker_action|identity_alias|quote_type|alternation|inference|unknown</evidence_basis>
<confidence>high|medium|low</confidence>

<reason>your reasoning</reason>
- In English or Chinese (both OK)
- Cite specific line numbers as evidence
- State which speech verb or evidence you found
- If auxiliary info contradicts the novel text, explain why

<summary>brief summary</summary>
- Format: speaker_name | what they said/did
- Keep concise for memory

<discovery>(Optional) Identity evidence discovered</discovery>
- Only include if you found new evidence about a character's identity
- Example: "角色在L42自我介绍为XX"
- This will be processed by a separate search agent"""

        user_content = f"""Annotate the speaker of this dialogue:

Line {line_num}: 「{dialogue}」

========================================
Auxiliary info - original text in navigation is 100% accurate.
Short-term memory labels and character state are reference only.
========================================

{scene_summary}

[Original Text Navigation - around target line]
{navigation_text}

{structure_hint}

{local_evidence_hint}

{short_mem_text}

{fact_summary}

{recent_speakers_hint}

[Character State]
{char_state_text}

========================================
You can use TWO tools:
1. read_novel_lines(start, count) — read specific lines
2. search_novel(keyword, context_lines=2) — search by keyword/name
Start reading 5-10 lines BEFORE the target line. Look for speech verbs naming the speaker.
If found, you are done. If not, expand gradually or use search_novel.
If you think an auxiliary label is wrong, say so in <reason>.
========================================="""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        total_pec = 0
        total_ec = 0
        tool_call_log = []
        cumulative_tokens = 0
        answered = False

        for round_i in range(MAX_TOOL_ROUNDS):
            text, pec, ec, tool_calls = call_ollama(messages, tools=LABELER_TOOLS, label=f"Labeler-R{round_i+1}")
            total_pec += pec
            total_ec += ec
            cumulative_tokens += pec + ec

            if not tool_calls:
                # Force at least one tool call on first round (unless override)
                if round_i == 0 and override_force_tool:
                    if not quiet:
                        print(f"    Model didn't call tool, forcing...")
                    messages.append({"role": "user", "content": "You MUST call read_novel_lines to read the novel text. You cannot answer without reading."})
                    continue
                # Check for garbage output on non-first rounds
                temp_speaker = self._parse_answer(text)
                if self._is_garbage_speaker(temp_speaker):
                    if not quiet:
                        print(f"    Garbage output '{temp_speaker}', asking model to self-correct...")
                    messages.append({"role": "user", "content": f"Your answer '{temp_speaker}' is not a valid character name. Think again based on the novel text and output a proper <answer>."})
                    continue
                log_agent(round_log, "Labeler", "labeler", messages, text, pec, ec,
                          total_pec=total_pec, total_ec=total_ec,
                          tool_calls_list=tool_call_log if tool_call_log else None)
                answered = True
                break

            tc_details = []
            parsed_tool_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                name = func.get("name", "")
                func_args, parse_error = parse_tool_arguments(name, raw_args)
                parsed_tool_calls.append({
                    "tool_call": normalize_tool_call(tc, func_args),
                    "function": name,
                    "arguments": func_args,
                    "parse_error": parse_error,
                })
                detail = {
                    "function": func.get("name"),
                    "arguments": func_args
                }
                if parse_error:
                    detail["argument_parse_note"] = parse_error
                tc_details.append(detail)

            tool_call_log.append({
                "round": round_i + 1,
                "function": tc_details[0]["function"] if tc_details else "",
                "args": tc_details[0]["arguments"] if tc_details else {},
                "prompt_eval_count": pec,
                "eval_count": ec,
            })

            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [item["tool_call"] for item in parsed_tool_calls],
            })

            for item in parsed_tool_calls:
                name = item["function"]
                func_args = item["arguments"]
                parse_error = item["parse_error"]
                if parse_error and not func_args:
                    invalid_msg = f"Tool arguments invalid: {parse_error}. Call the tool again with valid JSON arguments."
                    trace_tool_result("Labeler", round_i + 1, name, {}, invalid_msg, note="invalid tool arguments")
                    messages.append({"role": "tool", "content": invalid_msg})
                    continue
                if name == "read_novel_lines":
                    try:
                        s = int(func_args.get("start", 1) or 1)
                        c = int(func_args.get("count", 10) or 10)
                    except (TypeError, ValueError):
                        invalid_msg = "Tool arguments invalid: start and count must be integers. Call read_novel_lines again with valid JSON."
                        trace_tool_result("Labeler", round_i + 1, name, func_args, invalid_msg, note="invalid read_novel_lines arguments")
                        messages.append({"role": "tool", "content": invalid_msg})
                        continue
                    result = read_novel_lines(s, c)
                    self.tool_call_count += 1
                    if not quiet:
                        print(f"    Tool call #{self.tool_call_count}: read_novel_lines({s}-{s+c-1}) -> {len(result)} chars, pec={pec}, ec={ec}")
                    trace_tool_result("Labeler", round_i + 1, name, {"start": s, "count": c}, result)
                    messages.append({"role": "tool", "content": result})
                elif name == "search_novel":
                    keyword = func_args.get("keyword", "")
                    try:
                        ctx = int(func_args.get("context_lines", 2) or 2)
                    except (TypeError, ValueError):
                        ctx = 2
                    if not keyword:
                        invalid_msg = "Tool arguments invalid: keyword is required. Call search_novel again with valid JSON."
                        trace_tool_result("Labeler", round_i + 1, name, func_args, invalid_msg, note="missing search keyword")
                        messages.append({"role": "tool", "content": invalid_msg})
                        continue
                    result = search_novel(keyword, ctx)
                    self.tool_call_count += 1
                    if not quiet:
                        print(f"    Tool call #{self.tool_call_count}: search_novel('{keyword}') -> {len(result)} chars, pec={pec}, ec={ec}")
                    trace_tool_result("Labeler", round_i + 1, name, {"keyword": keyword, "context_lines": ctx}, result)
                    messages.append({"role": "tool", "content": result})
                else:
                    invalid_msg = f"Unsupported tool '{name}'. Use read_novel_lines or search_novel."
                    trace_tool_result("Labeler", round_i + 1, name, func_args, invalid_msg, note="unsupported tool")
                    messages.append({"role": "tool", "content": invalid_msg})

            # Token budget check
            if cumulative_tokens >= TOKEN_BUDGET:
                if not quiet:
                    print(f"    Token budget reached ({cumulative_tokens} >= {TOKEN_BUDGET}), stopping search")
                # Force final answer on next call
                messages.append({"role": "user", "content": "Token budget reached. Output your best answer with <answer>, <reason>, and <summary> tags."})
                continue

        else:
            if not quiet:
                print(f"    Max tool rounds reached ({MAX_TOOL_ROUNDS})")

        if not answered or not self._parse_answer(text):
            messages.append({
                "role": "user",
                "content": (
                    "Stop using tools. Based on the tool results already shown, output exactly one final "
                    "answer with <answer>, <reason>, and <summary> tags."
                ),
            })
            text, pec, ec, tool_calls = call_ollama(messages, tools=None, label="Labeler-Final")
            total_pec += pec
            total_ec += ec
            log_agent(round_log, "Labeler", "labeler", messages, text, pec, ec,
                      total_pec=total_pec, total_ec=total_ec,
                      tool_calls_list=tool_call_log if tool_call_log else None)

        round_log["tool_calls"] = tool_call_log

        # Track how many tool rounds the Labeler used (for Verifier decision)
        tool_rounds_used = len(tool_call_log)

        speaker = self._parse_answer(text)
        summary = self._parse_summary(text)
        reason = self._parse_reason(text)
        round_log["labeler_evidence"] = {
            "quote_type": self._parse_tag(text, "quote_type", "unclear"),
            "evidence_basis": self._parse_tag(text, "evidence_basis", "unknown"),
            "confidence": self._parse_tag(text, "confidence", "low"),
        }

        return speaker, summary, reason, total_pec, total_ec, tool_rounds_used

    def _parse_tag(self, text, tag, default=""):
        match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text or "", re.DOTALL)
        return match.group(1).strip() if match else default

    def _is_garbage_speaker(self, speaker):
        """Check if a speaker name is clearly garbage (not a valid character name)."""
        if not speaker:
            return True
        # Contains English letters -> garbage (Chinese novels should have Chinese names)
        if re.search(r'[a-zA-Z]', speaker):
            return True
        # Question words = model is confused
        if speaker in ("什么", "啥", "谁", "哪个", "哪里", "为什么", "怎么"):
            return True
        # Contains only punctuation/symbols
        if not re.search(r'[\u4e00-\u9fff]', speaker):
            return True
        return False

    def _parse_answer(self, text):
        """Extract and validate speaker name from <answer> tag."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            cleaned = validate_char_name(raw)
            if cleaned:
                return cleaned
            # If validation failed, try to extract first valid name
            # Sometimes model outputs "name (description)" - take just the name
            parts = re.split(r'\s*[—\-–(（]', raw)
            if parts:
                cleaned = validate_char_name(parts[0].strip())
                if cleaned:
                    return cleaned
            return raw  # Return as-is if we can't clean it
        # Fallback: first line of response
        fallback = text.strip().split("\n")[0][:50]
        cleaned = validate_char_name(fallback)
        return cleaned

    def _parse_summary(self, text):
        match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_reason(self, text):
        match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if match:
            return match.group(1).strip()[:200]
        return ""


# ============================================================
# Verifier Agent - independent cross-check of Labeler's output
# ============================================================

class VerifierAgent:
    """
    Independent verification agent.
    Each round, after Labeler outputs a speaker, Verifier reads the same context
    and independently determines the speaker. If it disagrees with Labeler,
    returns the suggested correction.
    """

    def __init__(self):
        self.tool_call_count = 0

    def _safe_print(self, msg):
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except (ValueError, OSError):
            pass

    def verify(self, line_num, dialogue, navigation_text, short_mem_text,
               scene_summary, char_state_text, labeler_speaker, round_log, quiet=False,
               risk_reasons=None, structure_hint="", local_evidence_hint=""):
        """
        Independent verification. Returns (verdict, suggested_speaker, reason).
        verdict: "confirm" or "disagree"
        """
        system_prompt = """You are an independent dialogue speaker VERIFIER. A primary annotator has already labeled a dialogue. Your job is to independently determine the speaker and compare.

You have the same information available as the primary annotator:
1. Original novel text in navigation (100% accurate)
2. Recent annotation history
3. Character state
4. The primary annotator's answer

RULES:
- Read the novel text independently using read_novel_lines
- Form your OWN conclusion about who is speaking
- Then compare with the primary answer
- First classify the quote type: direct_speech, quote_or_letter, password_or_signal, sound_effect, thought_not_spoken, or unclear
- Use 非人物发声 for non-person labels. Never output narrator, narr, non-person, or non-human as a speaker name.
- Embedded quotes are not automatically non-person. If the local text attributes the quote to a character
  (for example: a character says it, wants to say it, is about to say it, or looks as if saying it),
  keep that character as the speaker unless there is stronger contrary evidence.
- For quote_or_letter/password_or_signal/sound_effect/thought_not_spoken, be careful: the correct label may be 非人物发声, a signal, or the attributed source rather than the nearby character
- For rapid two-person exchanges, do not rely on alternation alone; cite the local narrative or adjacent dialogue structure.
- Alternation may justify a risk warning, but it is not enough for high-confidence disagreement by itself.
- Be conservative. Disagree only when explicit local evidence, verified identity alias, or clear quote-type evidence contradicts the primary answer.
- evidence_basis=explicit_attribution only when the local text directly attributes the target quote to a source
  with a speech/thought verb such as says, asks, answers, continues, wants to say, or looks as if saying.
- evidence_basis=speaker_action is weaker. Do not use it for a person's later movement or reaction unless it
  directly frames the target quote.
- If you agree, output <verdict>confirm</verdict>
- If you disagree, output <verdict>disagree</verdict> and provide your suggested speaker

OUTPUT:
<quote_type>type</quote_type>
<evidence_basis>explicit_attribution|speaker_action|identity_alias|quote_type|alternation|inference|unknown</evidence_basis>
<confidence>high|medium|low</confidence>
<verdict>confirm</verdict> or <verdict>disagree</verdict>
<suggested_speaker>name</suggested_speaker>
<reason>your evidence citing specific line numbers</reason>"""

        risk_text = "\n".join(f"- {r}" for r in (risk_reasons or [])) or "(none)"
        compact_navigation = compact_text(navigation_text, 8000, keep="middle")
        compact_short_mem = compact_text(short_mem_text, 4000, keep="tail")
        compact_char_state = compact_text(char_state_text, 4000, keep="tail")
        user_content = f"""Verify the speaker of this dialogue:

Line {line_num}: 「{dialogue}」

Primary annotator's answer: {labeler_speaker}

Risk reasons that triggered verification:
{risk_text}

========================================
Original text navigation:
{compact_navigation}

{structure_hint}

{local_evidence_hint}

{compact_short_mem}

{scene_summary}

[Character State]
{compact_char_state}

========================================
Call read_novel_lines to verify. Form your own conclusion first, then compare.
========================================"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        total_pec = 0
        total_ec = 0
        tool_call_log = []
        text = ""
        tool_results = []

        # Allow tool read + final verdict. Some APIs need a second model turn
        # after the tool result before they can produce the verdict.
        for round_i in range(3):
            text, pec, ec, tool_calls = call_ollama(messages, tools=[TOOL_READ_NOVEL], label=f"Verifier-R{round_i+1}")
            total_pec += pec
            total_ec += ec
            if not tool_calls:
                break
            parsed_tool_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                name = func.get("name", "")
                func_args, parse_error = parse_tool_arguments(name, raw_args)
                parsed_tool_calls.append({
                    "tool_call": normalize_tool_call(tc, func_args),
                    "function": name,
                    "arguments": func_args,
                    "parse_error": parse_error,
                })
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [item["tool_call"] for item in parsed_tool_calls],
            })
            for item in parsed_tool_calls:
                name = item["function"]
                func_args = item["arguments"]
                parse_error = item["parse_error"]
                if parse_error and not func_args:
                    invalid_msg = f"Tool arguments invalid: {parse_error}. Call read_novel_lines again with valid JSON arguments."
                    trace_tool_result("Verifier", round_i + 1, name, {}, invalid_msg, note="invalid tool arguments")
                    messages.append({"role": "tool", "content": invalid_msg})
                    tool_results.append(invalid_msg)
                    continue
                if name == "read_novel_lines":
                    try:
                        s = int(func_args.get("start", 1) or 1)
                        c = int(func_args.get("count", 10) or 10)
                    except (TypeError, ValueError):
                        invalid_msg = "Tool arguments invalid: start and count must be integers. Call read_novel_lines again with valid JSON."
                        trace_tool_result("Verifier", round_i + 1, name, func_args, invalid_msg, note="invalid read_novel_lines arguments")
                        messages.append({"role": "tool", "content": invalid_msg})
                        tool_results.append(invalid_msg)
                        continue
                    result = read_novel_lines(s, c)
                    trace_tool_result("Verifier", round_i + 1, name, {"start": s, "count": c}, result)
                    tool_results.append(f"[read_novel_lines start={s} count={c}]\n{result}")
                    self.tool_call_count += 1
                    tool_call_log.append({
                        "round": round_i + 1,
                        "function": "read_novel_lines",
                        "args": {"start": s, "count": c},
                        "prompt_eval_count": pec,
                        "eval_count": ec,
                    })
                    messages.append({"role": "tool", "content": result})
                else:
                    invalid_msg = f"Unsupported tool '{name}'. Use read_novel_lines."
                    trace_tool_result("Verifier", round_i + 1, name, func_args, invalid_msg, note="unsupported tool")
                    messages.append({"role": "tool", "content": invalid_msg})
                    tool_results.append(invalid_msg)

            final_user_content = f"""Based on the compact context and the tool result below, verify the speaker.

Line {line_num}: 「{dialogue}」
Primary annotator's answer: {labeler_speaker}

Risk reasons:
{risk_text}

[Compact local navigation]
{compact_navigation}

{structure_hint}

{local_evidence_hint}

[Tool result]
{compact_text(chr(10).join(tool_results), 12000, keep="tail")}

Output exactly:
<quote_type>type</quote_type>
<evidence_basis>explicit_attribution|speaker_action|identity_alias|quote_type|alternation|inference|unknown</evidence_basis>
<confidence>high|medium|low</confidence>
<verdict>confirm</verdict> or <verdict>disagree</verdict>
<suggested_speaker>name</suggested_speaker>
<reason>evidence citing line numbers</reason>"""
            final_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_user_content},
            ]
            text, pec, ec, _ = call_ollama(final_messages, tools=None, label="Verifier-Final")
            total_pec += pec
            total_ec += ec
            messages = final_messages
            break

        log_agent(round_log, "Verifier", "verifier", messages, text, pec if 'pec' in locals() else 0,
                  ec if 'ec' in locals() else 0, tool_calls_list=tool_call_log if tool_call_log else None,
                  total_pec=total_pec, total_ec=total_ec)

        # Parse verdict
        verdict = "confirm"
        suggested = labeler_speaker
        reason = ""
        basis = "unknown"
        confidence = "low"
        if "<verdict>disagree</verdict>" in text:
            verdict = "disagree"
            m = re.search(r"<suggested_speaker>(.*?)</suggested_speaker>", text, re.DOTALL)
            if m:
                raw = m.group(1).strip()
                cleaned = validate_char_name(raw)
                if cleaned:
                    suggested = cleaned
        m = re.search(r"<evidence_basis>(.*?)</evidence_basis>", text, re.DOTALL)
        if m:
            basis = m.group(1).strip().lower()
        m = re.search(r"<confidence>(.*?)</confidence>", text, re.DOTALL)
        if m:
            confidence = m.group(1).strip().lower()
        m = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        if m:
            reason = m.group(1).strip()[:200]

        if not quiet:
            if verdict == "disagree":
                self._safe_print(f"  Verifier disagrees: Labeler={labeler_speaker} -> Verifier={suggested}")
            else:
                self._safe_print(f"  Verifier confirms: {labeler_speaker}")

        return verdict, suggested, reason, basis, confidence


# ============================================================
# Quality Audit - baseline-preserving, scene-level SenseNova review
# ============================================================

class QualityAudit:
    """Audit a legacy candidate from independent views without exposing rationales across roles."""

    QUOTE_KINDS = {"person", "group", "non_person", "unclear"}
    QUOTE_TYPES = {
        "direct_speech", "group_speech", "embedded_quote",
        "thought_or_narration", "sound_or_text", "unclear",
    }
    EVIDENCE_BASES = {
        "explicit_attribution", "speaker_action", "identity_alias",
        "quote_type", "alternation", "inference", "unknown",
    }
    NON_SPEECH_REASON_PHRASES = (
        "not actual speech", "not spoken", "not as spoken", "not speech", "not dialogue",
        "not actual spoken", "no character utters", "nobody utters", "not a spoken line",
        "不是对话", "并非对话", "非对话", "不是人物说", "并非人物说", "无人说出",
        "没有角色说出", "非实际言语", "并非实际发言", "不是台词", "并非台词",
        "叙述性文字", "叙述性描写",
    )

    def __init__(self, dialogue_radius=None):
        radius = QUALITY_SCENE_RADIUS if dialogue_radius is None else dialogue_radius
        self.dialogue_radius = max(4, int(radius))
        self.model_call_count = 0

    @staticmethod
    def _same_speaker(left, right):
        return bool(left and right and (left == right or left in right or right in left))

    @staticmethod
    def _preferred_consensus_label(labels):
        labels = [validate_char_name(label) for label in labels]
        labels = [label for label in labels if label and is_valid_final_speaker(label)]
        if not labels:
            return ""
        for candidate in labels:
            if all(QualityAudit._same_speaker(candidate, other) for other in labels):
                return max(labels, key=len)
        return ""

    @classmethod
    def _speaker_vote_consensus(cls, labels, minimum_support):
        cleaned_labels = [validate_char_name(label) for label in labels]
        cleaned_labels = [label for label in cleaned_labels if label and is_valid_final_speaker(label)]
        best_label = ""
        best_support = 0
        for candidate in cleaned_labels:
            matching = [label for label in cleaned_labels if cls._same_speaker(candidate, label)]
            support = len(matching)
            preferred = max(matching, key=len) if matching else candidate
            if support > best_support or (support == best_support and len(preferred) > len(best_label)):
                best_label = preferred
                best_support = support
        if best_support < minimum_support:
            return "", best_support
        return best_label, best_support

    def _parse_card(self, text):
        raw_speaker = (
            get_ensemble_tag(text, "speaker")
            or get_ensemble_tag(text, "target_speaker")
            or get_ensemble_tag(text, "preferred_speaker")
        )
        speaker = validate_char_name(raw_speaker) if raw_speaker else ""
        citations = parse_line_citations(get_ensemble_tag(text, "citations"))
        return {
            "speaker": speaker or "",
            "voice_kind": get_ensemble_tag(text, "voice_kind", "unclear").strip().lower(),
            "quote_type": get_ensemble_tag(text, "quote_type", "unclear").strip().lower(),
            "evidence_basis": get_ensemble_tag(text, "evidence_basis", "unknown").strip().lower(),
            "confidence": get_ensemble_tag(text, "confidence", "low").strip().lower(),
            "citations": citations,
            "reason": get_ensemble_tag(text, "reason")[:500],
            "raw": text,
        }

    @staticmethod
    def _card_from_tool_args(args, raw_text=""):
        raw_speaker = (
            args.get("speaker")
            or args.get("target_speaker")
            or args.get("preferred_speaker")
            or ""
        )
        speaker = validate_char_name(raw_speaker) if raw_speaker else ""
        citations = args.get("citations") or []
        if isinstance(citations, (str, int)):
            citations = parse_line_citations(str(citations))
        else:
            normalized = []
            for item in citations:
                try:
                    normalized.append(int(item))
                except (TypeError, ValueError):
                    normalized.extend(parse_line_citations(str(item)))
            citations = list(dict.fromkeys(normalized))
        return {
            "speaker": speaker or "",
            "voice_kind": str(args.get("voice_kind") or "unclear").strip().lower(),
            "quote_type": str(args.get("quote_type") or "unclear").strip().lower(),
            "evidence_basis": str(args.get("evidence_basis") or "unknown").strip().lower(),
            "confidence": str(args.get("confidence") or "low").strip().lower(),
            "citations": citations,
            "reason": str(args.get("reason") or "")[:500],
            "raw": raw_text,
        }

    def _card_is_valid(self, card, packet, card_type):
        if not citations_are_local(card.get("citations", []), packet):
            return False
        if card_type == "quote":
            return (
                card.get("voice_kind") in self.QUOTE_KINDS
                and card.get("quote_type") in self.QUOTE_TYPES
            )
        return (
            is_valid_final_speaker(card.get("speaker"))
            and card.get("evidence_basis") in self.EVIDENCE_BASES
        )

    @classmethod
    def _normalize_card_consistency(cls, card, card_type):
        reason = (card.get("reason") or "").lower()
        if not any(phrase in reason for phrase in cls.NON_SPEECH_REASON_PHRASES):
            return card
        normalized = dict(card)
        if card_type == "quote":
            normalized["voice_kind"] = "non_person"
            if normalized.get("quote_type") in {"unclear", "direct_speech", "group_speech"}:
                normalized["quote_type"] = "thought_or_narration"
        else:
            normalized["speaker"] = NON_PERSON_LABEL
            normalized["evidence_basis"] = "quote_type"
        normalized["consistency_normalized"] = True
        return normalized

    def _run_xml_agent(self, agent_name, role, system_prompt, user_content,
                       round_log, packet, card_type):
        total_pec = 0
        total_ec = 0
        last_card = {}
        last_text = ""
        for attempt in range(1, QUALITY_AUDIT_RETRIES + 1):
            correction = ""
            if attempt > 1:
                correction = (
                    "\n\nYour previous response was unusable. Re-read the raw text and produce a fresh "
                    "decision. Copy Chinese names exactly from the source. Include at least one local "
                    "L<number> citation. Do not output English names, unknown, unclear as a speaker, or prose "
                    "before the required submit_review tool call."
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content + correction},
            ]
            output_tool = json.loads(json.dumps(QUALITY_OUTPUT_TOOL))
            if card_type == "quote":
                required = ["quote_type", "voice_kind", "confidence", "citations", "reason"]
            else:
                required = ["speaker", "evidence_basis", "confidence", "citations", "reason"]
            output_tool["function"]["parameters"]["required"] = required
            text, pec, ec, tool_calls = call_ollama(
                messages,
                tools=[output_tool],
                label=f"{agent_name}-A{attempt}",
                request_timeout=QUALITY_REQUEST_TIMEOUT,
                tool_choice="required",
            )
            self.model_call_count += 1
            total_pec += pec
            total_ec += ec
            tool_args = {}
            for tool_call in tool_calls or []:
                function = tool_call.get("function", {}) or {}
                if function.get("name") != "submit_review":
                    continue
                parsed_args, parse_error = parse_tool_arguments(
                    "submit_review", function.get("arguments", "{}")
                )
                if parsed_args and not parse_error:
                    tool_args = parsed_args
                    break
                if parsed_args:
                    tool_args = parsed_args
            card = self._card_from_tool_args(tool_args, text) if tool_args else self._parse_card(text)
            card = self._normalize_card_consistency(card, card_type)
            valid = self._card_is_valid(card, packet, card_type)
            card["valid"] = valid
            log_agent(
                round_log,
                agent_name if attempt == 1 else f"{agent_name}Retry{attempt}",
                role,
                messages,
                text,
                pec,
                ec,
                tool_calls_list=(
                    [{"function": "submit_review", "arguments": tool_args}]
                    if tool_args else None
                ),
            )
            last_card = card
            last_text = text
            if valid:
                break
        last_card["raw"] = last_text
        return last_card, total_pec, total_ec

    @staticmethod
    def _quote_output_format():
        return (
            "Call submit_review immediately. Set voice_kind, quote_type, confidence, integer citations, "
            "and a brief reason. Do not write analysis before the tool call."
        )

    @staticmethod
    def _speaker_output_format():
        return (
            "Call submit_review immediately. Set speaker to one Chinese source label or 非人物发声, then "
            "set evidence_basis, confidence, integer citations, and a brief reason. Do not write analysis "
            "before the tool call."
        )

    def _select_decision(self, baseline, baseline_evidence, quote_cards,
                         attribution_cards, final_card, packet):
        baseline = validate_char_name(baseline) or baseline
        valid_quotes = [card for card in quote_cards if card.get("valid")]
        valid_attributions = [card for card in attribution_cards if card.get("valid")]
        quote_counts = Counter(card.get("voice_kind") for card in valid_quotes)
        attribution_labels = [card.get("speaker") for card in valid_attributions]
        minimum_scene_support = max(3, (len(attribution_cards) * 4 + 4) // 5)
        attribution_consensus, attribution_support = self._speaker_vote_consensus(
            attribution_labels,
            minimum_scene_support,
        )
        attribution_plurality, attribution_plurality_support = self._speaker_vote_consensus(
            attribution_labels,
            1,
        )

        final_speaker = final_card.get("speaker") if final_card.get("valid") else ""
        final_panel_support = int(final_card.get("panel_support") or 0)
        final_panel_size = int(final_card.get("panel_size") or 1)
        final_panel_speakers = list(final_card.get("panel_speakers") or [])
        scene_support_for_final = sum(
            self._same_speaker(final_speaker, label) for label in attribution_labels
        ) if final_speaker else 0
        attribution_all_explicit = bool(valid_attributions) and all(
            card.get("evidence_basis") == "explicit_attribution" for card in valid_attributions
        )
        final_support_for_attribution_consensus = sum(
            self._same_speaker(attribution_consensus, label)
            for label in final_panel_speakers
        ) if attribution_consensus else 0
        source = "baseline-retained"
        speaker = baseline

        nonperson_support = quote_counts.get("non_person", 0)
        attribution_nonperson = sum(
            validate_char_name(label) == NON_PERSON_LABEL for label in attribution_labels
        )
        person_support = quote_counts.get("person", 0) + quote_counts.get("group", 0)

        if (
            final_speaker == NON_PERSON_LABEL
            and nonperson_support >= 2
            and attribution_nonperson >= max(2, (len(attribution_cards) + 1) // 2)
        ):
            speaker = NON_PERSON_LABEL
            source = "audited-non-person"
        elif (
            baseline == NON_PERSON_LABEL
            and final_speaker
            and final_speaker != NON_PERSON_LABEL
            and person_support == len(quote_cards)
            and attribution_consensus
            and self._same_speaker(final_speaker, attribution_consensus)
        ):
            speaker = attribution_consensus
            source = "audited-person-speech"
        elif (
            baseline == NON_PERSON_LABEL
            and attribution_consensus
            and attribution_consensus != NON_PERSON_LABEL
            and attribution_support == len(attribution_cards)
            and len(attribution_labels) == len(attribution_cards)
            and attribution_all_explicit
            and final_support_for_attribution_consensus >= 1
        ):
            speaker = attribution_consensus
            source = "explicit-local-person-restoration"
        elif (
            attribution_consensus
            and attribution_support >= 4
            and len(attribution_labels) >= 4
            and attribution_consensus != NON_PERSON_LABEL
            and baseline != NON_PERSON_LABEL
            and not self._same_speaker(baseline, attribution_consensus)
        ):
            speaker = attribution_consensus
            source = "strong-local-panel-correction"
        elif (
            attribution_consensus
            and final_speaker
            and self._same_speaker(final_speaker, attribution_consensus)
            and not self._same_speaker(baseline, attribution_consensus)
        ):
            speaker = attribution_consensus
            source = "audited-speaker-correction"
        elif (
            final_speaker
            and not self._same_speaker(final_speaker, baseline)
            and final_panel_support >= 2
            and final_panel_size >= 3
            and scene_support_for_final >= 3
        ):
            speaker = final_speaker
            source = "joint-panel-correction"
        elif not is_valid_final_speaker(baseline) and final_speaker:
            speaker = final_speaker
            source = "invalid-baseline-replacement"

        if not is_valid_final_speaker(speaker):
            if is_valid_final_speaker(baseline):
                speaker = baseline
                source = "invalid-audit-fallback"
            else:
                speaker = "?"
                source = "unresolved"

        all_high = bool(valid_attributions) and all(
            card.get("confidence") == "high" for card in valid_attributions
        )
        all_explicit = bool(valid_attributions) and all(
            card.get("evidence_basis") == "explicit_attribution" for card in valid_attributions
        )
        final_high = final_card.get("confidence") == "high"
        confirmed_anchor = bool(
            source == "audited-speaker-correction"
            and attribution_consensus
            and all_high
            and all_explicit
            and final_high
            and final_card.get("evidence_basis") == "explicit_attribution"
        )
        if source == "audited-non-person":
            confirmed_anchor = bool(
                nonperson_support == len(quote_cards)
                and attribution_nonperson == len(attribution_cards)
                and final_high
            )

        return {
            "speaker": speaker,
            "baseline_speaker": baseline,
            "baseline_changed": not self._same_speaker(speaker, baseline),
            "selection_source": source,
            "confirmed_anchor": confirmed_anchor,
            "quote_votes": dict(quote_counts),
            "attribution_consensus": attribution_consensus,
            "attribution_support": attribution_support,
            "attribution_valid_votes": len(attribution_labels),
            "attribution_plurality": attribution_plurality,
            "attribution_plurality_support": attribution_plurality_support,
            "final_speaker": final_speaker,
            "final_panel_support": final_panel_support,
            "final_panel_size": final_panel_size,
            "scene_support_for_final": scene_support_for_final,
            "final_support_for_attribution_consensus": final_support_for_attribution_consensus,
            "quote_type": final_card.get("quote_type", baseline_evidence.get("quote_type", "unclear")),
            "evidence_basis": final_card.get("evidence_basis", "unknown"),
            "confidence": final_card.get("confidence", "low"),
            "citations": final_card.get("citations", []),
        }

    @classmethod
    def _apply_split_neighbor_result(cls, decision, baseline, plurality,
                                     previous_cards, next_cards):
        updated = dict(decision)
        valid_previous = [card for card in previous_cards if card.get("valid")]
        valid_next = [card for card in next_cards if card.get("valid")]
        previous_consensus, previous_support = cls._speaker_vote_consensus(
            [card.get("speaker") for card in valid_previous],
            2,
        )
        next_consensus, next_support = cls._speaker_vote_consensus(
            [card.get("speaker") for card in valid_next],
            2,
        )
        updated["previous_neighbor_speaker"] = previous_consensus
        updated["previous_neighbor_support"] = previous_support
        updated["next_neighbor_speaker"] = next_consensus
        updated["next_neighbor_support"] = next_support
        if (
            previous_support >= 2
            and next_support >= 2
            and cls._same_speaker(previous_consensus, baseline)
            and cls._same_speaker(next_consensus, baseline)
            and plurality
            and not cls._same_speaker(plurality, baseline)
        ):
            updated["speaker"] = plurality
            updated["baseline_changed"] = True
            updated["selection_source"] = "split-neighbor-anchor-correction"
            updated["confirmed_anchor"] = False
        return updated

    def decide(self, novel_lines, dialogue_list, line_num, dialogue, baseline,
               baseline_evidence, round_log):
        calls_before = self.model_call_count
        packet = build_dialogue_packet(
            novel_lines,
            dialogue_list,
            line_num,
            dialogue,
            dialogue_radius=self.dialogue_radius,
            raw_padding=8,
        )
        packet_text = format_dialogue_packet(packet, max_raw_chars=22000)
        local_packet = build_dialogue_packet(
            novel_lines,
            dialogue_list,
            line_num,
            dialogue,
            dialogue_radius=min(5, self.dialogue_radius),
            raw_padding=6,
        )
        local_packet_text = format_dialogue_packet(local_packet, max_raw_chars=12000)
        target = packet["target"]
        quote_common = (
            "The TARGET is one exact quoted occurrence, not every quotation on its raw line. "
            "Classify whether a person actually utters that exact span. Embedded wording, labels, "
            "writing, remembered wording, facial expressions, eye messages, and object sounds are not "
            "automatically character speech. A human-made sigh or cry is person speech when the raw text "
            "binds the sound to that person. Cite only supplied raw lines.\n\n"
        )
        quote_prompts = [
            (
                "QuoteSyntaxAudit",
                "Audit the grammatical container of the TARGET quote. Ignore character identity and topic. "
                "Determine whether the typography is actual utterance or quotation embedded in narration. ",
            ),
            (
                "QuoteAdversarialAudit",
                "Assume a nearby character assignment may be a false positive. Search specifically for wording "
                "that makes the TARGET thought, text, metaphorical message, or sound rather than speech. ",
            ),
            (
                "QuoteSceneAudit",
                "Read the surrounding scene and decide whether anyone in the scene actually produces the exact "
                "TARGET span. Do not identify the speaker; classify voice kind only. ",
            ),
        ]
        quote_cards = []
        total_pec = 0
        total_ec = 0
        for agent_name, instruction in quote_prompts:
            card, pec, ec = self._run_xml_agent(
                agent_name,
                "independent_quote_audit",
                instruction + quote_common + self._quote_output_format(),
                local_packet_text,
                round_log,
                local_packet,
                "quote",
            )
            quote_cards.append(card)
            total_pec += pec
            total_ec += ec

        local_solver_instruction = (
            "Solve the target-centered block from its first turn to its last turn. Assign every displayed turn "
            "chronologically before returning only the TARGET speaker. Check immediate narrative on both sides. "
            "When the turns on both sides belong to one participant, explicitly test TARGET as the intervening "
            "reply from another participant."
        )
        attribution_prompts = [
            (f"LocalChronologyAudit{index}", local_solver_instruction, local_packet_text, local_packet)
            for index in range(1, 6)
        ]
        speaker_common = (
            " Use only this novel's raw text. Copy a Chinese name or stable unnamed role exactly from the source; "
            "never translate or romanize it. A mentioned person or addressee is not necessarily the speaker. "
            "A line discussing a character's identity, occupation, species, belongings, or interests is not evidence "
            "that the character speaks it. Do not use outside knowledge or novel-specific assumptions. "
            + self._speaker_output_format()
        )
        attribution_cards = []
        for agent_name, instruction, content, evidence_packet in attribution_prompts:
            card, pec, ec = self._run_xml_agent(
                agent_name,
                "independent_scene_attribution",
                instruction + speaker_common,
                content,
                round_log,
                evidence_packet,
                "speaker",
            )
            attribution_cards.append(card)
            total_pec += pec
            total_ec += ec

        candidate_labels = []
        for candidate in [baseline] + [card.get("speaker") for card in attribution_cards]:
            cleaned = validate_char_name(candidate)
            if cleaned and is_valid_final_speaker(cleaned) and not any(
                self._same_speaker(cleaned, existing) for existing in candidate_labels
            ):
                candidate_labels.append(cleaned)
        candidate_labels.sort(key=lambda value: (sum(ord(ch) for ch in value) + line_num) % 997)
        valid_attribution_labels = [
            card.get("speaker") for card in attribution_cards if card.get("valid")
        ]
        candidate_support = {
            label: sum(self._same_speaker(label, vote) for vote in valid_attribution_labels)
            for label in candidate_labels
        }
        quote_vote_summary = Counter(
            card.get("voice_kind") for card in quote_cards if card.get("valid")
        )
        final_user = (
            f"{local_packet_text}\n\n[Anonymous candidate labels; order carries no meaning]\n"
            + "\n".join(
                f"  Candidate {idx + 1}: {label} | independent scene support "
                f"{candidate_support.get(label, 0)}/{len(attribution_cards)}"
                for idx, label in enumerate(candidate_labels)
            )
            + f"\n\n[Independent quote-kind vote counts]\n{dict(quote_vote_summary)}"
        )
        final_common = (
            " Re-read the raw scene yourself. Candidate labels are hypotheses and no prior reasoning is available. "
            "If the exact target is narration, embedded wording, written text, or an object/ambient sound that "
            "nobody utters, select 非人物发声. Copy Chinese labels from the source and do not invent or romanize "
            "names. Topic, persona, species, occupation, presumed pronoun habits, and characteristic vocabulary "
            "are invalid evidence unless the raw text independently binds the exact target quote to that speaker. "
            "Cite target-bound raw lines. " + self._speaker_output_format()
        )
        final_prompts = [
            (
                "FinalEvidenceJudge",
                "Judge by exact grammatical quote binding and immediate narrative on both sides.",
            ),
            (
                "FinalTurnSequenceJudge",
                "Assign every displayed turn chronologically, including both neighbors of TARGET, before selecting "
                "TARGET. Do not infer who uses a pronoun from outside knowledge; derive it from this raw block.",
            ),
            (
                "FinalCandidateChallengeJudge",
                "Find the candidate with the strongest independent scene support, then actively try to falsify it. "
                "Reject it only for a direct raw-text contradiction; a topic or presumed speech habit is not one.",
            ),
        ]
        final_cards = []
        for agent_name, instruction in final_prompts:
            card, pec, ec = self._run_xml_agent(
                agent_name,
                "raw_evidence_final_judge",
                instruction + final_common,
                final_user,
                round_log,
                local_packet,
                "speaker",
            )
            final_cards.append(card)
            total_pec += pec
            total_ec += ec
        valid_final_cards = [card for card in final_cards if card.get("valid")]
        final_consensus, final_support = self._speaker_vote_consensus(
            [card.get("speaker") for card in valid_final_cards],
            2,
        )
        if final_consensus:
            final_card = next(
                dict(card) for card in valid_final_cards
                if self._same_speaker(final_consensus, card.get("speaker"))
            )
            final_card["speaker"] = final_consensus
            final_card["valid"] = True
        else:
            final_card = {
                "speaker": "",
                "quote_type": "unclear",
                "evidence_basis": "unknown",
                "confidence": "low",
                "citations": [],
                "reason": "final judge panel did not reach unanimous agreement",
                "valid": False,
            }
        final_card["panel_support"] = final_support
        final_card["panel_size"] = len(final_cards)
        final_card["panel_speakers"] = [
            card.get("speaker") for card in valid_final_cards if card.get("speaker")
        ]

        decision = self._select_decision(
            baseline,
            baseline_evidence or {},
            quote_cards,
            attribution_cards,
            final_card,
            packet,
        )
        previous_neighbor_cards = []
        next_neighbor_cards = []
        baseline_clean = validate_char_name(baseline) or baseline
        plurality = decision.get("attribution_plurality") or ""
        local_target_index = next(
            index for index, turn in enumerate(local_packet["turns"])
            if turn.get("is_target")
        )
        if (
            decision.get("selection_source") == "baseline-retained"
            and decision.get("attribution_plurality_support", 0) >= 3
            and is_valid_final_speaker(baseline_clean)
            and is_valid_final_speaker(plurality)
            and baseline_clean != NON_PERSON_LABEL
            and plurality != NON_PERSON_LABEL
            and not self._same_speaker(baseline_clean, plurality)
            and 0 < local_target_index < len(local_packet["turns"]) - 1
        ):
            neighbor_turns = {
                "Previous": local_packet["turns"][local_target_index - 1],
                "Next": local_packet["turns"][local_target_index + 1],
            }
            for side, turn in neighbor_turns.items():
                neighbor_system = (
                    f"Identify only the {side.upper()} neighboring dialogue turn {turn['id']} at "
                    f"L{turn['line']}: {turn['text']}. Do not identify or discuss the TARGET speaker. "
                    "Use that neighbor turn's own immediate narrative attribution and actions. Do not use topic, "
                    "persona, species, occupation, or presumed pronoun habits. "
                    + self._speaker_output_format()
                )
                destination = previous_neighbor_cards if side == "Previous" else next_neighbor_cards
                for index in range(1, 4):
                    card, pec, ec = self._run_xml_agent(
                        f"{side}NeighborAudit{index}",
                        "single_neighbor_anchor_audit",
                        neighbor_system,
                        local_packet_text,
                        round_log,
                        local_packet,
                        "speaker",
                    )
                    destination.append(card)
                    total_pec += pec
                    total_ec += ec
            decision = self._apply_split_neighbor_result(
                decision,
                baseline_clean,
                plurality,
                previous_neighbor_cards,
                next_neighbor_cards,
            )
        metadata = {
            "decision": decision,
            "packet": {
                "raw_start": packet["raw_start"],
                "raw_end": packet["raw_end"],
                "target_dialogue_index": target["dialogue_index"],
                "turn_count": len(packet["turns"]),
            },
            "quote_reviews": [
                {key: value for key, value in card.items() if key != "raw"}
                for card in quote_cards
            ],
            "attribution_reviews": [
                {key: value for key, value in card.items() if key != "raw"}
                for card in attribution_cards
            ],
            "final_reviews": [
                {key: value for key, value in card.items() if key != "raw"}
                for card in final_cards
            ],
            "final_review": {key: value for key, value in final_card.items() if key != "raw"},
            "previous_neighbor_reviews": [
                {key: value for key, value in card.items() if key != "raw"}
                for card in previous_neighbor_cards
            ],
            "next_neighbor_reviews": [
                {key: value for key, value in card.items() if key != "raw"}
                for card in next_neighbor_cards
            ],
            "model_calls": self.model_call_count - calls_before,
        }
        reason = (
            f"quality_audit={decision['selection_source']}; baseline={decision['baseline_speaker']}; "
            f"quote_votes={decision['quote_votes']}; scene_consensus={decision['attribution_consensus'] or 'none'}; "
            f"final={decision['final_speaker'] or 'invalid'}; "
            f"neighbors={decision.get('previous_neighbor_speaker') or 'none'}/"
            f"{decision.get('next_neighbor_speaker') or 'none'}"
        )
        summary = f"{decision['speaker']} | {dialogue[:80]}"
        return decision["speaker"], summary, reason, total_pec, total_ec, metadata


# ============================================================
# Quality Ensemble - blind, block-level attribution
# ============================================================

class QualityEnsemble:
    """Run independent raw-text reviews before selecting a speaker.

    It deliberately accepts no provisional speaker history. Every agent receives
    the same bounded, line-numbered raw packet and makes a novel-agnostic claim.
    """

    def __init__(self, dialogue_radius=None):
        radius = DIALOGUE_BLOCK_RADIUS if dialogue_radius is None else dialogue_radius
        self.dialogue_radius = max(1, int(radius))
        self.model_call_count = 0

    def _run_agent(self, agent_name, role, system_prompt, user_content, round_log, required_tags=()):
        if required_tags:
            user_content += (
                "\n\nFORMAT REQUIREMENT: Call submit_review with the required structured fields. "
                "Do not write analysis before the tool call. Required fields: "
                + ", ".join(required_tags)
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        output_tool = self._quality_output_tool(required_tags)
        model_text, pec, ec, tool_calls = call_ollama(
            messages,
            tools=[output_tool],
            label=agent_name,
            request_timeout=QUALITY_REQUEST_TIMEOUT,
            tool_choice={"type": "function", "function": {"name": "submit_review"}},
        )
        self.model_call_count += 1
        tool_args = self._extract_quality_tool_args(tool_calls)
        log_agent(
            round_log,
            agent_name,
            role,
            messages,
            model_text,
            pec,
            ec,
            tool_calls_list=[{"function": "submit_review", "arguments": tool_args}] if tool_args else None,
        )
        if tool_args:
            return self._tool_args_to_tag_text(tool_args), pec, ec

        if required_tags:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict structured-output formatter. Extract one final conclusion from "
                        "the analyst response below. Call submit_review only. Do not invent evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Required fields: {', '.join(required_tags)}\n\n"
                        f"Analyst response:\n{compact_text(model_text, 9000, keep='tail')}"
                    ),
                },
            ]
            repaired_text, repair_pec, repair_ec, repair_tool_calls = call_ollama(
                repair_messages,
                tools=[output_tool],
                label=f"{agent_name}FormatRepair",
                request_timeout=QUALITY_REQUEST_TIMEOUT,
                tool_choice={"type": "function", "function": {"name": "submit_review"}},
            )
            self.model_call_count += 1
            repaired_args = self._extract_quality_tool_args(repair_tool_calls)
            log_agent(
                round_log,
                f"{agent_name}FormatRepair",
                "structured_output_repair",
                repair_messages,
                repaired_text,
                repair_pec,
                repair_ec,
                tool_calls_list=[{"function": "submit_review", "arguments": repaired_args}] if repaired_args else None,
            )
            if repaired_args:
                return self._tool_args_to_tag_text(repaired_args), pec + repair_pec, ec + repair_ec
            return repaired_text, pec + repair_pec, ec + repair_ec
        return model_text, pec, ec

    @staticmethod
    def _quality_output_tool(required_tags):
        tool = json.loads(json.dumps(QUALITY_OUTPUT_TOOL))
        tool["function"]["parameters"]["required"] = list(required_tags)
        return tool

    @staticmethod
    def _extract_quality_tool_args(tool_calls):
        for tool_call in tool_calls or []:
            function = tool_call.get("function", {}) or {}
            if function.get("name") != "submit_review":
                continue
            args, parse_error = parse_tool_arguments("submit_review", function.get("arguments", "{}"))
            if not parse_error or args:
                return args
        return {}

    @staticmethod
    def _tool_args_to_tag_text(args):
        fields = (
            "speaker", "target_speaker", "preferred_speaker", "block_assignments",
            "quote_type", "voice_kind", "evidence_basis", "confidence", "reason",
        )
        rendered = []
        for field in fields:
            value = args.get(field)
            if value is not None and str(value).strip():
                rendered.append(f"<{field}>{value}</{field}>")
        citations = args.get("citations") or []
        if isinstance(citations, (str, int)):
            citations = [citations]
        citation_text = ", ".join(f"L{item}" for item in citations)
        if citation_text:
            rendered.append(f"<citations>{citation_text}</citations>")
        return "\n".join(rendered)

    def _parse_card(self, text, speaker_tags=("speaker",)):
        raw_speaker = ""
        for tag in speaker_tags:
            raw_speaker = get_ensemble_tag(text, tag)
            if raw_speaker:
                break
        speaker = validate_char_name(raw_speaker) if raw_speaker else ""
        if not speaker and raw_speaker.strip() == "?":
            speaker = "?"
        citations_text = get_ensemble_tag(text, "citations")
        return {
            "speaker": speaker,
            "quote_type": get_ensemble_tag(text, "quote_type", "unclear").strip().lower(),
            "voice_kind": get_ensemble_tag(text, "voice_kind", "unclear").strip().lower(),
            "evidence_basis": get_ensemble_tag(text, "evidence_basis", "unknown").strip().lower(),
            "confidence": get_ensemble_tag(text, "confidence", "low").strip().lower(),
            "citations": parse_line_citations(citations_text),
            "reason": get_ensemble_tag(text, "reason")[:500],
            "raw": text,
        }

    @staticmethod
    def _same_speaker(left, right):
        return bool(left and right and (left == right or left in right or right in left))

    @staticmethod
    def _confidence_is_high(value):
        return (value or "").strip().lower() in {"high", "certain"}

    def _card_has_local_citation(self, card, packet):
        return citations_are_local(card.get("citations", []), packet)

    def _select_decision(self, quote_card, local_card, block_card, arbiter_card, packet):
        local_speaker = local_card.get("speaker", "")
        block_speaker = block_card.get("speaker", "")
        consensus = local_speaker if self._same_speaker(local_speaker, block_speaker) else ""
        consensus_cited = bool(
            consensus
            and self._card_has_local_citation(local_card, packet)
            and self._card_has_local_citation(block_card, packet)
        )
        arbiter_speaker = arbiter_card.get("speaker", "")
        arbiter_cited = self._card_has_local_citation(arbiter_card, packet)
        arbiter_strong = (
            arbiter_cited
            and self._confidence_is_high(arbiter_card.get("confidence"))
            and arbiter_card.get("evidence_basis") in {"explicit_attribution", "quote_type"}
        )

        source = "arbiter"
        if consensus and not arbiter_speaker:
            speaker = consensus
            source = "blind-consensus-no-arbiter-answer"
        elif consensus and not self._same_speaker(consensus, arbiter_speaker) and not arbiter_strong:
            speaker = consensus
            source = "blind-consensus-over-weak-arbiter"
        elif arbiter_speaker:
            speaker = arbiter_speaker
        elif consensus:
            speaker = consensus
            source = "blind-consensus"
        elif local_speaker:
            speaker = local_speaker
            source = "local-fallback"
        elif block_speaker:
            speaker = block_speaker
            source = "block-fallback"
        elif quote_card.get("voice_kind") == "non_person":
            speaker = NON_PERSON_LABEL
            source = "quote-type-fallback"
        else:
            speaker = "?"
            source = "unresolved"

        if speaker == NON_PERSON_LABEL and quote_card.get("voice_kind") not in {"non_person", "unclear"}:
            if consensus and consensus != NON_PERSON_LABEL:
                speaker = consensus
                source = "blind-consensus-over-unverified-non-person"

        final_basis = arbiter_card.get("evidence_basis", "unknown")
        final_confidence = arbiter_card.get("confidence", "low")
        final_quote_type = arbiter_card.get("quote_type", "unclear")
        if source.startswith("blind-consensus"):
            final_basis = "blind_consensus"
            final_confidence = "high" if consensus_cited else "medium"
            final_quote_type = quote_card.get("quote_type") or block_card.get("quote_type") or "unclear"

        confirmed_anchor = bool(
            arbiter_cited
            and self._confidence_is_high(final_confidence)
            and final_basis in {"explicit_attribution", "quote_type"}
        )
        if consensus_cited and self._same_speaker(speaker, consensus):
            confirmed_anchor = True

        citations = arbiter_card.get("citations", [])
        if not citations and consensus_cited:
            citations = sorted(set(local_card.get("citations", []) + block_card.get("citations", [])))
        return {
            "speaker": speaker,
            "quote_type": final_quote_type,
            "evidence_basis": final_basis,
            "confidence": final_confidence,
            "citations": citations,
            "confirmed_anchor": confirmed_anchor,
            "selection_source": source,
            "blind_agreement": bool(consensus),
        }

    def decide(self, novel_lines, dialogue_list, line_num, dialogue, confirmed_anchors, round_log):
        """Return a quality-mode decision without consuming provisional labels."""
        calls_before = self.model_call_count
        packet = build_dialogue_packet(
            novel_lines,
            dialogue_list,
            line_num,
            dialogue,
            dialogue_radius=self.dialogue_radius,
        )
        packet_text = format_dialogue_packet(packet)
        anchor_text = compact_text(confirmed_anchors, 2500, keep="tail")

        quote_system = (
            "You classify the TARGET quoted span in a raw Chinese novel packet. "
            "Do not identify a nearby character. Decide whether it is actual person speech, "
            "group speech, narration/thought, embedded text, or sound/text. Cite only original "
            "packet lines using L<number>. Call submit_review with quote_type, voice_kind, confidence, "
            "citations, and a brief raw-text reason."
        )
        quote_text, quote_pec, quote_ec = self._run_agent(
            "QuoteTypeAgent", "quote_type", quote_system,
            f"Classify only the target quote. Prior annotations are absent.\n\n{packet_text}",
            round_log,
            required_tags=("quote_type", "voice_kind", "confidence", "citations"),
        )
        quote_card = self._parse_card(quote_text)

        local_system = (
            "You are a blind local speaker-attribution reviewer for a Chinese novel. Use only the raw "
            "evidence packet. Identify the TARGET speaker, not a mentioned person and not the speaker "
            "of an adjacent quote. A speech verb is strong only when it grammatically binds to the TARGET. "
            "Check both before-quote and after-quote attribution. Alternation is a last tie-breaker. "
            "An immediate narrative gesture that demonstrates a referent in the target quote can support the "
            "preceding quote; a later generic reaction cannot. "
            "Do not use outside knowledge, character stereotypes, name meanings, or the topic of a line as evidence. "
            "If the neighboring turns have raw-text anchors, test whether the target is the intervening reply instead "
            "of copying a nearby speaker. "
            "Do not use prior labels. Call submit_review with speaker, quote_type, evidence_basis, confidence, "
            "citations, and a brief target-bound reason."
        )
        local_text, local_pec, local_ec = self._run_agent(
            "LocalAttributor", "blind_local_attribution", local_system,
            f"Find the TARGET speaker independently.\n\n{packet_text}\n\n[Confirmed anchors only]\n{anchor_text}",
            round_log,
            required_tags=("speaker", "quote_type", "evidence_basis", "confidence", "citations"),
        )
        local_card = self._parse_card(local_text)

        block_system = (
            "You are a blind dialogue-block solver for a Chinese novel. Read the full local block and "
            "assign speakers jointly before deciding the TARGET. Use raw narrative anchors on either side. "
            "Treat an immediate demonstrative action after a quote as possible attribution for that quote, "
            "not as evidence for an earlier adjacent turn. "
            "Narrative can reset alternation and one character may speak consecutively. Do not inherit "
            "outside character knowledge or select a speaker only because the topic sounds characteristic. First find "
            "which turns are locally anchored; when the turns on both sides are anchored to one speaker, explicitly "
            "test the target as the intervening reply. "
            "previous labels or invent names. Call submit_review with block_assignments, target_speaker, quote_type, "
            "evidence_basis, confidence, citations, and a brief target-bound reason."
        )
        block_text, block_pec, block_ec = self._run_agent(
            "BlockSolver", "blind_block_attribution", block_system,
            f"Solve the dialogue block jointly. The target is marked.\n\n{packet_text}\n\n[Confirmed anchors only]\n{anchor_text}",
            round_log,
            required_tags=("block_assignments", "target_speaker", "quote_type", "evidence_basis", "confidence", "citations"),
        )
        block_card = self._parse_card(block_text, speaker_tags=("target_speaker", "speaker"))
        block_card["assignments"] = get_ensemble_tag(block_text, "block_assignments")[:1000]

        candidate_cards = {
            "quote_type_review": {
                key: quote_card[key]
                for key in ("quote_type", "voice_kind", "confidence", "citations", "reason")
            },
            "local_attribution_review": {
                key: local_card[key]
                for key in ("speaker", "quote_type", "evidence_basis", "confidence", "citations", "reason")
            },
            "block_attribution_review": {
                key: block_card[key]
                for key in ("speaker", "quote_type", "evidence_basis", "confidence", "citations", "reason", "assignments")
            },
        }
        counterfactual_system = (
            "You are an adversarial evidence reviewer for dialogue attribution. The raw packet is authoritative. "
            "Inspect the anonymous candidate claims and try to falsify them: an attribution verb may belong to "
            "a neighboring quote, a named person may be only the addressee, or the target may not be actual speech. "
            "Choose the most defensible target speaker only after checking the exact raw lines. Do not apply any "
            "outside character knowledge or decide from topic/persona alone. Challenge a shared candidate conclusion "
            "when the local turn sequence and post-quote narrative point elsewhere. "
            "novel-specific rule. Call submit_review with preferred_speaker, evidence_basis, confidence, citations, "
            "and a reason stating which candidate evidence survives or fails."
        )
        counterfactual_text, counterfactual_pec, counterfactual_ec = self._run_agent(
            "CounterfactualReviewer", "candidate_refutation", counterfactual_system,
            f"Challenge the candidate claims against the raw target evidence.\n\n{packet_text}\n\n"
            f"[Anonymous candidate cards]\n{json.dumps(candidate_cards, ensure_ascii=False, indent=2)}",
            round_log,
            required_tags=("preferred_speaker", "evidence_basis", "confidence", "citations"),
        )
        counterfactual_card = self._parse_card(
            counterfactual_text,
            speaker_tags=("preferred_speaker", "speaker"),
        )

        anonymous_cards = {
            **candidate_cards,
            "counterfactual_review": {
                key: counterfactual_card[key]
                for key in ("speaker", "evidence_basis", "confidence", "citations", "reason")
            },
        }
        arbiter_system = (
            "You are a citation-based adjudicator for dialogue speaker annotation. The raw packet is "
            "authoritative; anonymous reviews are hypotheses. Reject an attribution if it belongs to a "
            "different quote. Prefer direct quote-bound evidence and use dialogue flow only as a tie-breaker. "
            "Consider an immediate demonstrative action after a quote when it clearly continues that quote's referent. "
            "Do not rely on outside character knowledge, a name's presumed speech style, or the semantic topic alone. "
            "When both neighboring turns have local anchors, evaluate the target as a distinct intervening turn. "
            "Do not use novel-specific rules. Call submit_review with speaker, quote_type, evidence_basis, confidence, "
            "citations, and a brief target-bound reason."
        )
        arbiter_text, arbiter_pec, arbiter_ec = self._run_agent(
            "CitationArbiter", "citation_arbitration", arbiter_system,
            f"Adjudicate the TARGET speaker from raw text. Do not trust unsupported claims.\n\n{packet_text}\n\n"
            f"[Anonymous review cards]\n{json.dumps(anonymous_cards, ensure_ascii=False, indent=2)}",
            round_log,
            required_tags=("speaker", "quote_type", "evidence_basis", "confidence", "citations"),
        )
        arbiter_card = self._parse_card(arbiter_text)

        decision = self._select_decision(quote_card, local_card, block_card, arbiter_card, packet)
        reason = (
            f"quality_ensemble={decision['selection_source']}; "
            f"citations={','.join(f'L{line}' for line in decision['citations']) or 'none'}; "
            f"arbiter={arbiter_card.get('reason') or 'no reason'}"
        )[:800]
        metadata = {
            "decision": decision,
            "packet": {
                "raw_start": packet["raw_start"],
                "raw_end": packet["raw_end"],
                "target_dialogue_index": packet["target"]["dialogue_index"],
                "turn_count": len(packet["turns"]),
            },
            "reviews": {
                "quote_type": {key: value for key, value in quote_card.items() if key != "raw"},
                "local": {key: value for key, value in local_card.items() if key != "raw"},
                "block": {key: value for key, value in block_card.items() if key != "raw"},
                "counterfactual": {key: value for key, value in counterfactual_card.items() if key != "raw"},
                "arbiter": {key: value for key, value in arbiter_card.items() if key != "raw"},
            },
            "model_calls": self.model_call_count - calls_before,
        }
        total_pec = quote_pec + local_pec + block_pec + counterfactual_pec + arbiter_pec
        total_ec = quote_ec + local_ec + block_ec + counterfactual_ec + arbiter_ec
        summary = f"{decision['speaker']} | {dialogue[:80]}"
        return decision["speaker"], summary, reason, total_pec, total_ec, metadata


# ============================================================
# Boss (Python orchestrator)
# ============================================================

class Boss:
    def __init__(self, short_mem_rounds=20, decision_mode=None):
        self.decision_mode = (decision_mode or DECISION_MODE or "quality").strip().lower()
        if self.decision_mode not in {"quality", "ensemble", "legacy"}:
            raise ValueError(f"Unsupported decision mode: {self.decision_mode}")
        self.labeler = LabelerAgent()
        self.verifier = VerifierAgent()
        self.short_mem = ShortMemAgent(max_rounds=short_mem_rounds)
        self.dialogue_list = get_dialogue_list()
        with open(NOVEL_PATH, "r", encoding="utf-8") as f:
            self.novel_lines = f.readlines()
        self.ensemble = QualityEnsemble()
        self.quality_audit = QualityAudit()
        self.total_tokens = 0
        self.round_count = 0
        self.search_agent_triggers = 0
        self.corrections = 0
        # Import EvidenceVault and SearchAgent
        from evidence_vault import EvidenceVault
        from search_agent import SearchAgent
        self.vault = EvidenceVault(VAULT_PATH)
        self.search_agent = SearchAgent(
            call_ollama_fn=call_ollama,
            read_novel_fn=read_novel_lines,
            deep_search_fn=deep_search_identity,
            find_refs_fn=find_all_references
        )

    def _canonicalize_verified_alias(self, speaker):
        cleaned = validate_char_name(speaker)
        if not cleaned or cleaned in {NON_PERSON_LABEL, "?"}:
            return speaker, ""
        canonical, status = self.vault.alias_status(cleaned)
        if canonical != cleaned and status == "verified":
            return canonical, f"verified alias {cleaned} -> {canonical}"
        return cleaned, ""

    def _find_dialogue_index(self, line_num, dialogue):
        """Find the target dialogue in the extracted dialogue list."""
        for idx, (ln, text) in enumerate(self.dialogue_list):
            if ln == line_num and text == dialogue:
                return idx
        for idx, (ln, text) in enumerate(self.dialogue_list):
            if ln == line_num:
                return idx
        return None

    def _local_dialogue_structure(self, line_num, dialogue):
        """Analyze local dialogue structure without using any answer labels."""
        idx = self._find_dialogue_index(line_num, dialogue)
        prev_dialogue = self.dialogue_list[idx - 1] if idx is not None and idx > 0 else None
        next_dialogue = (
            self.dialogue_list[idx + 1]
            if idx is not None and idx + 1 < len(self.dialogue_list)
            else None
        )

        with open(NOVEL_PATH, "r", encoding="utf-8") as f:
            novel_lines = [line.rstrip("\n") for line in f.readlines()]

        narrative_between = []
        if prev_dialogue:
            prev_line = prev_dialogue[0]
            for ln in range(prev_line + 1, line_num):
                if 1 <= ln <= len(novel_lines):
                    text = novel_lines[ln - 1].strip()
                    if text and "「" not in text:
                        narrative_between.append((ln, text[:80]))

        local_start = max(1, line_num - 3)
        local_end = min(len(novel_lines), line_num + 2)
        local_text = "\n".join(novel_lines[local_start - 1:local_end])
        has_attribution = bool(SPEECH_ATTRIBUTION_RE.search(local_text))

        return {
            "index": idx,
            "previous": prev_dialogue,
            "next": next_dialogue,
            "narrative_between": narrative_between,
            "no_narrative_break_from_previous": bool(prev_dialogue and not narrative_between),
            "has_nearby_attribution_word": has_attribution,
            "local_start": local_start,
            "local_end": local_end,
        }

    def _format_structure_hint(self, structure):
        lines = ["[Local dialogue structure]"]
        prev_dialogue = structure.get("previous")
        next_dialogue = structure.get("next")
        if prev_dialogue:
            lines.append(f"  Previous dialogue: L{prev_dialogue[0]}「{prev_dialogue[1][:40]}」")
        else:
            lines.append("  Previous dialogue: none")
        if next_dialogue:
            lines.append(f"  Next dialogue: L{next_dialogue[0]}「{next_dialogue[1][:40]}」")
        if structure.get("no_narrative_break_from_previous"):
            lines.append("  No non-dialogue narrative line between previous dialogue and target.")
            lines.append("  If this is a two-person exchange and no explicit attribution exists, consider alternation.")
        elif structure.get("narrative_between"):
            examples = "; ".join(
                f"L{ln}: {text}" for ln, text in structure["narrative_between"][:3]
            )
            lines.append(f"  Narrative break before target: {examples}")
            lines.append("  A narrative break resets simple alternation unless it explicitly attributes speech.")
        lines.append(
            "  Nearby attribution words: "
            + ("present" if structure.get("has_nearby_attribution_word") else "not obvious")
        )
        return "\n".join(lines)

    def _build_navigation(self, line_num, nav_range=25):
        """Build original text navigation around target line - NO speaker labels."""
        lines = []
        start = max(1, line_num - nav_range)
        end = line_num + nav_range
        novel_lines = read_novel_lines(start, end - start + 1).split("\n")

        for line_text in novel_lines:
            if not line_text.strip():
                continue
            parts = line_text.split(":", 1)
            if len(parts) < 2:
                continue
            try:
                ln = int(parts[0].strip())
            except ValueError:
                continue
            content = parts[1].strip()

            marker = " "
            if ln == line_num:
                marker = ">>"

            if "\u300c" in content:
                display = content[:120]
                lines.append(f"  {marker}L{ln}: \u300c{display}\u300d")
            else:
                display = content[:120]
                lines.append(f"  {marker}L{ln}: [narr] {display}")

        return "\n".join(lines)

    def _safe_print(self, msg):
        """Print to stderr, resilient on Windows even after Ollama API calls that may corrupt stdout."""
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except (ValueError, OSError):
            pass

    def _is_short_or_fragment(self, dialogue):
        text = (dialogue or "").strip()
        han_count = len(re.findall(r"[一-鿿]", text))
        return len(text) <= 12 or han_count <= 4

    def _label_matches_local_candidates(self, label, local_evidence):
        cleaned = validate_char_name(label) or (label or "").strip()
        if not cleaned or cleaned == NON_PERSON_LABEL:
            return False
        candidates = (local_evidence or {}).get("attribution_candidates") or []
        named_candidates = []
        for item in candidates:
            candidate = validate_char_name(item.get("speaker")) or item.get("speaker", "")
            if candidate and not is_generic_speaker_label(candidate):
                named_candidates.append(candidate)
        if not named_candidates:
            return True
        return any(cleaned == cand or cleaned in cand or cand in cleaned for cand in named_candidates)

    def _risk_reasons(self, speaker, dialogue, tool_rounds_used, next_expected,
                      local_structure=None, local_evidence=None, labeler_evidence=None):
        """Return generic risk reasons that should trigger independent review."""
        reasons = []
        normalized = (speaker or "").strip()
        cleaned = validate_char_name(normalized) or normalized
        labeler_evidence = labeler_evidence or {}
        basis = (labeler_evidence.get("evidence_basis") or "unknown").strip().lower()
        confidence = (labeler_evidence.get("confidence") or "low").strip().lower()
        quote_type = (labeler_evidence.get("quote_type") or "unclear").strip().lower()
        short_fragment = self._is_short_or_fragment(dialogue)
        phase_conflict = bool(next_expected and cleaned and cleaned != next_expected)
        no_narrative_break = bool(
            local_structure and local_structure.get("no_narrative_break_from_previous")
        )
        local_candidates = (local_evidence or {}).get("attribution_candidates") or []
        quote_cautions = (local_evidence or {}).get("quote_type_cautions") or []

        if cleaned == "?":
            reasons.append("speaker unresolved")
        if is_generic_speaker_label(cleaned):
            reasons.append("speaker is a generic descriptor and may need canonical named identity")
        if validate_char_name(cleaned) == NON_PERSON_LABEL:
            reasons.append("speaker is non-person; quote type should be verified")
        if quote_cautions and self._is_concrete_speaker(cleaned):
            reasons.append("local quote-type cautions may mean this is not ordinary direct speech")
        if local_candidates and not self._label_matches_local_candidates(cleaned, local_evidence):
            reasons.append("nearby explicit attribution candidate does not match current speaker")
        if basis in {"alternation", "inference", "unknown"}:
            if short_fragment or no_narrative_break or local_candidates:
                reasons.append(f"labeler relied on weak evidence_basis={basis}")
        if confidence == "low":
            reasons.append("labeler confidence is low")
        if quote_type in {"thought_or_narration", "sound_or_text"} and self._is_concrete_speaker(cleaned):
            reasons.append(f"labeler quote_type={quote_type} conflicts with concrete speaker")
        if quote_type in {"direct_speech", "group_speech"} and validate_char_name(cleaned) == NON_PERSON_LABEL:
            reasons.append(f"labeler quote_type={quote_type} conflicts with non-person speaker")
        if phase_conflict:
            if no_narrative_break:
                reasons.append("speaker conflicts with adjacent two-person exchange without narrative break")
            elif short_fragment or tool_rounds_used <= 1:
                reasons.append("speaker conflicts with recent two-person exchange expectation")
        if no_narrative_break and self.short_mem.detect_rapid_exchange(3) and short_fragment:
            reasons.append("short line follows adjacent dialogue without narrative break")
        if local_structure and local_structure.get("has_nearby_attribution_word"):
            reasons.append("nearby speech attribution words should be checked")
        if self.short_mem.detect_rapid_exchange(4) and short_fragment:
            reasons.append("short or fragmentary line inside a rapid exchange")
        if reasons and tool_rounds_used <= 1 and short_fragment:
            reasons.append("low evidence depth for a short or fragmentary line")

        return list(dict.fromkeys(reasons))

    def _is_concrete_speaker(self, speaker):
        cleaned = validate_char_name(speaker)
        return bool(cleaned and cleaned not in (NON_PERSON_LABEL, "?") and not is_generic_speaker_label(cleaned))

    def _should_apply_verifier_change(self, current_speaker, suggested, verifier_reason,
                                      evidence_basis="unknown", confidence="low", local_evidence=None):
        """Conservatively decide whether Verifier may override the primary label."""
        current = validate_char_name(current_speaker) or current_speaker
        proposed = validate_char_name(suggested)
        if not proposed or proposed == current:
            return False

        current_is_concrete = self._is_concrete_speaker(current)
        current_is_generic = is_generic_speaker_label(current)
        proposed_is_generic = is_generic_speaker_label(proposed)

        # Never replace a concrete named speaker with a vague role/appearance label.
        if current_is_concrete and proposed_is_generic:
            return False

        basis = (evidence_basis or "unknown").strip().lower()
        conf = (confidence or "low").strip().lower()
        if conf not in {"high", "certain"}:
            return False

        reason = (verifier_reason or "").lower()
        nonperson_strong_phrases = (
            "not actual speech", "not direct speech", "narrative text",
            "narrative description", "title", "term", "sound effect",
            "nobody is speaking", "no character is speaking",
            "不是实际发言", "不是直接发言", "叙述文字", "叙述内容",
            "标题", "术语", "音效", "无人说出", "没有角色说出",
        )
        if proposed == NON_PERSON_LABEL and current_is_concrete:
            if basis != "quote_type" or not any(p in reason for p in nonperson_strong_phrases):
                return False

        # Alternation is allowed to trigger review, but it is no longer enough to rewrite labels.
        if basis in {"alternation", "dialogue_structure", "dialogue structure"}:
            return False

        if current == NON_PERSON_LABEL and proposed != NON_PERSON_LABEL and basis != "explicit_attribution":
            return False
        if basis == "quote_type" and proposed != NON_PERSON_LABEL:
            return False

        if basis == "identity_alias":
            return current_is_generic and not proposed_is_generic

        strong_bases = {"explicit_attribution", "quote_type"}
        if basis not in strong_bases:
            return False

        weak_phrases = (
            "alternating", "dialogue structure", "dialogue flow", "natural response",
            "交替", "对话结构", "对话逻辑", "自然回应",
        )
        strong_phrases = (
            "explicit", "directly attributes", "speech verb", "states", "says",
            "明确", "直接", "说：", "说道", "问道", "回答", "开口", "喊道", "叫道",
        )
        if any(p in reason for p in weak_phrases) and not any(p in reason for p in strong_phrases):
            return False
        if basis == "explicit_attribution" and not any(p in reason for p in strong_phrases):
            return False
        if basis == "explicit_attribution" and not self._label_matches_local_candidates(proposed, local_evidence):
            return False

        narrative_non_person_phrases = (
            "not direct speech", "narrative text", "narrative description",
            "不是直接发言", "叙述文字", "叙述内容",
        )
        if (
            current_is_concrete
            and proposed != NON_PERSON_LABEL
            and any(p in reason for p in narrative_non_person_phrases)
        ):
            return False

        return True

    def process_one(self, line_num, dialogue, round_log, quiet=False):
        self.round_count += 1

        if not quiet:
            self._safe_print(f"\n{'='*60}")
            self._safe_print(f"  Round {self.round_count}")
            self._safe_print(f"  Dialogue: L{line_num}\u300c{dialogue}\u300d")
            self._safe_print(f"{'='*60}")

        # 1. Build context: Novel Map (structure overview) + navigation (full text)
        context_text = build_context_index(line_num, 40)
        navigation_text = self._build_navigation(line_num, nav_range=25)
        local_structure = self._local_dialogue_structure(line_num, dialogue)
        structure_hint = self._format_structure_hint(local_structure)
        local_evidence = analyze_local_evidence(line_num, dialogue, local_structure)
        local_evidence_hint = format_local_evidence_hint(local_evidence)

        # 2. Get evidence and memory
        evidence_text = self.vault.get_state_text(current_line=line_num)
        short_mem_text = self.short_mem.get_summary()
        recent_speakers_hint = self.short_mem.get_recent_speakers_hint()

        # 3. Detect rapid exchange for extra warning
        if self.short_mem.detect_rapid_exchange(4):
            short_mem_text += ("\n\nRAPID EXCHANGE: Last 4+ dialogues alternate between two speakers.\n"
                               "RULE: Narrative attribution (#1) takes priority over alternation (#3).\n"
                               "Read 3-5 lines BEFORE the target to check for speech verbs.")

        # 4. Log boss task
        round_log["boss_task"] = {
            "dialogue_line": line_num,
            "dialogue_text": dialogue,
            "short_mem": short_mem_text,
            "recent_speakers_hint": recent_speakers_hint,
            "evidence": evidence_text,
            "navigation": navigation_text,
            "local_structure": local_structure,
            "local_evidence": local_evidence,
        }
        temp_log_event(
            "round_context_ready",
            round_log,
            navigation_len=len(navigation_text or ""),
            short_mem_len=len(short_mem_text or ""),
            evidence_len=len(evidence_text or ""),
            context_index_len=len(context_text or ""),
            local_evidence_len=len(local_evidence_hint or ""),
        )

        # 5. Decide from raw text. Quality mode keeps blind reviewers separate
        # from provisional memory; legacy mode remains available for comparison.
        ensemble_metadata = None
        audit_metadata = None
        baseline_speaker = ""
        confirmed_anchor = False
        if self.decision_mode == "quality":
            baseline_speaker, baseline_summary, baseline_reason, base_pec, base_ec, tool_rounds_used = self.labeler.label(
                line_num, dialogue, short_mem_text, "",
                evidence_text, navigation_text, "",
                round_log, quiet=quiet, override_force_tool=True,
                recent_speakers_hint=recent_speakers_hint,
                structure_hint=structure_hint,
                local_evidence_hint=local_evidence_hint,
            )
            baseline_evidence = dict(round_log.get("labeler_evidence", {}))
            speaker, summary, audit_reason, audit_pec, audit_ec, audit_metadata = self.quality_audit.decide(
                self.novel_lines,
                self.dialogue_list,
                line_num,
                dialogue,
                baseline_speaker,
                baseline_evidence,
                round_log,
            )
            pec = base_pec + audit_pec
            ec = base_ec + audit_ec
            reason_text = (
                f"baseline={baseline_speaker}: {baseline_reason} | {audit_reason}"
            )[:1000]
            decision = audit_metadata["decision"]
            confirmed_anchor = bool(decision.get("confirmed_anchor"))
            round_log["quality_audit"] = audit_metadata
            round_log["labeler_evidence"] = {
                "quote_type": decision.get("quote_type", baseline_evidence.get("quote_type", "unclear")),
                "evidence_basis": decision.get("evidence_basis", "unknown"),
                "confidence": decision.get("confidence", "low"),
            }
            if not quiet:
                self._safe_print(
                    f"  Quality audit: {baseline_speaker} -> {speaker} "
                    f"(source={decision.get('selection_source')}, anchor={confirmed_anchor})"
                )
        elif self.decision_mode == "ensemble":
            speaker, summary, reason_text, pec, ec, ensemble_metadata = self.ensemble.decide(
                self.novel_lines,
                self.dialogue_list,
                line_num,
                dialogue,
                self.short_mem.get_confirmed_summary(),
                round_log,
            )
            decision = ensemble_metadata["decision"]
            confirmed_anchor = bool(decision.get("confirmed_anchor"))
            tool_rounds_used = 0
            round_log["quality_ensemble"] = ensemble_metadata
            round_log["labeler_evidence"] = {
                "quote_type": decision.get("quote_type", "unclear"),
                "evidence_basis": decision.get("evidence_basis", "unknown"),
                "confidence": decision.get("confidence", "low"),
            }
            if not quiet:
                self._safe_print(
                    f"  Quality ensemble: {speaker} "
                    f"(source={decision.get('selection_source')}, anchor={confirmed_anchor})"
                )
        else:
            speaker, summary, reason_text, pec, ec, tool_rounds_used = self.labeler.label(
                line_num, dialogue, short_mem_text, "",
                evidence_text, navigation_text, "",
                round_log, quiet=quiet, override_force_tool=True,
                recent_speakers_hint=recent_speakers_hint,
                structure_hint=structure_hint,
                local_evidence_hint=local_evidence_hint,
            )
            confirmed_anchor = True
            if not quiet:
                self._safe_print(f"  Labeler: {speaker} (tools={tool_rounds_used})")

        if not is_valid_final_speaker(speaker):
            if is_valid_final_speaker(baseline_speaker):
                invalid_speaker = speaker
                speaker = validate_char_name(baseline_speaker)
                reason_text = (
                    f"{reason_text} | Output guard rejected {invalid_speaker!r}; retained baseline {speaker}"
                )[:1000]
                if audit_metadata is not None:
                    audit_metadata["decision"]["output_guard_rejected"] = invalid_speaker
                    audit_metadata["decision"]["speaker"] = speaker
                    audit_metadata["decision"]["selection_source"] = "output-guard-baseline"
            else:
                speaker = "?"
                confirmed_anchor = False
        self.total_tokens += pec + ec
        round_log["annotation_tokens"] = pec + ec

        # 5b. Phase hints are legacy-only. Quality mode never lets an
        # unconfirmed output steer the next raw-text decision.
        corrected = False
        next_exp = self.short_mem.get_next_expected() if self.decision_mode == "legacy" else None
        if self.decision_mode == "legacy" and next_exp and speaker != next_exp:
            self.short_mem.phase_violation = True
            if not quiet:
                self._safe_print(f"  Phase violation: expected {next_exp}, got {speaker}")
        else:
            self.short_mem.phase_violation = False

        # 6. SearchAgent: conditional trigger for temporary descriptors
        if is_generic_speaker_label(speaker):
            self.search_agent_triggers += 1
            if not quiet:
                try:
                    sys.stderr.write(f"  Temporary name: '{speaker}' -> triggering SearchAgent...\n")
                    sys.stderr.flush()
                except (ValueError, OSError):
                    pass

            search_result, s_pec, s_ec, s_tool_log = self.search_agent.investigate(
                speaker, line_num, max_tool_rounds=4, quiet=quiet
            )
            self.total_tokens += s_pec + s_ec

            if search_result["found"] and search_result["character"]:
                character = search_result["character"]
                aliases = search_result.get("aliases", [])
                evidence_list = search_result.get("evidence", [])
                status = search_result.get("status", "candidate")
                intro_line = search_result.get("introduction_line")

                self.vault.add_evidence(character, aliases, evidence_list, status, intro_line)
                if not quiet:
                    self._safe_print(f"  SearchAgent found: '{character}' ({status})")

                if status == "verified":
                    old_speaker = speaker
                    speaker = character
                    corrected = True
                    confirmed_anchor = True
                    self.corrections += 1
                    if not quiet:
                        self._safe_print(f"  Corrected: '{old_speaker}' -> '{character}'")
            else:
                if not quiet:
                    self._safe_print(f"  SearchAgent: no identity found for '{speaker}'")

        canonical_speaker, canonical_reason = self._canonicalize_verified_alias(speaker)
        if canonical_reason:
            old_speaker = speaker
            speaker = canonical_speaker
            corrected = True
            self.corrections += 1
            reason_text = f"{reason_text} | Canonicalized {old_speaker} -> {speaker}: {canonical_reason}".strip()
            if not quiet:
                self._safe_print(f"  Canonicalized: {old_speaker} -> {speaker}")

        # 7. Update EvidenceVault last_seen
        if speaker and speaker != NON_PERSON_LABEL and speaker != "non-human" and not is_generic_speaker_label(speaker):
            self.vault.update_last_seen(speaker, line_num)

        # 8. Fallback
        if not speaker:
            speaker = "?"

        # 8b. Legacy risk review. Quality mode already used independent blind
        # reviews and a citation arbiter, so reintroducing the selected label as
        # a verifier anchor would be counterproductive.
        risk_reasons = []
        if self.decision_mode == "legacy":
            risk_reasons = self._risk_reasons(
                speaker, dialogue, tool_rounds_used, next_exp,
                local_structure=local_structure,
                local_evidence=local_evidence,
                labeler_evidence=round_log.get("labeler_evidence", {}),
            )
        if risk_reasons:
            if not quiet:
                self._safe_print(f"  Risk review: {', '.join(risk_reasons)}")
            temp_log_event("risk_review_start", round_log, speaker=speaker, reasons=risk_reasons)
            try:
                verdict, suggested, verifier_reason, evidence_basis, confidence = self.verifier.verify(
                    line_num, dialogue, navigation_text,
                    f"{short_mem_text}\n\n{recent_speakers_hint}".strip(),
                    "", evidence_text, speaker, round_log, quiet=quiet,
                    risk_reasons=risk_reasons,
                    structure_hint=structure_hint,
                    local_evidence_hint=local_evidence_hint,
                )
            except Exception as exc:
                verdict = "skipped"
                suggested = speaker
                evidence_basis = "unknown"
                confidence = "low"
                verifier_reason = f"RiskVerifier skipped: {str(exc)[:200]}"
                if not quiet:
                    self._safe_print(f"  RiskVerifier skipped: {str(exc)[:200]}")
            round_log["risk_review"] = {
                "reasons": risk_reasons,
                "verdict": verdict,
                "suggested": suggested,
                "reason": verifier_reason,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
            }
            temp_log_event("risk_review_complete", round_log, review=round_log["risk_review"])
            if verdict == "disagree" and self._should_apply_verifier_change(
                speaker, suggested, verifier_reason, evidence_basis, confidence,
                local_evidence=local_evidence,
            ):
                old_speaker = speaker
                speaker = validate_char_name(suggested) or suggested
                corrected = True
                self.corrections += 1
                reason_text = f"{reason_text} | RiskVerifier corrected {old_speaker} -> {speaker}: {verifier_reason}".strip()
                if not quiet:
                    self._safe_print(f"  RiskVerifier corrected: {old_speaker} -> {speaker}")
            elif verdict == "disagree" and not quiet:
                self._safe_print(f"  RiskVerifier change rejected: {speaker} <-/-> {suggested}")

        # 9. Write to labeled.txt
        write_label(speaker)

        # 10. Update ShortMem with narrative context
        narr_before = get_narrative_before(line_num)
        self.short_mem.update(
            line_num,
            dialogue,
            speaker,
            reason_text,
            narrative_before=narr_before,
            confirmed=confirmed_anchor,
        )

        # 11. Save vault periodically
        if self.round_count % 50 == 0:
            self.vault.save()

        round_log["result"] = {
            "speaker": speaker,
            "summary": summary,
            "reason": reason_text,
            "corrected": corrected,
            "decision_mode": self.decision_mode,
            "confirmed_anchor": confirmed_anchor,
        }
        if ensemble_metadata is not None:
            ensemble_metadata["final_speaker"] = speaker
        if audit_metadata is not None:
            audit_metadata["final_speaker"] = speaker

        return speaker


# ============================================================
# Validation
# ============================================================

def _normalize_validation_label(value):
    value = (value or "").strip()
    if not value:
        return ""
    mapped = validate_char_name(value)
    if mapped == NON_PERSON_LABEL:
        return NON_PERSON_LABEL
    value = mapped or value
    value = re.sub(r"^[【\[\(（「『\s]+|[】\]\)）」』\s]+$", "", value)
    return value.strip()


def _is_generic_validation_label(value):
    value = _normalize_validation_label(value)
    if not value:
        return True
    return value == NON_PERSON_LABEL or value in TEMP_DESCRIPTORS


def _validation_lenient_match(acceptable, label_parts):
    normalized_answers = {_normalize_validation_label(part) for part in acceptable}
    normalized_labels = {_normalize_validation_label(part) for part in label_parts}
    normalized_answers.discard("")
    normalized_labels.discard("")
    if normalized_answers & normalized_labels:
        return True, "normalized-exact"
    for answer in normalized_answers:
        for label in normalized_labels:
            if not answer or not label:
                continue
            if _is_generic_validation_label(answer) or _is_generic_validation_label(label):
                continue
            if answer in label or label in answer:
                return True, "name-contained"
    return False, ""


def validate(error_limit=30):
    if not os.path.exists(LABELED_PATH):
        print("labeled.txt not found")
        return 0, 0, 0

    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        labeled = [line.strip() for line in f.readlines() if line.strip()]

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answer_lines = f.readlines()

    answers = []
    answer_line_nums = []
    answer_marker_nums = []
    multi_marker_lines = []
    for i, line in enumerate(answer_lines):
        matches = list(re.finditer(r"【([^】]+)】", line))
        if len(matches) > 1:
            multi_marker_lines.append((i + 1, len(matches)))
        for marker_idx, match in enumerate(matches, start=1):
            answers.append(match.group(1).strip())
            answer_line_nums.append(i + 1)
            answer_marker_nums.append(marker_idx)

    dialogue_count = len(get_dialogue_list()) if os.path.exists(NOVEL_PATH) else None

    total = min(len(labeled), len(answers))
    correct = 0
    lenient_correct = 0
    lenient_extra = 0
    lenient_reasons = Counter()
    wrong = []

    for i in range(total):
        label = labeled[i]
        answer = answers[i]
        acceptable = {part.strip() for part in answer.split("|") if part.strip()}
        label_parts = {part.strip() for part in label.split("|") if part.strip()}
        if acceptable & label_parts:
            correct += 1
            lenient_correct += 1
        else:
            matched, reason = _validation_lenient_match(acceptable, label_parts)
            if matched:
                lenient_correct += 1
                lenient_extra += 1
                lenient_reasons[reason] += 1
            wrong.append({
                "idx": i + 1,
                "answer_line": answer_line_nums[i] if i < len(answer_line_nums) else "?",
                "answer_marker": answer_marker_nums[i] if i < len(answer_marker_nums) else "?",
                "expected": answer,
                "got": label,
                "lenient_match": matched,
                "lenient_reason": reason,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    lenient_accuracy = lenient_correct / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Validation Report")
    print(f"{'='*60}")
    if dialogue_count is not None:
        print(f"  Novel dialogues: {dialogue_count}")
    print(f"  Labeled lines:    {len(labeled)}")
    print(f"  Answer markers:   {len(answers)}")
    print(f"  Total compared:   {total}")
    print(f"  Correct:         {correct}")
    print(f"  Wrong:           {len(wrong)}")
    print(f"  Accuracy:        {accuracy:.1f}%")
    print(f"  Lenient correct: {lenient_correct} (+{lenient_extra})")
    print(f"  Lenient accuracy:{lenient_accuracy:.1f}%")
    if lenient_reasons:
        reason_text = ", ".join(f"{name}={count}" for name, count in lenient_reasons.most_common())
        print(f"  Lenient reasons: {reason_text}")
    if multi_marker_lines:
        examples = ", ".join(
            f"L{line_no}({count})" for line_no, count in multi_marker_lines[:8]
        )
        extra = "" if len(multi_marker_lines) <= 8 else f", +{len(multi_marker_lines) - 8} more"
        print(f"  Multi-marker answer lines parsed: {len(multi_marker_lines)} [{examples}{extra}]")
    if dialogue_count is not None and len(answers) != dialogue_count:
        print(f"  WARNING: answer marker count ({len(answers)}) != novel dialogue count ({dialogue_count})")
    if len(labeled) != len(answers):
        print(f"  WARNING: labeled line count ({len(labeled)}) != answer marker count ({len(answers)})")

    if wrong:
        shown_wrong = wrong if error_limit <= 0 else wrong[:error_limit]
        label = "all" if error_limit <= 0 else f"first {error_limit}"
        print(f"\n  Error details ({label}):")
        for w in shown_wrong:
            marker_suffix = "" if w["answer_marker"] == 1 else f" marker#{w['answer_marker']}"
            extra = ""
            if w["lenient_match"]:
                extra = f" | lenient={w['lenient_reason']}"
            print(f"    #{w['idx']} (answer L{w['answer_line']}{marker_suffix}): expected={w['expected']} | got={w['got']}{extra}")

    return total, correct, len(wrong)


# ============================================================
# Main
# ============================================================

def _writeline(msg):
    """Write status line during annotation."""
    print(msg)


def main():
    global API_MODEL_FILTER, API_MODEL_PRIORITY, MODEL_PROVIDER, API_RETRIES, API_RETRY_DELAY
    global DECISION_MODE, DIALOGUE_BLOCK_RADIUS, QUALITY_REQUEST_TIMEOUT
    global QUALITY_SCENE_RADIUS, QUALITY_AUDIT_RETRIES
    parser = argparse.ArgumentParser(description="Multi-agent novel dialogue speaker annotation v4")
    parser.add_argument("--start", type=int, default=0, help="Start from dialogue index (0=resume)")
    parser.add_argument("--count", type=int, default=1, help="Number of dialogues to annotate")
    parser.add_argument("--short-mem", type=int, default=20, help="Short-term memory rounds")
    parser.add_argument("--decision-mode", choices=["quality", "ensemble", "legacy"], default=DECISION_MODE,
                        help="Annotation path: baseline-preserving quality audit (default), experimental ensemble, or legacy")
    parser.add_argument("--dialogue-block-radius", type=int, default=DIALOGUE_BLOCK_RADIUS,
                        help="Neighboring dialogue turns on each side for quality-mode block review")
    parser.add_argument("--quality-request-timeout", type=int, default=QUALITY_REQUEST_TIMEOUT,
                        help="Seconds before one quality-review API call fails over to another model")
    parser.add_argument("--quality-scene-radius", type=int, default=QUALITY_SCENE_RADIUS,
                        help="Neighboring dialogue turns on each side for baseline-preserving quality audit")
    parser.add_argument("--quality-audit-retries", type=int, default=QUALITY_AUDIT_RETRIES,
                        help="Fresh full-decision attempts after malformed quality-audit output")
    parser.add_argument("--validate", action="store_true", help="Run validation after annotation")
    parser.add_argument("--error-limit", type=int, default=_env_int("NOVEL_VALIDATE_ERROR_LIMIT", 30),
                        help="Validation error rows to print; 0 means all")
    parser.add_argument("--reset-state", action="store_true", help="Reset character state before starting")
    parser.add_argument("--data-dir", default=os.environ.get("NOVEL_DATA_DIR", ""),
                        help="Per-volume data directory containing novel.txt/answers.txt and output state")
    parser.add_argument("--novel", default=os.environ.get("NOVEL_PATH", ""),
                        help="Override novel text path for this run")
    parser.add_argument("--answers", default=os.environ.get("NOVEL_ANSWERS_PATH", ""),
                        help="Override answer file path for validation")
    parser.add_argument("--labeled", default=os.environ.get("NOVEL_LABELED_PATH", ""),
                        help="Override labeled output path")
    parser.add_argument("--log", default=os.environ.get("NOVEL_LOG_PATH", ""),
                        help="Override JSONL log output path")
    parser.add_argument("--state", default=os.environ.get("NOVEL_STATE_PATH", ""),
                        help="Override character_state output path")
    parser.add_argument("--vault", default=os.environ.get("NOVEL_VAULT_PATH", ""),
                        help="Override evidence_vault output path")
    parser.add_argument("--provider", choices=["ollama", "api-fallback"], default=os.environ.get("NOVEL_MODEL_PROVIDER", "ollama"),
                        help="Model provider: ollama or api-fallback")
    parser.add_argument("--api-model", default=os.environ.get("NOVEL_API_MODEL", ""),
                        help="When using api-fallback, restrict calls to one provider/model substring")
    parser.add_argument("--api-priority", default=os.environ.get("NOVEL_API_PRIORITY", ""),
                        help="Comma-separated provider/model names to move to the front of api-fallback order")
    parser.add_argument("--api-round-robin-offset", type=int,
                        default=_env_int("NOVEL_API_ROUND_ROBIN_OFFSET", 0),
                        help="Initial offset for every API round-robin key pool; use different offsets for parallel runs")
    parser.add_argument("--health-check", choices=["all", "first", "none"],
                        default=os.environ.get("NOVEL_API_HEALTH_CHECK", "all"),
                        help="API startup health checks: all models, first priority model only, or none")
    parser.add_argument("--api-retries", type=int, default=_env_int("NOVEL_API_RETRIES", API_RETRIES),
                        help="Attempts per API model for transient errors and missing required tool calls")
    parser.add_argument("--api-retry-delay", type=float, default=_env_float("NOVEL_API_RETRY_DELAY", API_RETRY_DELAY),
                        help="Base seconds for exponential retry delay")
    args = parser.parse_args()
    configure_paths(
        data_dir=args.data_dir,
        novel_path=args.novel,
        answers_path=args.answers,
        labeled_path=args.labeled,
        log_path=args.log,
        state_path=args.state,
        vault_path=args.vault,
    )
    MODEL_PROVIDER = args.provider
    API_MODEL_FILTER = args.api_model.strip()
    API_MODEL_PRIORITY = [item.strip() for item in args.api_priority.split(",") if item.strip()]
    API_RETRIES = max(1, args.api_retries)
    API_RETRY_DELAY = max(0.0, args.api_retry_delay)
    DECISION_MODE = args.decision_mode
    DIALOGUE_BLOCK_RADIUS = max(1, args.dialogue_block_radius)
    QUALITY_REQUEST_TIMEOUT = max(1, args.quality_request_timeout)
    QUALITY_SCENE_RADIUS = max(4, args.quality_scene_radius)
    QUALITY_AUDIT_RETRIES = max(1, args.quality_audit_retries)
    if DECISION_MODE == "quality" and MODEL_PROVIDER == "api-fallback":
        if not API_MODEL_FILTER:
            API_MODEL_FILTER = "sense-nova"
        elif "sense" not in API_MODEL_FILTER.lower():
            parser.error("quality decision mode currently permits SenseNova API models only")

    print("=" * 60)
    print("  Multi-agent Novel Dialogue Speaker Annotation v4")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Novel: {NOVEL_PATH}")
    print(f"  Answers: {ANSWERS_PATH}")
    print(f"  Labeled: {LABELED_PATH}")
    print(f"  Provider: {MODEL_PROVIDER}")
    if MODEL_PROVIDER == "ollama":
        print(f"  Model: {OLLAMA_MODEL}")
        print(f"  Server: {OLLAMA_BASE_URL}")
    else:
        print(f"  API context limit: {API_CONTEXT_LIMIT}")
        print(f"  API max output tokens: {API_MAX_OUTPUT_TOKENS}")
        if API_MODEL_FILTER:
            print(f"  API model filter: {API_MODEL_FILTER}")
        if API_MODEL_PRIORITY:
            print(f"  API model priority: {', '.join(API_MODEL_PRIORITY)}")
        print(f"  API retries: {API_RETRIES} (base delay {API_RETRY_DELAY:g}s)")
        print(f"  API health check: {args.health_check}")
    print(f"  Short-term memory: {args.short_mem} rounds")
    print(f"  Decision mode: {DECISION_MODE}")
    if DECISION_MODE == "quality":
        print(f"  Quality scene radius: {QUALITY_SCENE_RADIUS} turns per side")
        print(f"  Quality audit retries: {QUALITY_AUDIT_RETRIES}")
        print(f"  Quality request timeout: {QUALITY_REQUEST_TIMEOUT}s")
    elif DECISION_MODE == "ensemble":
        print(f"  Dialogue block radius: {DIALOGUE_BLOCK_RADIUS} turns per side")
        print(f"  Quality request timeout: {QUALITY_REQUEST_TIMEOUT}s")
    print(f"  Token budget: {TOKEN_BUDGET}")
    print(f"  Max tool rounds: {MAX_TOOL_ROUNDS}")

    if MODEL_PROVIDER == "api-fallback":
        init_api_fallback(health_check=args.health_check)
        if args.api_round_robin_offset:
            groups = {model.round_robin_group for model in API_MODELS if model.round_robin_group}
            for group in groups:
                API_ROUND_ROBIN_CURSOR[group] = args.api_round_robin_offset
            if groups:
                print(
                    "  API round-robin offset: "
                    f"{args.api_round_robin_offset} for {', '.join(sorted(groups))}"
                )

    # Roster check
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from check_roster import load_roster, scan_py_files
        roster = load_roster(None)
        violations = scan_py_files(SCRIPT_DIR, roster)
        if violations:
            print(f"\nERROR: {len(violations)} novel-specific name leaks found in .py files:")
            for fp, lineno, word, line in violations[:10]:
                rel = os.path.relpath(fp, ROOT_DIR)
                print(f"  {rel}:L{lineno} matched '{word}'")
            if len(violations) > 10:
                print(f"  ... and {len(violations)-10} more")
            print("  Remove hardcoded novel-specific names before running annotation.\n")
            sys.exit(2)
        else:
            print(f"  Roster check: PASSED (no leaks, terms={len(roster)})")
    except ImportError:
        pass

    print("=" * 60)

    if not os.path.exists(NOVEL_PATH):
        print(f"ERROR: novel file not found: {NOVEL_PATH}")
        sys.exit(2)
    if args.validate and not os.path.exists(ANSWERS_PATH):
        print(f"ERROR: answers file not found: {ANSWERS_PATH}")
        sys.exit(2)

    # Reset state if requested
    if args.reset_state:
        print("  Resetting all state...")
        for p in [STATE_PATH, LOG_PATH, TEMP_LOG_PATH, LABELED_PATH, VAULT_PATH]:
            if os.path.exists(p):
                os.remove(p)
        print("  State cleared. Starting fresh.")

    # Get dialogues
    dialogues = get_dialogue_list()
    labeled_count = get_labeled_count()

    print(f"\n  Total dialogues in novel: {len(dialogues)}")
    print(f"  Already labeled: {labeled_count}")
    print(f"  Remaining: {len(dialogues) - labeled_count}")

    if args.count <= 0:
        print("  Count is 0; configuration check only, no annotation requested.")
        if args.validate:
            validate(error_limit=args.error_limit)
        return

    start_idx = args.start if args.start > 0 else labeled_count
    end_idx = min(start_idx + args.count, len(dialogues))

    if start_idx >= len(dialogues):
        print("\n  All dialogues annotated!")
        return

    print(f"  This run: #{start_idx+1} to #{end_idx} ({end_idx - start_idx} dialogues)")

    def fmt_duration(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    start_time = time.time()
    total_rounds = end_idx - start_idx
    quiet_mode = total_rounds > 20

    boss = Boss(short_mem_rounds=args.short_mem, decision_mode=DECISION_MODE)

    batch_tool_calls = 0
    batch_tokens = 0
    batch_model_calls = 0

    for idx in range(start_idx, end_idx):
        line_num, dialogue = dialogues[idx]
        round_log = new_round(idx - start_idx + 1, line_num, dialogue)
        set_current_round_trace(round_log)
        temp_log_event(
            "round_start",
            round_log,
            absolute_index=idx + 1,
            run_start_index=start_idx + 1,
            run_end_index=end_idx,
            formal_log_path=LOG_PATH,
            temp_log_path=TEMP_LOG_PATH,
            labeled_path=LABELED_PATH,
        )
        try:
            speaker = boss.process_one(line_num, dialogue, round_log, quiet=quiet_mode)
            log_entry(round_log)
            temp_log_event(
                "round_complete",
                round_log,
                speaker=speaker,
                formal_log_written=True,
                formal_log_path=LOG_PATH,
            )
        except Exception as exc:
            round_log["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            temp_log_event(
                "round_failed",
                round_log,
                error=round_log["error"],
                formal_log_written=False,
                formal_log_path=LOG_PATH,
            )
            raise
        finally:
            clear_current_round_trace()

        batch_tool_calls += len(round_log.get("tool_calls", []))
        labeler = round_log.get("agents", {}).get("Labeler", {})
        batch_tokens += round_log.get("annotation_tokens", labeler.get("total_tokens", 0))
        batch_model_calls += (
            round_log.get("quality_audit", {}).get("model_calls", 0)
            + round_log.get("quality_ensemble", {}).get("model_calls", 0)
        )

        done = idx - start_idx + 1
        elapsed = time.time() - start_time
        avg_sec = elapsed / done
        remaining_sec = avg_sec * (total_rounds - done)

        if quiet_mode:
            elapsed = time.time() - start_time
            avg_sec = elapsed / done
            remaining_sec = avg_sec * (total_rounds - done)
            _writeline(f"  [{done}/{total_rounds}] L{line_num} -> {speaker:<8s} | "
                       f"calls={batch_model_calls:>3d} tools={batch_tool_calls:>3d} avg={avg_sec:.0f}s "
                       f"elapsed={fmt_duration(elapsed)} remaining={fmt_duration(remaining_sec)}")
        else:
            _writeline(f"  [{done}/{total_rounds}] L{line_num} -> {speaker:<8s} | "
                       f"calls={batch_model_calls:>3d} tools={batch_tool_calls:>3d} avg={avg_sec:.0f}s "
                       f"elapsed={fmt_duration(elapsed)} remaining={fmt_duration(remaining_sec)}")

    if quiet_mode:
        _writeline("")

    _writeline(f"\n{'='*60}")
    _writeline(f"  Annotation complete")
    _writeline(f"  Dialogues annotated: {end_idx - start_idx}")
    _writeline(
        "  Quality review model calls: "
        f"{boss.quality_audit.model_call_count + boss.ensemble.model_call_count}"
    )
    _writeline(f"  Tool calls: {boss.labeler.tool_call_count}")
    _writeline(f"  SearchAgent triggers: {boss.search_agent_triggers}")
    _writeline(f"  Corrections: {boss.corrections}")
    _writeline(f"  Evidence vault: {VAULT_PATH}")
    _writeline(f"  Log: {LOG_PATH}")
    _writeline(f"  Temp trace log: {TEMP_LOG_PATH}")
    _writeline(f"{'='*60}")

    if args.validate:
        validate(error_limit=args.error_limit)


if __name__ == "__main__":
    main()
