# -*- coding: utf-8 -*-
"""
SearchAgent - dedicated identity investigator triggered by temporary names.

When Labeler outputs a temporary descriptor or role label, SearchAgent searches
the novel for the character's actual name.
"""

import json
import re

SEARCH_SYSTEM_PROMPT = """You are a novel identity investigator. Your job is to find the real identity of temporary character descriptors.

## Your task
When you receive a temporary/ambiguous descriptor or role label, search the novel to find that character's actual name.

## Investigation Strategy
1. First, read 10-20 lines around the target line to understand context.
2. Search for self-introduction patterns within ~200 lines forward:
   - "我叫XX", "咱是XX", "我是XX", "名字是XX", "吾乃XX", "咱的名字是"
3. Check if the temporary descriptor refers to a known character by matching context.
4. If the character introduces themselves by name, that's VERIFIED evidence.
5. If narrative states the character's name, that's VERIFIED evidence.
6. If you can only infer who they are (e.g., a descriptor seen near a known character), that's CANDIDATE evidence.

## Available Tools
- read_novel_lines(start, count): Read novel lines
- deep_search_identity(temp_name, around_line, search_forward=200, search_backward=100): Automated search for identity clues
- find_all_references(name, max_results=10): Find all occurrences of a name

## Output Format
Return ONLY a JSON object, no other text:
{
  "found": true/false,
  "character": "canonical_name",
  "aliases": ["alias1", "alias2"],
  "introduction_line": null or line number,
  "status": "verified" or "candidate",
  "evidence": [
    {"text": "exact text with evidence", "line": line_number, "type": "self_intro"}
  ]
}

If not found:
{
  "found": false,
  "character": "",
  "evidence": []
}"""


class SearchAgent:
    def __init__(self, call_ollama_fn, read_novel_fn, deep_search_fn, find_refs_fn):
        self.call_ollama = call_ollama_fn
        self.read_novel = read_novel_fn
        self.deep_search = deep_search_fn
        self.find_refs = find_refs_fn
        self.tool_call_count = 0

    def _safe_print(self, msg):
        """Print to stderr, which is more resilient on Windows."""
        import sys
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except (ValueError, OSError):
            pass

    def investigate(self, temp_name, around_line, max_tool_rounds=4, quiet=False):
        from run_label import TOOL_READ_NOVEL, normalize_tool_call, parse_tool_arguments
        TOOL_DEEP_SEARCH = {
            "type": "function",
            "function": {
                "name": "deep_search_identity",
                "description": "Search for identity clues for a temporary character descriptor near a given line. Looks for self-introduction patterns and name reveals.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "temp_name": {"type": "string", "description": "The temporary descriptor to search for"},
                        "around_line": {"type": "integer", "description": "The line number around which to search"},
                        "search_forward": {"type": "integer", "description": "Lines to search forward (default: 200)"},
                        "search_backward": {"type": "integer", "description": "Lines to search backward (default: 100)"}
                    },
                    "required": ["temp_name", "around_line"]
                }
            }
        }
        TOOL_FIND_REFS = {
            "type": "function",
            "function": {
                "name": "find_all_references",
                "description": "Find all occurrences of a character name in the novel, returning line numbers and context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The character name to search for"},
                        "max_results": {"type": "integer", "description": "Maximum results (default: 10)"}
                    },
                    "required": ["name"]
                }
            }
        }
        search_tools = [TOOL_READ_NOVEL, TOOL_DEEP_SEARCH, TOOL_FIND_REFS]

        messages = [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Investigate the identity of this temporary character descriptor:\n\n"
                f"Temporary name: \"{temp_name}\"\n"
                f"Target dialogue is around line: {around_line}\n\n"
                f"Search the novel to find the character's actual name.\n"
                f"Start by reading context around line {around_line}.\n"
                f"Then use deep_search_identity or find_all_references."
            )}
        ]

        total_pec = 0
        total_ec = 0
        tool_call_log = []
        final_text = ""

        for round_i in range(max_tool_rounds):
            text, pec, ec, tool_calls = self.call_ollama(messages, tools=search_tools, label=f"SearchAgent-R{round_i+1}")
            total_pec += pec
            total_ec += ec
            final_text = text

            if not tool_calls:
                break

            parsed_tool_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", {})
                name = func.get("name", "")
                func_args, parse_error = parse_tool_arguments(name, raw_args)
                parsed_tool_calls.append({
                    "tool_call": normalize_tool_call(tc, func_args),
                    "function": name,
                    "arguments": func_args,
                    "parse_error": parse_error,
                })

            first_call = parsed_tool_calls[0] if parsed_tool_calls else {}
            tool_call_log.append({
                "round": round_i + 1,
                "function": first_call.get("function", ""),
                "args": first_call.get("arguments", {}),
                "parse_error": first_call.get("parse_error", ""),
                "pec": pec,
                "ec": ec
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
                    result = f"Tool arguments invalid: {parse_error}. Call the tool again with valid JSON arguments."
                    if not quiet:
                        self._safe_print(f"    [SearchAgent] invalid tool args for {name}: {parse_error}")
                    messages.append({"role": "tool", "content": result})
                    continue

                if name == "read_novel_lines":
                    try:
                        s = int(func_args.get("start", 1) or 1)
                        c = int(func_args.get("count", 10) or 10)
                    except (TypeError, ValueError):
                        result = "Tool arguments invalid: start and count must be integers. Call read_novel_lines again with valid JSON."
                        messages.append({"role": "tool", "content": result})
                        continue
                    result = self.read_novel(s, c)
                    self.tool_call_count += 1
                    if not quiet:
                        self._safe_print(f"    [SearchAgent] read_novel_lines({s}-{s+c-1}) -> {len(result)} chars")
                    messages.append({"role": "tool", "content": result})
                elif name == "deep_search_identity":
                    tn = func_args.get("temp_name", temp_name)
                    try:
                        al = int(func_args.get("around_line", around_line) or around_line)
                    except (TypeError, ValueError):
                        al = around_line
                    try:
                        fwd = int(func_args.get("search_forward", 200) or 200)
                    except (TypeError, ValueError):
                        fwd = 200
                    try:
                        bwd = int(func_args.get("search_backward", 100) or 100)
                    except (TypeError, ValueError):
                        bwd = 100
                    result = self.deep_search(tn, al, fwd, bwd)
                    self.tool_call_count += 1
                    if not quiet:
                        self._safe_print(f"    [SearchAgent] deep_search('{tn}', around={al}) -> {len(result)} chars")
                    messages.append({"role": "tool", "content": result})
                elif name == "find_all_references":
                    n = func_args.get("name", "")
                    try:
                        mr = int(func_args.get("max_results", 10) or 10)
                    except (TypeError, ValueError):
                        mr = 10
                    result = self.find_refs(n, mr)
                    self.tool_call_count += 1
                    if not quiet:
                        self._safe_print(f"    [SearchAgent] find_references('{n}') -> {len(result)} chars")
                    messages.append({"role": "tool", "content": result})
                else:
                    result = f"Unsupported tool '{name}'. Use read_novel_lines, deep_search_identity, or find_all_references."
                    messages.append({"role": "tool", "content": result})
        else:
            if not quiet:
                self._safe_print(f"    [SearchAgent] Max rounds ({max_tool_rounds})")

        result = self._parse_result(final_text)
        return result, total_pec, total_ec, tool_call_log

    def _parse_result(self, text):
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                return {
                    "found": parsed.get("found", False),
                    "character": parsed.get("character", ""),
                    "aliases": parsed.get("aliases", []),
                    "introduction_line": parsed.get("introduction_line"),
                    "status": parsed.get("status", "candidate"),
                    "evidence": parsed.get("evidence", [])
                }
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"found": False, "character": "", "aliases": [], "evidence": []}
