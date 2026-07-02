#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小说角色名泄露检查工具。

确保项目代码（.py 文件）不包含特定小说的角色名/地名/专有名词，
以保证 Prompt 和相关代码在用于其他小说时保持通用性，不会"作弊"。

用法：
  python check_roster.py [--path <目录>] [--roster <名单文件>]

默认扫描 src/ 目录，优先使用 config/forbidden_terms.txt。
"""

import os
import re
import sys
import argparse
from typing import List, Optional, Tuple

# ──────────────────────────── 默认黑名单 ────────────────────────────
# Keep source code novel-agnostic. Do not put project-specific character names
# here. Put them in config/forbidden_terms.txt or pass a roster with --roster.
DEFAULT_ROSTER: List[str] = []
DEFAULT_ROSTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "forbidden_terms.txt",
)

# ──────────────────────────── 逻辑 ────────────────────────────

def load_roster(path: Optional[str]) -> List[str]:
    """从文件读取名单，每行一个词；否则使用私有默认名单。"""
    if path is None:
        if os.path.exists(DEFAULT_ROSTER_PATH):
            path = DEFAULT_ROSTER_PATH
        else:
            return DEFAULT_ROSTER
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def scan_py_files(root: str, roster: List[str]) -> List[Tuple[str, int, str, str]]:
    """扫描目录下所有 .py 文件，返回违规列表。

    返回格式: [(文件路径, 行号, 匹配词, 行内容), ...]
    """
    violations: List[Tuple[str, int, str, str]] = []
    if not roster:
        return violations
    pattern = re.compile("|".join(re.escape(word) for word in roster))

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            # 跳过检查工具自身（名单里的词只用于匹配，不进入 Prompt）
            if filename in ("check_roster.py", "check_code.py"):
                continue
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        match = pattern.search(line)
                        if match:
                            violations.append(
                                (filepath, lineno, match.group(), line.strip())
                            )
            except UnicodeDecodeError:
                print(f"  ⚠️ 跳过（无法解码）: {filepath}", file=sys.stderr)

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 .py 文件中是否泄露了小说专有名词")
    parser.add_argument(
        "--path",
        default=None,
        help="扫描目录（默认: src/）",
    )
    parser.add_argument(
        "--roster",
        default=None,
        help="名单文件路径，每行一个词（默认使用内置名单）",
    )
    args = parser.parse_args()

    root = args.path if args.path else os.path.dirname(__file__)
    roster = load_roster(args.roster)

    print(f"🔍 扫描目录: {root}")
    print(f"📋 名单词数: {len(roster)}")
    print()

    violations = scan_py_files(root, roster)

    if not violations:
        print("✅ 未发现角色名泄露，所有 .py 文件通过检查。")
        return 0

    # 按文件分组
    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for filepath, lineno, word, line in violations:
        by_file.setdefault(filepath, []).append((lineno, word, line))

    print(f"❌ 发现 {len(violations)} 处违规，涉及 {len(by_file)} 个文件：")
    print()
    for filepath, entries in sorted(by_file.items()):
        print(f"  📄 {filepath}")
        for lineno, word, line in entries:
            print(f"     行 {lineno}: 匹配「{word}」")
            truncated = line[:100] + ("…" if len(line) > 100 else "")
            print(f"              {truncated}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
