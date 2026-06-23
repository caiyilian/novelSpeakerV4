#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准确率计算工具。

比较 data/labeled.txt 与 data/answers.txt 的标注结果。
支持多选项匹配：【A|B|C】表示 A、B、C 中任一都算正确。

用法：
  python validate.py                     # 默认路径
  python validate.py --labeled <文件> --answers <文件>
"""

import re
import os
import sys
import argparse
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

LABELED = os.path.join(ROOT_DIR, "data", "labeled.txt")
ANSWERS = os.path.join(ROOT_DIR, "data", "answers.txt")


def parse_answers(path):
    """从 answers.txt 提取所有 【标注】 并解析为多选项集合。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    accept = []
    for m in re.finditer(r"【([^】]+)】", raw):
        options = [o.strip() for o in m.group(1).split("|")]
        accept.append(set(options))
    return accept


def parse_labels(path):
    """读取 labeled.txt，每行一个说话人。"""
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def compute(labels, accept):
    """逐条比对，支持多选项。"""
    N = min(len(labels), len(accept))
    correct = 0
    errors = []
    for i in range(N):
        if labels[i] in accept[i]:
            correct += 1
        else:
            exp = "|".join(sorted(accept[i]))
            errors.append((i + 1, exp, labels[i]))
    return correct, N, errors


def print_report(labels, accept, errors, N, correct, detail=False):
    wrong = N - correct
    print(f"总对话数:     {N}")
    print(f"正确:         {correct} ({correct / N * 100:.1f}%)")
    print(f"错误:         {wrong} ({wrong / N * 100:.1f}%)")
    print()

    # 分段
    for seg_name, seg_end, seg_start in [("前50轮", 50, 0), ("后50轮", 100, 50)]:
        if N >= seg_end:
            seg_correct = sum(1 for i in range(seg_start, seg_end) if labels[i] in accept[i])
            print(f"{seg_name}:     {seg_correct}/{seg_end - seg_start} ({seg_correct / (seg_end - seg_start) * 100:.1f}%)")
    print()

    # 错误模式统计
    pats = Counter()
    for idx, exp, got in errors:
        pats[f"{exp} → {got}"] += 1
    print("错误模式 Top 15:")
    for pat, cnt in pats.most_common(15):
        print(f"  ×{cnt:>3}  {pat}")

    # 详细列表
    if detail and errors:
        print()
        print("详细错误列表:")
        for idx, exp, got in errors[:detail if isinstance(detail, int) else len(errors)]:
            print(f"  #{idx}: 期望={exp}  实际={got}")


def main():
    parser = argparse.ArgumentParser(description="计算标注准确率，支持多选项匹配")
    parser.add_argument("--labeled", default=LABELED, help="标注结果文件路径")
    parser.add_argument("--answers", default=ANSWERS, help="答案文件路径")
    parser.add_argument("--detail", type=int, nargs="?", const=-1, default=0,
                        help="显示详细错误列表（可选指定行数）")
    args = parser.parse_args()

    labels = parse_labels(args.labeled)
    accept = parse_answers(args.answers)
    correct, N, errors = compute(labels, accept)
    print_report(labels, accept, errors, N, correct, detail=args.detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
