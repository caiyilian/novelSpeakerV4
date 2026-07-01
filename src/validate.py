#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准确率计算工具。

比较 data/labeled.txt 与 data/answers.txt 的标注结果。
支持多选项匹配：【A|B|C】表示 A、B、C 中任一都算正确。

用法：
  python src/validate.py
  python src/validate.py --detail    (显示错误详情)
"""

import re, os, sys, argparse
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELED = os.path.join(ROOT_DIR, "data", "labeled.txt")
ANSWERS = os.path.join(ROOT_DIR, "data", "answers.txt")


def parse_answers(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    accept = []
    for m in re.finditer(r"【([^】]+)】", raw):
        accept.append(set(o.strip() for o in m.group(1).split("|")))
    return accept


def parse_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", default=LABELED)
    parser.add_argument("--answers", default=ANSWERS)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--start", type=int, default=1, help="1-based dialogue index to start validation")
    parser.add_argument("--count", type=int, default=0, help="number of dialogues to validate; 0 means through the end")
    args = parser.parse_args()

    labels = parse_labels(args.labeled)
    accept = parse_answers(args.answers)
    total = min(len(labels), len(accept))
    start = max(1, args.start)
    start_i = min(start - 1, total)
    end_i = total if args.count <= 0 else min(start_i + args.count, total)
    N = end_i - start_i
    if N <= 0:
        print("No dialogues in selected validation range.")
        return 1

    correct = 0
    errors = []
    for i in range(start_i, end_i):
        if labels[i] in accept[i]:
            correct += 1
        else:
            errors.append((i + 1, "|".join(sorted(accept[i])), labels[i]))

    wrong = N - correct
    print(f"验证范围: #{start_i + 1}-#{end_i} / total {total}")
    print(f"总对话数: {N}")
    print(f"正确:     {correct} ({correct / N * 100:.1f}%)")
    print(f"错误:     {wrong} ({wrong / N * 100:.1f}%)")
    print()

    pats = Counter()
    for _, exp, got in errors:
        pats[f"{exp} -> {got}"] += 1
    print("错误模式:")
    for pat, cnt in pats.most_common(15):
        print(f"  x{cnt:>3}  {pat}")

    if args.detail and errors:
        print()
        print("详细错误列表:")
        for idx, exp, got in errors:
            print(f"  #{idx}: 期望={exp}  实际={got}")


if __name__ == "__main__":
    sys.exit(main())
