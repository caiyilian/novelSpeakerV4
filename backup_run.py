#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动备份脚本：将 data/ 下除 novel.txt 和 answers.txt 之外的所有文件
移动到 backup/{timestamp}/ 中，清空 data 目录为非运行状态。

用法：
  python backup_run.py
"""

import os
import sys
import shutil
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
BACKUP_DIR = os.path.join(ROOT_DIR, "backup")

KEEP_FILES = {"novel.txt", "answers.txt"}


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"❌ data/ 目录不存在")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(target, exist_ok=True)

    moved = 0
    for name in os.listdir(DATA_DIR):
        if name in KEEP_FILES:
            print(f"  ⏭️  保留: {name}")
            continue

        src = os.path.join(DATA_DIR, name)
        dst = os.path.join(target, name)

        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            shutil.rmtree(src)
        else:
            shutil.move(src, dst)

        print(f"  ✅ 已移动: {name}")
        moved += 1

    print(f"\n🎉 备份完成: {target}")
    print(f"   移动了 {moved} 个文件/文件夹")
    print(f"   data/ 现仅保留: {', '.join(sorted(KEEP_FILES))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
