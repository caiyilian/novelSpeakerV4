import os
import re
import argparse
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
NOVEL_PATH = os.path.join(ROOT_DIR, "data", "novel.txt")
LABELED_PATH = os.path.join(ROOT_DIR, "data", "labeled.txt")


def get_label_count():
    if not os.path.exists(LABELED_PATH):
        return 0
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        count = len(f.readlines())
    return count


def extract_dialogue_with_line_numbers(text):
    dialogues = []
    text_lines = text.split("\n")

    for line_num, line in enumerate(text_lines, start=1):
        matches = re.findall(r"「([^」]+)」", line)
        for dialogue in matches:
            dialogues.append((line_num, dialogue))

    return dialogues


def main():
    parser = argparse.ArgumentParser(description="获取待标注的对话")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="最大一次性获取的对话数量（默认1）"
    )
    parser.add_argument(
        "--threshold", type=int, default=10, help="视为连续对话的最大行号间隔（默认10）"
    )
    args = parser.parse_args()

    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    dialogues = extract_dialogue_with_line_numbers(content)
    now_count = get_label_count()

    if now_count >= len(dialogues):
        print("已经标注完毕")
        return

    # 获取从当前索引开始的对话批次
    batch = []
    last_line_num = None
    idx = now_count

    while idx < len(dialogues) and len(batch) < args.batch_size:
        line_num, dialogue = dialogues[idx]

        # 如果是第一个对话，直接添加
        if last_line_num is None:
            batch.append((line_num, dialogue))
            last_line_num = line_num
            idx += 1
            continue

        # 检查是否连续（行号间隔小于等于阈值）
        if line_num - last_line_num <= args.threshold:
            batch.append((line_num, dialogue))
            last_line_num = line_num
            idx += 1
        else:
            # 遇到不连续的对话，停止
            break

    if len(batch) == 0:
        # 这不应该发生，但为了安全
        print("已经标注完毕")
        return

    # 输出批次信息
    if len(batch) == 1:
        # 单句标注模式
        line_num, dialogue = batch[0]
        print(f"待标注对话：第{line_num}行「{dialogue}」")
        print()
        print("请仔细分析 novel.txt 中对应行的上下文，判断说话角色。")
        print("然后调用 python src/write_label.py --name <角色名>")
    else:
        # 批量标注模式
        print(f"待标注对话批次（{len(batch)}句）：")
        for i, (line_num, dialogue) in enumerate(batch, 1):
            print(f"{i}. 第{line_num}行：「{dialogue}」")
        print()
        print("请仔细分析 novel.txt 中对应行的上下文，判断每句话的说话角色。")
        print("然后调用 write_label.py --name 角色名1 --name 角色名2 ... 按顺序标注。")


if __name__ == "__main__":
    main()
