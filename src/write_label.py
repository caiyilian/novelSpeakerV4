# write_label.py
import sys
import os
import argparse

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LABELED_PATH = os.path.join(ROOT_DIR, "data", "labeled.txt")


def main():
    parser = argparse.ArgumentParser(description="追加名字到 labeled.txt")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        action="append",
        help="要追加的名字（可以多次使用，按顺序）",
    )
    args = parser.parse_args()

    names = args.name

    # 追加到文件，每个名字一行
    with open(LABELED_PATH, "a", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print(f"已标注 {len(names)} 个角色：{', '.join(names)}")


if __name__ == "__main__":
    main()
