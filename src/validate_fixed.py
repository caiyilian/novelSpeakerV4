import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LABELED_PATH = os.path.join(ROOT_DIR, "data", "labeled.txt")
ANSWERS_PATH = os.path.join(ROOT_DIR, "data", "answers.txt")

with open(LABELED_PATH, "r", encoding="utf-8") as f:
    labeled = [line.strip() for line in f.readlines() if line.strip()]

with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
    answer_lines = f.readlines()

answers = []
for line in answer_lines:
    match = re.search(r"【([^】]+)】", line)
    if match:
        answers.append(match.group(1))

total = min(len(labeled), len(answers))
correct = 0
wrong = []

for i in range(total):
    label = labeled[i]
    answer = answers[i]
    acceptable = answer.split("|")
    label_parts = label.split("|")
    if any(part in acceptable for part in label_parts):
        correct += 1
    else:
        wrong.append({"idx": i + 1, "expected": answer, "got": label})

accuracy = correct / total * 100 if total > 0 else 0

print(f"{'='*60}")
print(f"  验证报告（修复版）")
print(f"{'='*60}")
print(f"  总对话数:    {total}")
print(f"  正确数:      {correct}")
print(f"  错误数:      {len(wrong)}")
print(f"  正确率:      {accuracy:.1f}%")

if wrong:
    # 统计错误模式
    patterns = {}
    for w in wrong:
        key = f"{w['expected']} → {w['got']}"
        patterns[key] = patterns.get(key, 0) + 1
    
    print(f"\n  错误模式统计:")
    for p, c in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"    {p}  ×{c}")
    
    print(f"\n  错误详情（全部）:")
    for w in wrong:
        print(f"    #{w['idx']}: 期望={w['expected']} | 实际={w['got']}")