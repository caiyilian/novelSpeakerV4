import ast, sys

with open(r'E:\projects\novelSpeakerV4\run_label.py', 'r', encoding='utf-8') as f:
    code = f.read()

try:
    ast.parse(code)
    print('OK: 语法检查通过')
except SyntaxError as e:
    print(f'FAIL: 语法错误: {e}')
    sys.exit(1)

checks = [
    ('context_lines', False, '上下文片段不应出现在输入中'),
    ('请先调用 read_novel_lines', True, '用户消息应包含工具调用提示'),
    ('你必须用 read_novel_lines', True, 'System prompt 需强调无原文'),
    ('我没有给你小说原文', True, '用户消息需说明无原文'),
    ('她', False, '不应出现具体人名'),
    ('赫萝', False, '不应出现具体角色名'),
    ('罗伦斯', False, '不应出现具体角色名'),
    ('贤狼', False, '不应出现具体角色名'),
]

all_ok = True
for s, should_exist, desc in checks:
    exists = s in code
    if should_exist and not exists:
        print(f'  WARN: 应存在但未找到: "{s}" — {desc}')
        all_ok = False
    elif not should_exist and exists:
        print(f'  WARN: 不应存在但找到了: "{s}" — {desc}')
        all_ok = False

if all_ok:
    print('所有检查通过')