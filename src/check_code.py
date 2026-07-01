import ast
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_LABEL = ROOT_DIR / "src" / "run_label.py"
FORBIDDEN_TERMS = ROOT_DIR / "config" / "forbidden_terms.txt"


def load_forbidden_terms():
    if not FORBIDDEN_TERMS.exists():
        return []
    with FORBIDDEN_TERMS.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    code = RUN_LABEL.read_text(encoding="utf-8")

    try:
        ast.parse(code)
        print("OK: syntax check passed")
    except SyntaxError as exc:
        print(f"FAIL: syntax error: {exc}")
        return 1

    checks = [
        ("read_novel_lines", True, "Labeler must be able to read original text"),
        ("TEMP_DESCRIPTORS", True, "temporary descriptors should be explicit and generic"),
        ("validate_char_name", True, "speaker names should be validated"),
    ]

    ok = True
    for needle, should_exist, desc in checks:
        exists = needle in code
        if should_exist and not exists:
            print(f"WARN: missing {needle!r} - {desc}")
            ok = False
        elif not should_exist and exists:
            print(f"WARN: forbidden marker {needle!r} - {desc}")
            ok = False

    for term in load_forbidden_terms():
        if term in code:
            print(f"WARN: forbidden project-specific term found in run_label.py: {term!r}")
            ok = False

    if ok:
        print("All checks passed")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
