#!/usr/bin/env python3
"""
Validate Python code blocks extracted from Markdown files.

Usage:
    python scripts/validate_snippets.py [--fix]

Behaviour:
    1. Find all *.md files in the repo (excluding .git).
    2. Extract every ```python ... ``` fenced code block.
    3. Syntax-check every block (compile).
    4. For blocks that have no network calls / subprocess / LLM imports,
       also *execute* them in an isolated namespace.
    5. Report pass / skip / fail for every block.

Network / LLM blocks are detected heuristically and only syntax-checked —
they are marked SKIP (execution) but still must parse cleanly.
"""

import ast
import re
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SKIP_EXECUTION_PATTERNS = [
    # network / live environment calls
    r"\.hf\.space",
    r"localhost:\d+",
    r"subprocess",
    r"uvicorn",
    r"openenv push",
    r"git clone",
    # LLM / GPU
    r"AutoModelForCausalLM",
    r"GRPOTrainer",
    r"GRPOConfig",
    r"trainer\.train",
    r"notebook_login",
    r"from_pretrained",
    r"torch\.cuda",
    # package installation (side effects)
    r"!pip",
    r"pip install",
    # file system mutations we don't want
    r'os\.makedirs',
    r'open\(.+, *["\']w["\']',
    # imports that won't be available locally
    r"from envs\.",
    r"from openenv",
    r"from trl",
    r"from transformers",
    r"from datasets",
    r"import trackio",
    # pseudo-code / class skeletons with undefined names
    r"class \w+\(ABC\)",
    r"class \w+\(Environment\)",
    r"class \w+\(EnvClient\)",
    r"class \w+\(HTTPEnvClient\)",
    r"while not done:",
    r"policy\.choose",
    r"environment\.observe",
    # standalone env calls without context
    r"^env\.(reset|step|state)\(",
]

_SKIP_RE = re.compile("|".join(SKIP_EXECUTION_PATTERNS))

FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_markdown_files():
    return sorted(REPO_ROOT.rglob("*.md"))


def extract_snippets(md_path: Path):
    """Return list of (heading_context, code_str) tuples."""
    text = md_path.read_text(encoding="utf-8")
    snippets = []
    last_heading = "(top-level)"
    for line in text.split("\n"):
        if line.startswith("#"):
            last_heading = line.strip()
    # Re-do properly: track heading as we scan
    current_heading = "(top-level)"
    pos = 0
    for m in FENCE_RE.finditer(text):
        # Find last heading before this match
        before = text[:m.start()]
        heading_matches = list(re.finditer(r"^#+.+", before, re.MULTILINE))
        if heading_matches:
            current_heading = heading_matches[-1].group().strip()
        snippets.append((current_heading, m.group(1)))
    return snippets


def should_skip_execution(code: str) -> bool:
    return bool(_SKIP_RE.search(code))


def syntax_check(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<snippet>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def execute_snippet(code: str) -> tuple[bool, str]:
    ns = {}
    try:
        exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    md_files = find_markdown_files()
    total = 0
    passed = 0
    skipped = 0
    failed = 0
    failures = []

    for md_path in md_files:
        rel = md_path.relative_to(REPO_ROOT)
        snippets = extract_snippets(md_path)
        if not snippets:
            continue

        for i, (heading, code) in enumerate(snippets):
            total += 1
            label = f"{rel} [{heading}] snippet #{i+1}"

            # 1. Syntax check (always)
            ok, err = syntax_check(code)
            if not ok:
                failed += 1
                failures.append((label, "SYNTAX", err, code))
                print(f"  FAIL  {label}")
                print(f"        {err}")
                continue

            # 2. Execution check (only for pure-logic blocks)
            if should_skip_execution(code):
                skipped += 1
                print(f"  SKIP  {label}")
                continue

            ok, err = execute_snippet(code)
            if ok:
                passed += 1
                print(f"  PASS  {label}")
            else:
                failed += 1
                failures.append((label, "RUNTIME", err, code))
                print(f"  FAIL  {label}")
                # Print first 3 lines of traceback
                for line in err.strip().split("\n")[:4]:
                    print(f"        {line}")

    print()
    print("=" * 70)
    print(f"Results: {total} snippets | {passed} PASS | {skipped} SKIP | {failed} FAIL")
    print("=" * 70)

    if failures:
        print("\nFailed snippets:\n")
        for label, kind, err, code in failures:
            print(f"--- {kind}: {label} ---")
            print(err.strip())
            print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
