#!/usr/bin/env python3
"""
Validate code cells in Jupyter notebooks.

Usage:
    python scripts/validate_notebooks.py

Behaviour:
    1. Find all *.ipynb files in the repo.
    2. Extract every code cell.
    3. Syntax-check every cell (compile).
    4. For cells that have no network calls / subprocess / LLM imports,
       also *execute* them in a shared namespace per notebook
       (so cells can depend on definitions from earlier cells).
    5. Report pass / skip / fail for every cell.
"""

import json
import re
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

SKIP_EXECUTION_PATTERNS = [
    # shell / network
    r"!pip",
    r"!git",
    r"!find",
    r"!cat",
    r"subprocess",
    r"uvicorn",
    r"notebook_login",
    # live environment connections
    r"\.hf\.space",
    r"localhost:\d+",
    r"EnvClient\(",
    r"TextArenaEnv\(",
    r"OpenSpielEnv\(",
    r"EchoEnv\(",
    # LLM / GPU
    r"AutoModelForCausalLM",
    r"AutoTokenizer",
    r"GRPOTrainer",
    r"GRPOConfig",
    r"trainer\.",
    r"from_pretrained",
    r"torch\.cuda",
    r"generate_rollout_completions",
    # imports that won't be available locally
    r"from envs\.",
    r"from openenv",
    r"from trl",
    r"from transformers",
    r"from datasets",
    r"from huggingface_hub",
    # file system mutations or reads that depend on prior cloned/created files
    r'os\.makedirs',
    r'open\(.+, *["\']w["\']',
    r'open\(env_file',
    r"shutil\.",
    r"echo-env-modified",
    r"word_game/",
    # cleanup / teardown depending on skipped setup
    r"server\.terminate",
    r"server\.wait",
    r"server\.pid",
    # training utilities
    r"import trackio",
]

_SKIP_RE = re.compile("|".join(SKIP_EXECUTION_PATTERNS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_notebooks():
    return sorted(REPO_ROOT.rglob("*.ipynb"))


def extract_code_cells(nb_path: Path):
    """Return list of (cell_index, source_str) for code cells."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                cells.append((i, source))
    return cells


def should_skip_execution(code: str) -> bool:
    return bool(_SKIP_RE.search(code))


def syntax_check(code: str, label: str) -> tuple[bool, str]:
    # Strip Jupyter magic / shell commands for syntax check
    lines = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append(f"# SHELL: {line}")
        else:
            lines.append(line)
    cleaned = "\n".join(lines)
    try:
        compile(cleaned, label, "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def execute_cell(code: str, ns: dict, label: str) -> tuple[bool, str]:
    # Strip shell / magic lines
    lines = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append(f"# SHELL: {line}")
        else:
            lines.append(line)
    cleaned = "\n".join(lines)
    try:
        exec(compile(cleaned, label, "exec"), ns)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    notebooks = find_notebooks()
    total = 0
    passed = 0
    skipped = 0
    failed = 0
    failures = []

    for nb_path in notebooks:
        rel = nb_path.relative_to(REPO_ROOT)
        print(f"\n{'='*60}")
        print(f"Notebook: {rel}")
        print("=" * 60)

        cells = extract_code_cells(nb_path)
        if not cells:
            print("  (no code cells)")
            continue

        # Shared namespace so cells can reference each other
        nb_ns = {}

        for cell_idx, code in cells:
            total += 1
            label = f"cell[{cell_idx}]"

            # 1. Syntax check
            ok, err = syntax_check(code, label)
            if not ok:
                failed += 1
                failures.append((f"{rel} {label}", "SYNTAX", err, code))
                print(f"  FAIL  {label}: {err}")
                continue

            # 2. Execution (only for pure-logic cells)
            if should_skip_execution(code):
                skipped += 1
                print(f"  SKIP  {label}")
                continue

            ok, err = execute_cell(code, nb_ns, label)
            if ok:
                passed += 1
                print(f"  PASS  {label}")
            else:
                failed += 1
                failures.append((f"{rel} {label}", "RUNTIME", err, code))
                print(f"  FAIL  {label}")
                for line in err.strip().split("\n")[:4]:
                    print(f"        {line}")

    print()
    print("=" * 70)
    print(f"Results: {total} cells | {passed} PASS | {skipped} SKIP | {failed} FAIL")
    print("=" * 70)

    if failures:
        print("\nFailed cells:\n")
        for label, kind, err, code in failures:
            print(f"--- {kind}: {label} ---")
            print(err.strip())
            print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
