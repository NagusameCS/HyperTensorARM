"""Repo hygiene: tracked source files must be free of decorative emoji.

Run: .venv311/bin/python -m pytest tests/test_no_emoji.py -q
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that must stay emoji-free (docs + code that ships in this repo).
TRACKED_FILES = [
    "README.md",
    "ARM_CLAIMS.md",
    "AGENTS.md",
    ".github/copilot-instructions.md",
    "docs/E2E_PIPELINE.md",
    "Makefile",
    "hyperretro/models/_export.py",
    "hyperretro/models/_stream.py",
    "hyperretro/models/__init__.py",
    "hyperretro/models/_compress.py",
    "hyperretro/models/e2e_cli.py",
    "scripts/e2e.py",
]

EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # pictographs, transport, symbols
    (0x2600, 0x27BF),    # misc symbols / dingbats
    (0x2B00, 0x2BFF),    # arrows / stars
    (0xFE00, 0xFE0F),    # variation selectors
)


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in EMOJI_RANGES)


def test_tracked_files_have_no_emoji():
    offenders = []
    for rel in TRACKED_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, ch in enumerate(text):
            if _is_emoji(ch):
                line = text[:i].count("\n") + 1
                offenders.append(f"{rel}:{line}")
                break
    assert not offenders, "emoji found in tracked files: " + ", ".join(offenders)
