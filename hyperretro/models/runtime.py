"""Locate, build and install the geodessical ARM inference runtime.

The `hyperarm` CLI shells out to the `geodessical` binary for inference,
PPL evaluation and verification. This module centralizes discovery of that
binary so the tool works from an editable checkout, a pip wheel, or a
user-level install:

Search order for ``find_geodessical()``:
1. ``$HYPERARM_RUNTIME`` (explicit override)
2. ``~/.hyperarm/bin/geodessical`` (user-level install)
3. package data ``hyperretro/bin/geodessical`` (shipped with the wheel)
4. repo checkout ``build_host_arm/geodessical`` (developer build)
5. ``geodessical`` on ``$PATH``
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent.parent
_PKG_BIN = _PKG_DIR / "bin" / "geodessical"
_REPO_ROOT = _PKG_DIR.parent.parent
_REPO_BUILD = _REPO_ROOT / "build_host_arm" / "geodessical"
_REPO_BUILD_SCRIPT = _REPO_ROOT / "build_host_arm.sh"
_USER_BIN = Path.home() / ".hyperarm" / "bin" / "geodessical"


def find_geodessical() -> Path | None:
    """Return the path to the geodessical binary, or None if not found."""
    candidates: list[Path] = []
    env = os.environ.get("HYPERARM_RUNTIME")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [_USER_BIN, _PKG_BIN, _REPO_BUILD]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    which = shutil.which("geodessical")
    return Path(which) if which else None


def build_geodessical(repo_root: Path | None = None) -> Path | None:
    """Build the runtime from the source checkout (macOS arm64, clang).

    Requires a checkout of the HyperTensor ARM repository. Returns the
    binary path on success, None on failure.
    """
    root = Path(repo_root) if repo_root else _REPO_ROOT
    script = root / "build_host_arm.sh"
    if not script.is_file():
        return None
    subprocess.run(["bash", str(script)], cwd=root, check=True)
    out = root / "build_host_arm" / "geodessical"
    return out if out.is_file() else None


def install_runtime(repo_root: Path | None = None, force: bool = False) -> Path | None:
    """Copy (or build, if needed) geodessical into ~/.hyperarm/bin.

    Args:
        repo_root: optional path to a source checkout for building.
        force: rebuild even if a binary was found.

    Returns:
        Path to the installed binary, or None if no binary is available.
    """
    src = None if force else find_geodessical()
    if src is None:
        src = build_geodessical(repo_root)
    if src is None:
        return None
    _USER_BIN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, _USER_BIN)
    return _USER_BIN
