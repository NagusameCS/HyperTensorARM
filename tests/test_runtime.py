"""Packaging: runtime discovery for the `hyperarm` CLI."""

import os
import stat
from pathlib import Path

from hyperretro.models.runtime import find_geodessical, _PKG_BIN


def test_env_override_wins(monkeypatch, tmp_path):
    fake = tmp_path / "geodessical"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HYPERARM_RUNTIME", str(fake))
    assert find_geodessical() == fake


def test_packaged_binary_ships_and_is_executable():
    # The wheel ships a compiled geodessical inside hyperretro/bin/.
    assert _PKG_BIN.is_file(), f"runtime binary missing: {_PKG_BIN}"
    assert os.access(_PKG_BIN, os.X_OK), "runtime binary is not executable"
    found = find_geodessical()
    assert found is not None
    assert found.name == "geodessical"


def test_missing_env_and_binary_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("HYPERARM_RUNTIME", raising=False)
    monkeypatch.setattr("hyperretro.models.runtime._USER_BIN", tmp_path / "nope")
    monkeypatch.setattr("hyperretro.models.runtime._PKG_BIN", tmp_path / "nope2")
    monkeypatch.setattr("hyperretro.models.runtime._REPO_BUILD", tmp_path / "nope3")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert find_geodessical() is None
