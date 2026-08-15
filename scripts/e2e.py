#!/usr/bin/env python3
"""HyperTensor ARM end-to-end CLI (thin wrapper).

See hyperretro/models/e2e_cli.py — installed as the `hyperarm` command.
Usage:
    python scripts/e2e.py compress in.gguf out.gguf --ffn-rank 1024 --int4
    python scripts/e2e.py stream   in.gguf out.gguf --ffn-rank 1024 --int4
    python scripts/e2e.py verify   out.gguf
    python scripts/e2e.py quantize out.gguf out-q4km.gguf Q4_K_M
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperretro.models.e2e_cli import main

if __name__ == "__main__":
    sys.exit(main())
