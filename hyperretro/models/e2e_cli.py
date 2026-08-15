"""HyperTensor ARM end-to-end CLI (module form).

Installed as the `hyperarm` console command; also usable via
`python -m hyperretro.models.e2e_cli`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GEODESSICAL = REPO_ROOT / "build_host_arm" / "geodessical"


def _add_compress_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="source GGUF")
    p.add_argument("output", help="output GGUF")
    p.add_argument("--ffn-rank", type=int, default=1024,
                   help="SVD rank for FFN matrices (default 1024)")
    p.add_argument("--attn-rank", type=int, default=0,
                   help="SVD rank for attention projections (default 0)")
    p.add_argument("--int4", action="store_true",
                   help="quantize factors to block-wise int4")
    p.add_argument("--int4-block-size", type=int, default=128)
    p.add_argument("--int4-awq", action="store_true",
                   help="AWQ-aware quantization (needs calibration corpus)")
    p.add_argument("--activation-corpus", default=None)


def cmd_compress(args: argparse.Namespace) -> int:
    from hyperretro.models import load_model, compress_model, export_model

    print(f"[e2e] loading {args.input}")
    model = load_model(args.input)
    print(f"[e2e] compressing ffn_rank={args.ffn_rank} int4={args.int4}")
    cm = compress_model(
        model,
        ffn_rank=args.ffn_rank,
        attn_rank=args.attn_rank,
        int4=args.int4,
        int4_block_size=args.int4_block_size,
        int4_awq=args.int4_awq,
        activation_corpus=args.activation_corpus,
    )
    print(f"[e2e] manifest: {cm.manifest.get('ffn', '?')} factored")
    export_model(cm, args.output, format="gguf")
    print(f"[e2e] wrote {args.output}")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    from hyperretro.models import stream_compress_gguf

    stats = stream_compress_gguf(
        args.input, args.output,
        ffn_rank=args.ffn_rank,
        attn_rank=args.attn_rank,
        int4=args.int4,
        int4_block_size=args.int4_block_size,
        int4_awq=args.int4_awq,
    )
    print(f"[e2e] wrote {args.output}: {stats}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    model = args.model
    if not GEODESSICAL.exists():
        print(f"[e2e] geodessical not built at {GEODESSICAL}; run ./build_host_arm.sh first")
        return 1
    print(f"[e2e] verifying {model} with geodessical --ppl-eval")
    r = subprocess.run([str(GEODESSICAL), model, "--ppl-eval"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "PPL]" in line or "PPL-JSON" in line:
            print("  " + line.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    exe = REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if not exe.exists():
        print("[e2e] llama-quantize not built; see third_party/llama.cpp")
        return 1
    r = subprocess.run([str(exe), args.input, args.output, args.qtype], check=False)
    return r.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hyperarm",
        description="HyperTensor ARM end-to-end pipeline "
                    "(compress / stream / verify / quantize GGUF models)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress", help="in-memory compress + export")
    _add_compress_args(pc)
    pc.set_defaults(fn=cmd_compress)

    ps = sub.add_parser("stream", help="single-pass streaming compress (large models)")
    _add_compress_args(ps)
    ps.set_defaults(fn=cmd_stream)

    pv = sub.add_parser("verify", help="verify a GGUF with geodessical --ppl-eval")
    pv.add_argument("model")
    pv.set_defaults(fn=cmd_verify)

    pq = sub.add_parser("quantize", help="re-quantize with llama-quantize")
    pq.add_argument("input")
    pq.add_argument("output")
    pq.add_argument("qtype", default="Q4_K_M", nargs="?")
    pq.set_defaults(fn=cmd_quantize)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
