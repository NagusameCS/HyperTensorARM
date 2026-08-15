"""HyperTensor ARM end-to-end CLI (module form).

Installed as the `hyperarm` console command; also usable via
`python -m hyperretro.models.e2e_cli`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hyperretro.models.runtime import (
    find_geodessical,
    install_runtime,
)


def _add_compress_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="source GGUF")
    p.add_argument("output", help="output GGUF")
    p.add_argument("--ffn-rank", type=int, default=1024,
                   help="SVD rank for FFN matrices (default 1024)")
    p.add_argument("--attn-rank", type=int, default=0,
                   help="GRC shared-basis rank for attention Q/K/V (default 0)")
    p.add_argument("--int4", action="store_true",
                   help="quantize factors to block-wise int4")
    p.add_argument("--int4-block-size", type=int, default=128)
    p.add_argument("--int4-awq", action="store_true",
                   help="AWQ-aware quantization (needs calibration corpus)")
    p.add_argument("--activation-corpus", default=None)
    p.add_argument("--sink", type=int, default=0,
                   help="attention-sink columns restored verbatim in GRC (0 = vanilla)")


def cmd_compress(args: argparse.Namespace) -> int:
    from hyperretro.models import load_model, compress_model, export_model

    print(f"[e2e] loading {args.input}")
    model = load_model(args.input)
    print(f"[e2e] compressing ffn_rank={args.ffn_rank} attn_rank={args.attn_rank} "
          f"sink={args.sink} int4={args.int4}")
    cm = compress_model(
        model,
        ffn_rank=args.ffn_rank,
        attn_rank=args.attn_rank,
        int4=args.int4,
        int4_block_size=args.int4_block_size,
        int4_awq=args.int4_awq,
        activation_corpus=args.activation_corpus,
        sink_T=args.sink,
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
        sink_T=args.sink,
    )
    print(f"[e2e] wrote {args.output}: {stats}")
    return 0


def _runtime_or_fail() -> Path | None:
    exe = find_geodessical()
    if exe is None:
        print("[hyperarm] geodessical runtime not found. Fix with:")
        print("  hyperarm install-runtime            # build from this checkout")
        print("  hyperarm install-runtime --repo PATH # build from a source checkout")
        print("  export HYPERARM_RUNTIME=/path/to/geodessical")
        return None
    return exe


def cmd_verify(args: argparse.Namespace) -> int:
    exe = _runtime_or_fail()
    if exe is None:
        return 1
    print(f"[e2e] verifying {args.model} with {exe.name} --ppl-eval")
    r = subprocess.run([str(exe), args.model, "--ppl-eval"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "PPL]" in line or "PPL-JSON" in line:
            print("  " + line.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    exe = _runtime_or_fail()
    if exe is None:
        return 1
    cmd = [str(exe), args.model, "-p", args.prompt, "-n", str(args.n_tokens)]
    if args.temp is not None:
        cmd += ["--temp", str(args.temp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    return 0


def cmd_install_runtime(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve() if args.repo else None
    dst = install_runtime(repo_root=repo, force=args.force)
    if dst is None:
        print("[hyperarm] could not obtain the runtime binary.")
        print("  Pass --repo /path/to/HyperTensorARM (a source checkout) or")
        print("  build it there manually with ./build_host_arm.sh")
        return 1
    print(f"[hyperarm] runtime installed: {dst}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    print(f"platform: {platform.platform()}")
    print(f"python:   {sys.version.split()[0]}")
    try:
        from importlib.metadata import version
        print(f"hypertensor: {version('hypertensor')}")
    except Exception:
        print("hypertensor: (not installed; running from source)")

    exe = find_geodessical()
    if exe is None:
        print("runtime:  NOT FOUND (run: hyperarm install-runtime)")
    else:
        print(f"runtime:  {exe}")
        try:
            r = subprocess.run([str(exe), "--version"],
                               capture_output=True, text=True, timeout=30)
            print(f"          {r.stdout.strip().splitlines()[0] if r.stdout.strip() else 'ok'}")
        except Exception:
            pass

    for mod, label in (
        ("torch", "torch (needed for compress/stream)"),
        ("gguf", "gguf (GGUF read/write)"),
        ("numpy", "numpy"),
    ):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v is None:
                from importlib.metadata import version
                try:
                    v = version(mod)
                except Exception:
                    v = "installed"
            print(f"{mod:<8} {v}  ({label})")
        except Exception:
            print(f"{mod:<8} MISSING  ({label})")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    exe = Path(__file__).resolve().parent.parent.parent / "third_party" / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if not exe.exists():
        print("[e2e] llama-quantize not built; see third_party/llama.cpp")
        print("  (install llama.cpp and build llama-quantize, or quantize with")
        print("   your own llama.cpp checkout)")
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

    ppl = sub.add_parser("ppl", help="perplexity of a GGUF (alias of verify)")
    ppl.add_argument("model")
    ppl.set_defaults(fn=cmd_verify)

    pi = sub.add_parser("infer", help="generate text with the ARM runtime")
    pi.add_argument("model")
    pi.add_argument("-p", "--prompt", default="Hello", help="prompt text")
    pi.add_argument("-n", "--n-tokens", type=int, default=32,
                    help="tokens to generate (default 32)")
    pi.add_argument("--temp", type=float, default=None, help="sampling temperature")
    pi.set_defaults(fn=cmd_infer)

    pq = sub.add_parser("quantize", help="re-quantize with llama-quantize")
    pq.add_argument("input")
    pq.add_argument("output")
    pq.add_argument("qtype", default="Q4_K_M", nargs="?")
    pq.set_defaults(fn=cmd_quantize)

    pir = sub.add_parser("install-runtime",
                         help="build and install the geodessical runtime to ~/.hyperarm")
    pir.add_argument("--repo", default=None,
                     help="path to a HyperTensor ARM source checkout (to build from)")
    pir.add_argument("--force", action="store_true", help="rebuild even if found")
    pir.set_defaults(fn=cmd_install_runtime)

    pd = sub.add_parser("doctor", help="report environment and dependency status")
    pd.set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
