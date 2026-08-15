"""
jury_states.py — shared Saiyan state loader for the geometric-jury scripts
(jury_discovery.py, jury_solver.py, jury_advance.py).

Loads 6-domain Saiyan trajectory states from outputs/saiyan_states/*_saiyan.pt
when they exist. When they are absent (clean checkout), generates deterministic
synthetic states so the experiments run standalone on any platform (x86, ARM,
Apple Silicon). Synthetic states are saved back to outputs/saiyan_states/ so
subsequent runs reload the same data.
"""
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

DOMAINS = {
    "Goku": "math", "Vegeta": "code", "Gohan": "science",
    "Piccolo": "logic", "Trunks": "creative", "Yamcha": "general",
}

K_DIM = 20
STATE_DIR = Path("outputs/saiyan_states")


def _state_dir() -> Path:
    if STATE_DIR.exists():
        return STATE_DIR
    alt = Path("/home/ubuntu/outputs/saiyan_states")
    return alt if alt.exists() else STATE_DIR


def _synthetic_states(field="proj", target=50, importance=False):
    """Deterministic synthetic Saiyan manifolds: 6 well-separated directions
    on the K_DIM sphere, trajectories = direction + noise, normalized."""
    g = torch.Generator().manual_seed(42)
    # 6 independent directions, orthogonalized so domains are separable
    raw = torch.randn(len(DOMAINS), K_DIM, generator=g)
    bases = []
    for v in raw:
        u = v.clone()
        for b in bases:
            u = u - (u @ b) * b
        u = F.normalize(u, dim=0)
        bases.append(u)
    states = {}
    for (name, domain), base in zip(DOMAINS.items(), bases):
        trajs = []
        for i in range(target):
            noise = torch.randn(K_DIM, generator=g) * 0.18
            p = F.normalize((base + noise).unsqueeze(0), dim=1).squeeze(0)
            t = {field: p.float(), "parent": name, "domain": domain}
            if importance:
                t["importance"] = 1.0
            trajs.append(t)
        states[name] = trajs
    # Persist so future runs take the same path
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for name, trajs in states.items():
        payload = {
            "K": K_DIM,
            "trajectories": [{"proj": t[field].float()} for t in trajs],
        }
        torch.save(payload, STATE_DIR / f"{name}_saiyan.pt")
    return states


def _augment(trajs, target, field="proj"):
    if len(trajs) >= target:
        return trajs[:target]
    rng = random.Random(42)
    result = list(trajs)
    while len(result) < target:
        i, j = rng.sample(range(len(trajs)), 2)
        a = rng.random()
        noise = torch.randn_like(trajs[0][field]) * 0.02 * 0.5
        mixed = F.normalize(
            (trajs[i][field] * a + trajs[j][field] * (1 - a) + noise).unsqueeze(0),
            dim=1,
        ).squeeze(0)
        t = {field: mixed, "parent": trajs[0]["parent"], "domain": trajs[0]["domain"]}
        if "importance" in trajs[0]:
            t["importance"] = 1.0
        result.append(t)
    return result


def load_saiyans(field="proj", target=None, importance=False):
    """Return {name: [trajectory dicts]} for the six domains.

    field:     key used for the projection vector ('proj' or 'feat').
    target:    if set, augment (or trim) each domain to this many trajectories.
    importance: include an 'importance' field (jury_solver uses it).
    """
    saiyans = {}
    sdir = _state_dir()
    for pt_file in sorted(sdir.glob("*_saiyan.pt")):
        name = pt_file.stem.replace("_saiyan", "")
        if name not in DOMAINS:
            continue
        data = torch.load(pt_file, map_location="cpu")
        trajs = []
        for t in data.get("trajectories", []):
            if isinstance(t, dict) and "proj" in t:
                traj = {field: t["proj"].float(), "parent": name, "domain": DOMAINS[name]}
                if importance:
                    traj["importance"] = 1.0
                trajs.append(traj)
        saiyans[name] = trajs

    if not saiyans:
        saiyans = _synthetic_states(field=field, target=target or 50, importance=importance)

    if target:
        for name in saiyans:
            saiyans[name] = _augment(saiyans[name], target, field=field)
    return saiyans
