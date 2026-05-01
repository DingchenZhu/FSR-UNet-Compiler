"""
Feature Buffer address allocation benchmark — USR-Net (256x256).
Implements three strategies and compares against theoretical optimal.

Key fix: skip tensors are the SAME physical tensor as the layer output,
just with extended live range. No double-counting.

Run:  python ir/mem_alloc.py
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Tensor:
    name: str
    size: int        # words
    def_layer: int
    last_use: int    # inclusive
    buffer: str      # "a" or "b"


def build_unet_tensors() -> List[Tensor]:
    """
    USR-Net 32-layer output tensors with correct live ranges.
    Skip tensors are the same physical allocation as their layer output —
    they just have extended last_use (live until the decoder cat).

    Sizes estimated from compilation: h_out * ceil(w_out * cout / 64)
    Buffer assignment: ping-pong a/b per the hardware constraint.
    """
    # (name, size_words, def_layer, last_use_layer, buffer)
    #   last_use = def+1 for normal; skip tensors extend to decoder cat layer
    raw = [
        # Encoder
        ("L00", 8192,  0,  1, "b"),          # 256x256 c8
        ("L01", 8192,  1, 30, "a"),          # 256x256 c8 — SKIP → L30 cat
        ("L02", 2048,  2,  3, "b"),          # 128x128 c8 (pool)
        ("L03", 4096,  3,  4, "a"),          # 128x128 c16
        ("L04", 4096,  4, 28, "b"),          # 128x128 c16 — SKIP → L28 cat
        ("L05", 1024,  5,  6, "a"),          # 64x64 c16 (pool)
        ("L06", 2048,  6,  7, "b"),          # 64x64 c32
        ("L07", 2048,  7, 24, "a"),          # 64x64 c32 — SKIP → L24 cat
        ("L08",  512,  8,  9, "b"),          # 32x32 c32 (pool)
        ("L09",  512,  9, 10, "a"),          # 32x32 c32
        ("L10", 1024, 10, 11, "b"),          # 32x32 c64
        ("L11", 1024, 11, 12, "a"),          # 32x32 c64
        ("L12", 1024, 12, 20, "b"),          # 32x32 c64 — SKIP → L20 cat
        ("L13",  256, 13, 14, "a"),          # 16x16 c64 (pool)
        # Deep
        ("L14",  256, 14, 15, "b"),
        ("L15",  256, 15, 16, "a"),
        ("L16",  256, 16, 17, "b"),
        ("L17",  256, 17, 18, "a"),
        ("L18",  256, 18, 19, "b"),
        ("L19", 1024, 19, 20, "a"),          # 16x16 c256 (depth_to_space → 32x32)
        # Decoder
        ("L20", 2048, 20, 21, "b"),          # 32x32 c128 (after cat with L12)
        ("L21", 1024, 21, 22, "a"),
        ("L22", 1024, 22, 23, "b"),
        ("L23", 2048, 23, 24, "a"),          # 32x32 c128 (depth_to_space → 64x64)
        ("L24", 4096, 24, 25, "b"),          # 64x64 c64 (after cat with L07)
        ("L25", 1024, 25, 26, "a"),
        ("L26", 1024, 26, 27, "b"),
        ("L27", 4096, 27, 28, "a"),          # 64x64 c64 (depth_to_space → 128x128)
        ("L28", 4096, 28, 29, "b"),          # 128x128 c16 (after cat with L04)
        ("L29", 8192, 29, 30, "a"),          # 128x128 c32 (depth_to_space → 256x256)
        ("L30", 4096, 30, 31, "b"),          # 256x256 c4 (after cat with L01)
        ("L31", 1024, 31, 32, "a"),          # 256x256 c1
    ]
    return [Tensor(name, size, d, l, buf) for name, size, d, l, buf in raw]


def peak_usage(placements: Dict[str, int], tensors: List[Tensor]) -> Dict[str, int]:
    max_layer = max(t.last_use for t in tensors)
    peaks: Dict[str, int] = {"a": 0, "b": 0}
    for layer in range(max_layer + 1):
        for buf in ("a", "b"):
            end = max(
                (placements[t.name] + t.size
                 for t in tensors
                 if t.buffer == buf and t.def_layer <= layer <= t.last_use
                 and t.name in placements),
                default=0
            )
            peaks[buf] = max(peaks[buf], end)
    return peaks


# ── Algorithm 1: Linear Scan (Poletto & Sarkar 1999) ─────────────────────────

def linear_scan(tensors: List[Tensor]) -> Tuple[Dict[str, int], float]:
    t0 = time.perf_counter()
    placements: Dict[str, int] = {}
    for buf in ("a", "b"):
        buf_t = sorted([t for t in tensors if t.buffer == buf], key=lambda t: t.def_layer)
        active: List[Tuple[int, int, int]] = []  # (start, end, last_use)
        for t in buf_t:
            active = [(s, e, lu) for s, e, lu in active if lu >= t.def_layer]
            active_sorted = sorted(active, key=lambda x: x[0])
            candidate = 0
            for s, e, lu in active_sorted:
                if candidate + t.size <= s:
                    break
                candidate = max(candidate, e)
            placements[t.name] = candidate
            active.append((candidate, candidate + t.size, t.last_use))
    return placements, (time.perf_counter() - t0) * 1000


# ── Algorithm 2: TVM Workspace (Best-Fit Decreasing) ─────────────────────────

def tvm_workspace(tensors: List[Tensor]) -> Tuple[Dict[str, int], float]:
    t0 = time.perf_counter()
    placements: Dict[str, int] = {}
    for buf in ("a", "b"):
        buf_t = sorted([t for t in tensors if t.buffer == buf], key=lambda t: -t.size)
        placed: List[Tuple[str, int, int, int, int]] = []  # name,start,end,def,last
        for t in buf_t:
            conflicts = sorted(
                [p for p in placed if not (p[4] < t.def_layer or t.last_use < p[3])],
                key=lambda p: p[1]
            )
            best_start, best_gap = 0, math.inf
            if not conflicts:
                best_start = 0
            else:
                gap = conflicts[0][1]
                if gap >= t.size and gap < best_gap:
                    best_start, best_gap = 0, gap
                for i in range(len(conflicts) - 1):
                    g_start = conflicts[i][2]
                    gap = conflicts[i+1][1] - g_start
                    if gap >= t.size and gap < best_gap:
                        best_start, best_gap = g_start, gap
                if best_gap == math.inf:
                    best_start = conflicts[-1][2]
            placements[t.name] = best_start
            placed.append((t.name, best_start, best_start + t.size, t.def_layer, t.last_use))
    return placements, (time.perf_counter() - t0) * 1000


# ── Algorithm 3: MLIR Bufferization (alias analysis + linear scan) ────────────

def mlir_bufferization(tensors: List[Tensor]) -> Tuple[Dict[str, int], float]:
    t0 = time.perf_counter()
    placements: Dict[str, int] = {}
    for buf in ("a", "b"):
        buf_t = sorted([t for t in tensors if t.buffer == buf], key=lambda t: t.def_layer)
        in_place: Dict[str, str] = {}
        for i, t in enumerate(buf_t):
            if i == 0:
                continue
            pred = buf_t[i - 1]
            # In-place: pred is "done" exactly when t starts, same size, not a skip
            if (pred.last_use == t.def_layer
                    and pred.size == t.size
                    and pred.last_use - pred.def_layer == 1):  # pred is not a skip
                in_place[t.name] = pred.name

        active: List[Tuple[int, int, int]] = []
        for t in buf_t:
            if t.name in in_place:
                root = t.name
                while root in in_place:
                    root = in_place[root]
                if root in placements:
                    placements[t.name] = placements[root]
                    continue
            active = [(s, e, lu) for s, e, lu in active if lu >= t.def_layer]
            active_sorted = sorted(active, key=lambda x: x[0])
            candidate = 0
            for s, e, lu in active_sorted:
                if candidate + t.size <= s:
                    break
                candidate = max(candidate, e)
            placements[t.name] = candidate
            active.append((candidate, candidate + t.size, t.last_use))
    return placements, (time.perf_counter() - t0) * 1000


# ── Theoretical optimal ───────────────────────────────────────────────────────

def theoretical_optimal(tensors: List[Tensor]) -> Dict[str, int]:
    """
    Hand-compute lower bound: at every layer, sum the sizes of simultaneously
    live tensors in each buffer. The true peak is this maximum sum.
    We use linear scan (which is optimal for this interval structure).
    placements, _ = linear_scan(tensors)
    """
    placements, _ = linear_scan(tensors)
    return placements


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark():
    tensors = build_unet_tensors()
    skips = [t for t in tensors if t.last_use - t.def_layer > 1]
    normals = [t for t in tensors if t.last_use - t.def_layer == 1]

    print("=" * 68)
    print("  Feature Buffer Allocation Benchmark  —  USR-Net (256x256 input)")
    print("=" * 68)
    print(f"  Total tensors: {len(tensors)}  "
          f"(skips: {len(skips)}, normals: {len(normals)})")
    print(f"  Skip tensors:")
    for t in sorted(skips, key=lambda t: t.def_layer):
        print(f"    {t.name:<8}  L{t.def_layer:02d}→L{t.last_use:02d}  "
              f"buf={t.buffer}  size={t.size:5d}w")
    print()

    # Theoretical optimal = what the hand-tuned golden aims for
    opt_place = theoretical_optimal(tensors)
    opt_peaks = peak_usage(opt_place, tensors)
    opt_total = opt_peaks["a"] + opt_peaks["b"]

    # Analytical lower bound: at peak layer, sum all live tensor sizes per buf
    # Peak-A moment: L01(8192)+L07(2048)+L27(4096) all live in buf-a simultaneously
    # (L27 def=27, L01 last=30, L07 last=24, so at L27: L01+L07+L27 alive in a)
    # Let's compute analytically
    max_layer = max(t.last_use for t in tensors)
    lb = {"a": 0, "b": 0}
    for layer in range(max_layer + 1):
        for buf in ("a", "b"):
            live_sum = sum(t.size for t in tensors
                           if t.buffer == buf and t.def_layer <= layer <= t.last_use)
            lb[buf] = max(lb[buf], live_sum)
    lb_total = lb["a"] + lb["b"]

    results = []
    for name, fn in [
        ("Linear Scan (Poletto & Sarkar 1999)", linear_scan),
        ("TVM Workspace (Best-Fit Decreasing)",  tvm_workspace),
        ("MLIR Bufferization (alias+linear)",    mlir_bufferization),
    ]:
        pl, elapsed = fn(tensors)
        peaks = peak_usage(pl, tensors)
        total = peaks["a"] + peaks["b"]
        results.append((name, peaks, total, elapsed))

    hdr = f"  {'Algorithm':<40} {'BufA':>6} {'BufB':>6} {'Total':>7}  {'vs LB':>10}  {'ms':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, peaks, total, elapsed in results:
        delta = total - lb_total
        pct = delta / lb_total * 100 if lb_total > 0 else 0
        print(f"  {name:<40} {peaks['a']:>6} {peaks['b']:>6} {total:>7}"
              f"  {delta:+6d}w ({pct:+.1f}%)  {elapsed:>5.2f}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'Analytical lower bound (min possible)':<40} "
          f"{lb['a']:>6} {lb['b']:>6} {lb_total:>7}  {'(reference)':>13}  {'—':>5}")
    print()

    # Peak breakdown per algorithm at the tightest moment
    print("  Peak-pressure analysis (buffer A, layer-by-layer live sum):")
    layer_sums_a = {}
    for layer in range(max_layer + 1):
        s = sum(t.size for t in tensors if t.buffer == "a"
                and t.def_layer <= layer <= t.last_use)
        if s >= lb["a"] * 0.8:
            live_names = [t.name for t in tensors if t.buffer == "a"
                          and t.def_layer <= layer <= t.last_use]
            print(f"    Layer {layer:2d}: {s:5d}w — {live_names}")


if __name__ == "__main__":
    run_benchmark()
