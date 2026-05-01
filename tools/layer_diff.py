"""
Per-layer field-level diff between our pseudo_instructions output and golden.

Strategy:
- Parse both files line-by-line via ast.literal_eval.
- Group each instruction into a "layer bucket" using layer_idx (DL/WL/QL) or
  the most-recent layer_idx for layerless instructions (DS, OffchipDataLoader,
  OffchipDataStorer, WeightLoader). WL has no `layer_idx` field but follows a
  DL within the same layer.
- The layer numbering schemes differ:
    * golden: DL/WL/QL layer_idx is contiguous 0..18 (DL) and 1..19 (QL).
    * ours:   layer_idx may have gaps (skipping concat-only fused layers).
  We canonicalise to a "logical layer" 0..18 by mapping in encounter order.
- Within each layer, we compare instructions as a multiset (Counter of
  filtered-field tuples) since golden uses interleaved scheduling and we use
  sequential — instruction order within a layer differs but the set should be
  identical.
- Skip set (scheduling-only fields): code_num, dependency, dest, src1, src2,
  src3, src4, is_offset, quant_config_idx. Address fields like bas_addr /
  base_addrs_res / base_addr_pooling are kept in the comparison because they
  are semantic correctness checks.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# Fields that legitimately differ between schedulings or come from post-passes
# we have not yet replicated. We MUST NOT include any address or template
# semantic field here.
#
# NOTE on WL is_new artifacts: WL is_new diffs that surface in the per-layer
# output are largely artifacts of the interleaved scheduling in the golden.
# When DLs from different layers are interleaved, the "most recent DL" heuristic
# used to bucket WL instructions assigns them to the wrong layer. This is NOT a
# real divergence in the emitter — kept un-skipped for visibility, but treat
# any WL is_new diff as suspect until the layer's full schedule is inspected.
SKIP_FIELDS = {
    "code_num",
    "dependency",
    "dest",
    "src1",
    "src2",
    "src3",
    "src4",
    "is_offset",          # post-pass, set by offset-fusion logic
    "quant_config_idx",   # quant config swap state, manager-driven
    "layer_idx",          # QL uses 0-based (ours) vs 1-based (archived golden);
                          # DL.layer_idx is the BUCKETING key, not a compared field
}


def parse_file(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ast.literal_eval(line))
    return out


def assign_logical_layers(insts: List[dict]) -> List[int]:
    """
    Walk the instruction stream and assign each inst a logical conv-layer
    index (0..N-1) based on the DataLoader layer_idx. DataLoader is the
    canonical anchor because both ours and golden number it identically:
    DL.layer_idx is 0-based and contiguous (with gaps in ours for
    fused-out concat layers — those gaps map to nothing).

    Bucketing rules:
      - DL: belongs to logical = (next-claimed slot for this layer_idx).
      - WL: inherits the most recent DL's logical (WL has no layer_idx but
        always follows the DL it pairs with).
      - DS: inherits the most recent DL's logical (DS comes after the
        WL that wrote it).
      - QL: in golden, QL.layer_idx = DL.layer_idx + 1 (1-based); in ours
        QL.layer_idx == DL.layer_idx (both 0-based). To unify, we associate
        each QL with the *next* DL's conv-layer in stream order (i.e. the
        QL "warms up" the next layer). For the very-first QL before any
        DL has been seen, we forward-search for the next DL.
      - OffchipDataLoader / OffchipDataStorer: inherit the most recent DL's
        logical (or -1 if none yet).

    This canonicalises the mapping between ours (sparse DL.layer_idx with
    gaps) and golden (contiguous DL.layer_idx 0..18), without depending on
    a hand-written table.
    """
    n = len(insts)
    logical = [-1] * n

    # Pre-pass 1: stream-order list of DL indices and their layer_idx.
    dl_indices: List[int] = [
        i for i, inst in enumerate(insts)
        if inst["op_code"] == "DataLoader" and "layer_idx" in inst
    ]

    # Pre-pass 2: claim a logical slot by sorted *value* of DL.layer_idx,
    # NOT by stream encounter order. Golden interleaves (e.g. emits
    # layer_idx=11 before 9/10), so encounter-order produces wrong buckets.
    # Both ours and golden have 19 distinct DL layer_idx values; mapping
    # each set's sorted order to 0..18 aligns the conv layers correctly:
    #   golden DL layer_idx: [0,1,2,3,...,18]            -> 0..18
    #   ours  DL layer_idx: [0,1,2,4,5,7,8,10,...,22]    -> 0..18
    distinct_li = sorted({insts[i]["layer_idx"] for i in dl_indices})
    layer_idx_to_logical: Dict[int, int] = {
        li: slot for slot, li in enumerate(distinct_li)
    }
    for i in dl_indices:
        logical[i] = layer_idx_to_logical[insts[i]["layer_idx"]]

    # Pre-pass 3: walk forward, propagating cur_logical for non-DL ops.
    # QL is special: it forward-binds to the *next* DL. All other ops
    # backward-bind to the most recent DL.
    cur_logical = -1
    for i, inst in enumerate(insts):
        op = inst["op_code"]
        if op == "DataLoader":
            cur_logical = logical[i]
            continue
        if op == "QuantLoader":
            # Forward-search for next DL.
            j = i + 1
            forward_logical = cur_logical
            while j < n:
                if insts[j]["op_code"] == "DataLoader" and "layer_idx" in insts[j]:
                    forward_logical = logical[j]
                    break
                j += 1
            logical[i] = forward_logical
        else:
            # WL / DS / Offchip* inherit cur_logical.
            logical[i] = cur_logical
    return logical


def filter_fields(inst: dict) -> Tuple:
    """Return a hashable tuple of (op_code, sorted (k,v) of non-skip fields)."""
    items = [(k, v) for k, v in inst.items() if k not in SKIP_FIELDS]
    items.sort()
    # Make value hashable (most are int/str; lists become tuples).
    items = [(k, tuple(v) if isinstance(v, list) else v) for k, v in items]
    return tuple(items)


def group_by_layer(insts: List[dict], logical: List[int]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = defaultdict(list)
    for inst, ll in zip(insts, logical):
        out[ll].append(inst)
    return out


def zero_addr_stats(layer_insts: List[dict]) -> Dict[str, Tuple[int, int]]:
    """Return {'DL_bas_addr': (zero_count, total), 'WL_bas_addr':..., ...}."""
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for inst in layer_insts:
        op = inst["op_code"]
        if op == "DataLoader":
            stats["DL_bas_addr"][1] += 1
            if inst.get("bas_addr", -1) == 0:
                stats["DL_bas_addr"][0] += 1
        elif op == "WeightLoader":
            stats["WL_bas_addr"][1] += 1
            if inst.get("bas_addr", -1) == 0:
                stats["WL_bas_addr"][0] += 1
        elif op == "QuantLoader":
            stats["QL_bas_addr"][1] += 1
            if inst.get("bas_addr", -1) == 0:
                stats["QL_bas_addr"][0] += 1
        elif op == "DataStorer":
            stats["DS_base_addrs_res"][1] += 1
            if inst.get("base_addrs_res", -1) == 0:
                stats["DS_base_addrs_res"][0] += 1
            stats["DS_base_addr_pooling"][1] += 1
            if inst.get("base_addr_pooling", -1) == 0:
                stats["DS_base_addr_pooling"][0] += 1
    return {k: tuple(v) for k, v in stats.items()}


def diff_layer(
    ours_layer: List[dict],
    golden_layer: List[dict],
    max_examples: int = 5,
) -> Tuple[int, List[Tuple[str, dict]]]:
    """
    Multiset diff. Return (num_diffs, examples) where examples is a list of
    (kind, payload) describing first few mismatches. Kind in
    {'only_in_ours', 'only_in_golden'}.
    """
    ours_c = Counter(filter_fields(i) for i in ours_layer)
    gold_c = Counter(filter_fields(i) for i in golden_layer)

    only_ours = ours_c - gold_c
    only_gold = gold_c - ours_c
    n = sum(only_ours.values()) + sum(only_gold.values())

    examples: List[Tuple[str, dict]] = []
    for tup, cnt in list(only_ours.items())[:max_examples]:
        examples.append(("only_in_ours", {"count": cnt, "inst": dict(tup)}))
    for tup, cnt in list(only_gold.items())[:max_examples]:
        examples.append(("only_in_golden", {"count": cnt, "inst": dict(tup)}))

    return n, examples


def diff_layer_pairwise(
    ours_layer: List[dict],
    golden_layer: List[dict],
    max_examples: int = 0,
) -> List[Dict[str, object]]:
    """
    Pair up "structurally similar" instructions across the two multisets and
    return the field-level deltas. Pairing is best-effort: for each
    only-in-ours instruction, find the only-in-golden instruction with the
    same op_code and most-matching fields. Useful for spotting "same shape,
    one field off" cases like a wrong bas_addr.

    `max_examples=0` means no cap on the number of pairs walked.
    """
    # Build per-op buckets of remaining (unmatched) fields.
    ours_only: List[dict] = []
    gold_only: List[dict] = []
    ours_c = Counter(filter_fields(i) for i in ours_layer)
    gold_c = Counter(filter_fields(i) for i in golden_layer)
    common = ours_c & gold_c
    ours_remaining = ours_c - common
    gold_remaining = gold_c - common

    def expand(c: Counter) -> List[dict]:
        out = []
        for tup, cnt in c.items():
            for _ in range(cnt):
                out.append(dict(tup))
        return out

    ours_only = expand(ours_remaining)
    gold_only = expand(gold_remaining)

    # Pair greedily by op_code + most matching fields.
    deltas: List[Dict[str, object]] = []
    used_g = [False] * len(gold_only)
    for o in ours_only:
        best_j = -1
        best_match = -1
        for j, g in enumerate(gold_only):
            if used_g[j]:
                continue
            if g.get("op_code") != o.get("op_code"):
                continue
            ks = set(o.keys()) | set(g.keys())
            m = sum(1 for k in ks if o.get(k) == g.get(k))
            if m > best_match:
                best_match = m
                best_j = j
        if best_j == -1:
            deltas.append({"kind": "ours_unmatched", "ours": o})
        else:
            used_g[best_j] = True
            g = gold_only[best_j]
            ks = set(o.keys()) | set(g.keys())
            diffs = {k: (o.get(k), g.get(k)) for k in ks if o.get(k) != g.get(k)}
            deltas.append({
                "kind": "field_diff",
                "op_code": o.get("op_code"),
                "diffs": diffs,
            })
        if max_examples and len(deltas) >= max_examples:
            break
    for j, g in enumerate(gold_only):
        if used_g[j]:
            continue
        if max_examples and len(deltas) >= max_examples:
            break
        deltas.append({"kind": "golden_unmatched", "golden": g})
    return deltas


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ours",
        default="/home/scratch.hansz_coreai/design/tvm-design/output/unet/pseudo_instructions.txt",
    )
    p.add_argument(
        "--golden",
        default="/home/hansz/scratch-data/design/tvm-tiling/golden/pseudo_code_load_next_mid.txt",
    )
    p.add_argument(
        "--mode",
        choices=("summary", "details", "zero", "all"),
        default="all",
        help="summary=count diffs; details=per-layer field deltas; zero=zero-address stats; all=everything",
    )
    p.add_argument("--max-examples", type=int, default=8)
    p.add_argument("--only-layer", type=int, default=None,
                   help="Restrict output to a single logical layer index")
    args = p.parse_args()

    ours = parse_file(args.ours)
    gold = parse_file(args.golden)
    o_logical = assign_logical_layers(ours)
    g_logical = assign_logical_layers(gold)
    o_groups = group_by_layer(ours, o_logical)
    g_groups = group_by_layer(gold, g_logical)

    layers = sorted(set(o_groups.keys()) | set(g_groups.keys()))

    if args.mode in ("zero", "all"):
        print("=" * 78)
        print("ZERO-ADDRESS STATS PER LOGICAL LAYER")
        print("=" * 78)
        for L in layers:
            if args.only_layer is not None and L != args.only_layer:
                continue
            os_ = zero_addr_stats(o_groups.get(L, []))
            gs_ = zero_addr_stats(g_groups.get(L, []))
            keys = sorted(set(os_.keys()) | set(gs_.keys()))
            line = f"layer L={L:<2}"
            for k in keys:
                oz, ot = os_.get(k, (0, 0))
                gz, gt = gs_.get(k, (0, 0))
                tag = "" if (oz == gz and ot == gt) else "  <-- DIFF"
                line += f"\n    {k:<22} ours {oz:>4}/{ot:<4}  golden {gz:>4}/{gt:<4}{tag}"
            print(line)

    if args.mode in ("summary", "details", "all"):
        print()
        print("=" * 78)
        print("PER-LAYER MULTISET DIFFS (skip set: %s)" % ", ".join(sorted(SKIP_FIELDS)))
        print("=" * 78)
        total_diffs = 0
        for L in layers:
            if args.only_layer is not None and L != args.only_layer:
                continue
            o_l = o_groups.get(L, [])
            g_l = g_groups.get(L, [])
            n, examples = diff_layer(o_l, g_l, max_examples=args.max_examples)
            total_diffs += n
            if n == 0:
                continue
            print(f"\nlayer L={L}: {n} diffs   (ours={len(o_l)}, golden={len(g_l)})")
            if args.mode in ("details", "all"):
                # Walk ALL pairs (max_examples=0) so the per-group counter
                # reflects real diff totals, not a sample.
                deltas = diff_layer_pairwise(o_l, g_l, max_examples=0)
                # Group field-diffs by (op_code, set of field names that differ)
                # to make the printout digestible.
                grouped: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
                samples: Dict[Tuple[str, Tuple[str, ...]], dict] = {}
                others: List[Dict[str, object]] = []
                for d in deltas:
                    if d.get("kind") != "field_diff":
                        others.append(d)
                        continue
                    op = d["op_code"]
                    diffs = d["diffs"]
                    key = (op, tuple(sorted(diffs.keys())))
                    grouped[key] += 1
                    if key not in samples:
                        samples[key] = diffs
                for (op, fnames), cnt in grouped.most_common():
                    print(f"    [{op}] {cnt}x  fields={list(fnames)}")
                    sample = samples[(op, fnames)]
                    for fn in fnames:
                        ours_v, gold_v = sample[fn]
                        print(f"        {fn}: ours={ours_v!r:<24} golden={gold_v!r}")
                for d in others[:3]:
                    print(f"    [unmatched] {d}")
        print()
        print(f"TOTAL field-level diffs across all layers: {total_diffs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
