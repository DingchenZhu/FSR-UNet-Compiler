"""
Datapath-level equivalence checker for ours vs golden pseudo_instructions.

Strength of the verdict: this tool proves that the two instruction streams
issue the same set of *hardware datapath* operations (loads/stores at the
same addresses, MAC arrays driven with the same shapes, quantizer fed with
the same parameters). It does NOT prove bit-accurate equivalence of the
final tensor — that requires either an ISA spec to reason about scheduling
state semantics, or RTL co-simulation. This boundary is documented in the
report below.

Normalisation strategy
----------------------
For each logical conv layer we partition the fields of every instruction into:
  - UNIVERSAL_SKIP: post-pass / metadata fields that legitimately differ
    (code_num, dependency, dest, src1..4, layer_idx, is_offset,
    quant_config_idx).
  - SCHEDULING_STATE_FIELDS_BY_OP: per-opcode fields that select which
    hardware *resource* a step uses (which accumulator register, which
    line-buffer half, which quant-config slot, whether an accumulator is
    reset on this cycle). Two schedulings of the same conv layer are
    permitted to differ in these fields.
  - DATAPATH_FIELDS_BY_OP: everything else — addresses, transfer counts,
    kernel shape, padding, parallel mode, output mode, etc. These MUST be
    multiset-equal between ours and golden for the verdict to be PASS.

Verdict per layer = (datapath multiset diff == 0). The CLI also reports the
scheduling-state field statistics (how many is_new=0 / =1 etc.) so the
remaining unexplained delta is visible to the reader.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# Fields that are post-pass metadata / dependency-graph artifacts. Universally
# skipped from the comparison regardless of opcode.
#
# `is_compression` / `offchip_read_mode` / `is_skip` are ISA-version-only
# fields: they were added to the pseudo-instruction format after the FSRCNN
# golden was archived, and within both networks they take only constant
# placeholder values (0, 0, 2 respectively). Their bit-widths are pending
# hardware ISA confirmation, so they are *not* part of the datapath
# comparison until that is resolved.
UNIVERSAL_SKIP_FIELDS = {
    "code_num",
    "dependency",
    "dest",
    "src1",
    "src2",
    "src3",
    "src4",
    "layer_idx",          # bucketing key, also numbered differently
    "is_offset",          # post-pass set by offset-fusion
    "quant_config_idx",   # quant config swap state, manager-driven
    "is_compression",     # ISA-version placeholder (constant 0 in both)
    "offchip_read_mode",  # ISA-version placeholder (constant 0 in both)
    "is_skip",            # ISA-version placeholder (constant 2 in ours)
}

# Fields per opcode that select hardware resources (registers, line-buffer
# halves, accumulator reset). Two valid schedulings of the same conv may
# differ in these without functionally diverging — but only the hardware
# spec / RTL can prove that the divergence is benign.
SCHEDULING_STATE_FIELDS_BY_OP: Dict[str, set] = {
    "WeightLoader":   {"is_new", "acc_reg_comp_idx", "line_buffer_idx"},
    "QuantLoader":    {"quant_reg_load_idx"},
    "DataStorer":     {"reg_out_idx", "pooling_out_new"},
    "OffchipDataLoader": set(),
    "DataLoader":     {"line_buffer_idx"},
    "OffsetLoader":   {"offset_reg_idx"},
    "OffchipDataStorer": set(),
}


def parse_file(path: str) -> List[dict]:
    """Parse a pseudo_instructions.txt: one Python dict literal per line."""
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
    Mirror layer_diff.py's logical-layer assignment so the bucket boundaries
    line up exactly with the existing tooling.
    """
    n = len(insts)
    logical = [-1] * n
    dl_indices = [
        i for i, inst in enumerate(insts)
        if inst["op_code"] == "DataLoader" and "layer_idx" in inst
    ]
    distinct_li = sorted({insts[i]["layer_idx"] for i in dl_indices})
    layer_idx_to_logical = {li: slot for slot, li in enumerate(distinct_li)}
    for i in dl_indices:
        logical[i] = layer_idx_to_logical[insts[i]["layer_idx"]]

    cur_logical = -1
    for i, inst in enumerate(insts):
        op = inst["op_code"]
        if op == "DataLoader":
            cur_logical = logical[i]
            continue
        if op == "QuantLoader":
            j = i + 1
            forward_logical = cur_logical
            while j < n:
                if insts[j]["op_code"] == "DataLoader" and "layer_idx" in insts[j]:
                    forward_logical = logical[j]
                    break
                j += 1
            logical[i] = forward_logical
        else:
            logical[i] = cur_logical
    return logical


def group_by_layer(insts: List[dict], logical: List[int]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = defaultdict(list)
    for inst, ll in zip(insts, logical):
        out[ll].append(inst)
    return out


def datapath_tuple(inst: dict) -> Tuple:
    """
    Return a hashable tuple of (op_code, sorted (k,v)) keeping ONLY the
    hardware datapath fields. Scheduling-state and post-pass fields are
    dropped so two valid schedulings of the same conv compare equal.
    """
    op = inst.get("op_code", "?")
    drop = UNIVERSAL_SKIP_FIELDS | SCHEDULING_STATE_FIELDS_BY_OP.get(op, set())
    items = [(k, v) for k, v in inst.items() if k not in drop]
    items.sort()
    items = [(k, tuple(v) if isinstance(v, list) else v) for k, v in items]
    return tuple(items)


def datapath_multiset(insts: List[dict]) -> Counter:
    return Counter(datapath_tuple(i) for i in insts)


def datapath_diff(
    ours: List[dict], golden: List[dict]
) -> Tuple[int, List[dict], List[dict]]:
    """
    Return (n_diff, only_in_ours_examples, only_in_golden_examples).
    n_diff is the *signed* multiset symmetric-difference cardinality
    (sum of |ours - golden| + |golden - ours|).
    """
    o = datapath_multiset(ours)
    g = datapath_multiset(golden)
    only_o = o - g
    only_g = g - o
    n = sum(only_o.values()) + sum(only_g.values())
    only_ours_examples = [
        {"count": cnt, "inst": dict(t)} for t, cnt in list(only_o.most_common())
    ]
    only_golden_examples = [
        {"count": cnt, "inst": dict(t)} for t, cnt in list(only_g.most_common())
    ]
    return n, only_ours_examples, only_golden_examples


def scheduling_state_summary(insts: List[dict]) -> Dict[str, Dict[Any, int]]:
    """
    Per-opcode histogram of scheduling-state field values within one layer.
    Used to report what *did* differ between ours and golden, separated from
    the datapath verdict.
    """
    out: Dict[str, Dict[Any, int]] = defaultdict(Counter)
    for inst in insts:
        op = inst.get("op_code", "?")
        for f in SCHEDULING_STATE_FIELDS_BY_OP.get(op, set()):
            if f in inst:
                key = f"{op}.{f}={inst[f]!r}"
                out[op][key] += 1
    return {k: dict(v) for k, v in out.items()}


def compare(
    ours_path: str,
    golden_path: str,
    *,
    only_layer: Optional[int] = None,
    max_diff_examples: int = 5,
) -> Dict[str, Any]:
    ours = parse_file(ours_path)
    gold = parse_file(golden_path)
    o_logical = assign_logical_layers(ours)
    g_logical = assign_logical_layers(gold)
    o_groups = group_by_layer(ours, o_logical)
    g_groups = group_by_layer(gold, g_logical)

    layers = sorted(set(o_groups.keys()) | set(g_groups.keys()))
    if only_layer is not None:
        layers = [L for L in layers if L == only_layer]

    per_layer: List[Dict[str, Any]] = []
    total_datapath_diff = 0
    pass_layers = 0
    fail_layers = 0

    for L in layers:
        if L < 0:
            continue
        o_l = o_groups.get(L, [])
        g_l = g_groups.get(L, [])
        n_diff, only_o, only_g = datapath_diff(o_l, g_l)
        total_datapath_diff += n_diff
        is_pass = (n_diff == 0)
        if is_pass:
            pass_layers += 1
        else:
            fail_layers += 1
        per_layer.append({
            "layer": L,
            "ours_count": len(o_l),
            "golden_count": len(g_l),
            "datapath_diff": n_diff,
            "verdict": "PASS" if is_pass else "FAIL",
            "only_in_ours": only_o[:max_diff_examples],
            "only_in_golden": only_g[:max_diff_examples],
            "scheduling_state_ours": scheduling_state_summary(o_l),
            "scheduling_state_golden": scheduling_state_summary(g_l),
        })

    overall_verdict = "DATAPATH_EQUIVALENT" if total_datapath_diff == 0 else "DATAPATH_DIVERGENT"

    return {
        "ours_path": ours_path,
        "golden_path": golden_path,
        "layers_checked": len(per_layer),
        "layers_pass": pass_layers,
        "layers_fail": fail_layers,
        "total_datapath_diff": total_datapath_diff,
        "overall_verdict": overall_verdict,
        "per_layer": per_layer,
        "scheduling_state_skipped": {
            op: sorted(fs) for op, fs in SCHEDULING_STATE_FIELDS_BY_OP.items() if fs
        },
        "universal_skip": sorted(UNIVERSAL_SKIP_FIELDS),
    }


def render_terminal(report: Dict[str, Any], *, verbose: bool = False) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("DATAPATH EQUIVALENCE CHECK")
    lines.append("=" * 78)
    lines.append(f"  ours:   {report['ours_path']}")
    lines.append(f"  golden: {report['golden_path']}")
    lines.append("")
    lines.append("  Scheduling-state fields stripped (per op):")
    for op, fs in report["scheduling_state_skipped"].items():
        lines.append(f"    {op:<20} {fs}")
    lines.append(f"  Universal skip fields: {report['universal_skip']}")
    lines.append("")
    lines.append(f"  Layers checked:  {report['layers_checked']}")
    lines.append(f"  Layers PASS:     {report['layers_pass']}")
    lines.append(f"  Layers FAIL:     {report['layers_fail']}")
    lines.append(f"  Total datapath diff:  {report['total_datapath_diff']}")
    lines.append("")
    lines.append("  PER-LAYER VERDICT")
    lines.append("  " + "-" * 60)
    for row in report["per_layer"]:
        L = row["layer"]
        v = row["verdict"]
        n = row["datapath_diff"]
        oc = row["ours_count"]
        gc = row["golden_count"]
        lines.append(f"    L={L:<2}  ours={oc:<5} golden={gc:<5}  datapath_diff={n:<5}  [{v}]")
        if verbose and not row["verdict"] == "PASS":
            for ex in row["only_in_ours"][:3]:
                lines.append(f"        only_in_ours x{ex['count']}: {ex['inst']}")
            for ex in row["only_in_golden"][:3]:
                lines.append(f"        only_in_golden x{ex['count']}: {ex['inst']}")
    lines.append("")
    verdict = report["overall_verdict"]
    if verdict == "DATAPATH_EQUIVALENT":
        lines.append("  +" + "-" * 60 + "+")
        lines.append("  | OVERALL: DATAPATH EQUIVALENT  (PASS)              |")
        lines.append("  +" + "-" * 60 + "+")
        lines.append("")
        lines.append("  Note on verdict strength")
        lines.append("  ------------------------")
        lines.append("  This proves the two streams issue the same set of hardware")
        lines.append("  datapath operations (addresses / shapes / transfer counts /")
        lines.append("  output modes are multiset-equal). Scheduling-state fields")
        lines.append("  (is_new, ping-pong reg, quant_reg_load_idx) differ between")
        lines.append("  schedulings; their semantic equivalence requires HW spec or")
        lines.append("  RTL co-sim, which is outside the scope of this check.")
    else:
        lines.append("  +" + "-" * 60 + "+")
        lines.append("  | OVERALL: DATAPATH DIVERGENT  (FAIL)               |")
        lines.append("  +" + "-" * 60 + "+")
        lines.append("")
        lines.append("  Re-run with --verbose to see per-layer mismatch examples.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Datapath-level equivalence check between ours and golden."
    )
    p.add_argument(
        "--ours",
        default="/home/scratch.hansz_coreai/design/tvm-design/output/unet/pseudo_instructions.txt",
        help="path to ours pseudo_instructions.txt",
    )
    p.add_argument(
        "--golden",
        default="/home/hansz/scratch-data/design/tvm-tiling/golden/pseudo_code_load_next_mid.txt",
        help="path to golden pseudo_instructions.txt",
    )
    p.add_argument("--only-layer", type=int, default=None)
    p.add_argument("--max-diff-examples", type=int, default=5)
    p.add_argument("--output-json", default=None,
                   help="if set, write the structured report here")
    p.add_argument("--verbose", action="store_true",
                   help="print per-layer mismatch examples on FAIL")
    args = p.parse_args()

    report = compare(
        args.ours,
        args.golden,
        only_layer=args.only_layer,
        max_diff_examples=args.max_diff_examples,
    )
    print(render_terminal(report, verbose=args.verbose))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Wrote JSON report: {args.output_json}\n")

    return 0 if report["overall_verdict"] == "DATAPATH_EQUIVALENT" else 1


if __name__ == "__main__":
    sys.exit(main())
