"""Auto-Tiling generalization — Phase 1 demo.

Two new modules that DO NOT modify the existing legacy table-based logic in
`tiling/tiling.py` (`choose_tiling`, `_UNET_LAYER_TABLE`):

  1. `infer_template_params(layer, hw_spec) -> dict`
        Derives the obvious / mechanically-deducible TilingPlan fields from
        the LayerDesc + hardware spec (no template selection — just direct
        derivations).

  2. `TilingConstraintChecker.check(layer, plan) -> List[ConstraintViolation]`
        Validates a (LayerDesc, TilingPlan) pair against a set of basic
        legality rules. Used to sanity-check both auto-derived and table-
        driven plans.

This is a Phase-1 scaffolding module. It is intentionally narrow: only the
clearly mechanical fields are inferred here. Template-specific decisions
(line_buffer_reshape, weight_parall_mode, ky_outer, etc.) remain in
`choose_tiling` and the `_UNET_LAYER_TABLE` shape map, and will be migrated
in subsequent phases as the constraint surface is fleshed out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ir.layer_desc import LayerDesc


# ---------------------------------------------------------------------------
# Hardware spec — mechanical constants that bound auto-tiling choices.
# ---------------------------------------------------------------------------
@dataclass
class HardwareSpec:
    """SDSR hardware constants relevant to tiling field derivation."""

    MAC_LANES: int = 8
    LINE_BUFFER_ROWS_MAX: int = 4
    LINE_BUFFER_CAPACITY_PIXELS: int = 1024
    WEIGHT_SRAM_SLOTS: int = 3
    W_MACRO_TILE_MAX: int = 128


# ---------------------------------------------------------------------------
# Constraint violation record — emitted by TilingConstraintChecker.
# ---------------------------------------------------------------------------
@dataclass
class ConstraintViolation:
    """One legality issue found in a (LayerDesc, TilingPlan) pair."""

    field: str
    message: str
    severity: str  # "ERROR" or "WARNING"

    def __str__(self) -> str:  # readable in self-test prints
        return f"[{self.severity}] {self.field}: {self.message}"


# ---------------------------------------------------------------------------
# Module 1: infer_template_params
# ---------------------------------------------------------------------------
# Candidate ladders for h_out_per_step and cin_group. Listed largest-first;
# the picker walks them and returns the first divisor-compatible value.
_H_OUT_CANDIDATES = (4, 2, 1)
_CIN_GROUP_CANDIDATES = (16, 8, 4, 2, 1)


def _pick_h_out_per_step(h_in: int) -> int:
    """Pick h_out_per_step from {4,2,1} preferring the largest divisor of h_in.

    Falls back to h_in itself only when h_in < 1, which should never happen
    for valid layers; in practice the {4,2,1} ladder always yields 1 as a
    divisor of any positive h_in.
    """
    for cand in _H_OUT_CANDIDATES:
        if h_in % cand == 0:
            return cand
    return h_in


def _pick_cin_group(cin_per_group: int, hw_spec: HardwareSpec) -> int:
    """Pick the largest cin_group from {16,8,4,2,1} that divides cin_per_group.

    Bounded above by the MAC array width — values larger than MAC_LANES
    cannot be packed in a single MAC lane group regardless of divisibility.
    """
    upper = max(1, min(cin_per_group, hw_spec.MAC_LANES))
    for cand in _CIN_GROUP_CANDIDATES:
        if cand <= upper and cin_per_group % cand == 0:
            return cand
    return 1  # always divides


def infer_template_params(
    layer: LayerDesc,
    hw_spec: Optional[HardwareSpec] = None,
) -> Dict[str, Any]:
    """Derive mechanical TilingPlan fields from a LayerDesc + HardwareSpec.

    Returns a dict with keys:
      - weight_transnum_base
      - h_out_per_step
      - cin_group
      - load_total_num

    For deformable / offset_gen layers, weight_transnum_base is fixed at 24
    (bilinear WL) — see HardwareSpec / golden sd_sr_codegen. Other fields use
    the same divisor-based picker as standard conv.
    """
    if hw_spec is None:
        hw_spec = HardwareSpec()

    h_in = layer.h_in
    cin = layer.cin
    groups = max(1, layer.groups)
    cin_per_group = cin // groups
    k_h = layer.k_h

    # weight_transnum_base.
    #   * deformable / offset_gen: fixed at 24 (bilinear WL spec).
    #   * standard conv:           k_h * k_h * (cin // groups).
    if layer.op == "offset_gen" or layer.deformable:
        weight_transnum_base = 24
    else:
        weight_transnum_base = k_h * k_h * cin_per_group

    h_out_per_step = _pick_h_out_per_step(h_in)
    cin_group = _pick_cin_group(cin_per_group, hw_spec)

    # load_total_num is rows-per-macro-tile along H. Use ceil so non-divisible
    # H still produces a valid block count; divisible H comes out exact.
    load_total_num = math.ceil(h_in / h_out_per_step) if h_out_per_step > 0 else 0

    return {
        "weight_transnum_base": weight_transnum_base,
        "h_out_per_step": h_out_per_step,
        "cin_group": cin_group,
        "load_total_num": load_total_num,
    }


# ---------------------------------------------------------------------------
# Module 2: TilingConstraintChecker
# ---------------------------------------------------------------------------
class TilingConstraintChecker:
    """Validates a (LayerDesc, TilingPlan) pair against legality rules.

    The checker is stateless: each `check()` call returns a fresh list of
    `ConstraintViolation` records. ERROR-severity violations are guarantees
    of incorrect codegen; WARNING-severity ones flag suspicious-but-not-
    necessarily-fatal cases (e.g. a hand-tuned override that diverges from
    the obvious mechanical formula).
    """

    def __init__(self, hw_spec: Optional[HardwareSpec] = None) -> None:
        self.hw_spec = hw_spec or HardwareSpec()

    def check(self, layer: LayerDesc, plan: Any) -> List[ConstraintViolation]:
        """Run all rules. `plan` is duck-typed (TilingPlan or compatible)."""
        violations: List[ConstraintViolation] = []
        violations.extend(self._check_h_out_divisibility(layer, plan))
        violations.extend(self._check_cin_group_divisibility(layer, plan))
        violations.extend(self._check_weight_transnum_base(layer, plan))
        violations.extend(self._check_line_buffer_capacity(layer, plan))
        return violations

    # -- individual rules ---------------------------------------------------

    def _check_h_out_divisibility(
        self,
        layer: LayerDesc,
        plan: Any,
    ) -> List[ConstraintViolation]:
        """Rule 1: h_out_per_step must divide h_in OR plan must carry a
        last_step_transnum field that handles the residue."""
        h_step = getattr(plan, "h_out_per_step", None)
        if not h_step or h_step <= 0:
            return []
        if layer.h_in % h_step == 0:
            return []
        # Residue mode: legal if plan declares last_step_transnum.
        last_step = getattr(plan, "last_step_transnum", None)
        if last_step is not None and last_step != 0:
            return []
        return [ConstraintViolation(
            field="h_out_per_step",
            message=(
                f"h_out_per_step={h_step} does not divide h_in={layer.h_in} "
                f"and plan has no last_step_transnum to handle the remainder."
            ),
            severity="ERROR",
        )]

    def _check_cin_group_divisibility(
        self,
        layer: LayerDesc,
        plan: Any,
    ) -> List[ConstraintViolation]:
        """Rule 2: (cin // groups) must be divisible by cin_group."""
        cin_group = getattr(plan, "cin_group", None)
        if not cin_group or cin_group <= 0:
            return []
        groups = max(1, layer.groups)
        cin_per_group = layer.cin // groups
        if cin_per_group == 0:
            return []  # degenerate; not this rule's concern
        if cin_per_group % cin_group != 0:
            return [ConstraintViolation(
                field="cin_group",
                message=(
                    f"cin_group={cin_group} does not divide "
                    f"cin/groups={layer.cin}/{groups}={cin_per_group}."
                ),
                severity="ERROR",
            )]
        return []

    def _check_weight_transnum_base(
        self,
        layer: LayerDesc,
        plan: Any,
    ) -> List[ConstraintViolation]:
        """Rule 3: weight_transnum_base ?= k_h * k_h * (cin//groups) for
        standard conv. Skipped for deformable / offset_gen (bilinear WL has
        its own fixed 24-byte block size).
        """
        if layer.op == "offset_gen" or layer.deformable:
            return []
        wt = getattr(plan, "weight_transnum_base", None)
        if wt is None:
            return []
        groups = max(1, layer.groups)
        expected = layer.k_h * layer.k_h * (layer.cin // groups)
        if wt != expected:
            return [ConstraintViolation(
                field="weight_transnum_base",
                message=(
                    f"weight_transnum_base={wt} differs from mechanical "
                    f"k_h*k_h*(cin/groups)={layer.k_h}*{layer.k_h}*"
                    f"({layer.cin}/{groups})={expected}."
                ),
                severity="WARNING",
            )]
        return []

    def _check_line_buffer_capacity(
        self,
        layer: LayerDesc,
        plan: Any,
    ) -> List[ConstraintViolation]:
        """Rule 4: h_out_per_step * line_buffer_rows fits in the line buffer.

        The line buffer holds LINE_BUFFER_CAPACITY_PIXELS pixels total; one
        row consumes `w_in` pixels. So the budget for stacked rows is
        capacity // w_in.
        """
        rows = getattr(plan, "line_buffer_rows", None)
        h_step = getattr(plan, "h_out_per_step", None)
        if rows is None or h_step is None:
            return []
        if layer.w_in <= 0:
            return []
        budget = self.hw_spec.LINE_BUFFER_CAPACITY_PIXELS // layer.w_in
        used = h_step * rows
        if used > budget:
            return [ConstraintViolation(
                field="line_buffer_rows",
                message=(
                    f"h_out_per_step({h_step}) * line_buffer_rows({rows})="
                    f"{used} exceeds line-buffer budget "
                    f"{self.hw_spec.LINE_BUFFER_CAPACITY_PIXELS}//w_in("
                    f"{layer.w_in})={budget}."
                ),
                severity="WARNING",
            )]
        return []


# ---------------------------------------------------------------------------
# Self-test: run on a few representative SD-UNet layers and print results.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Direct invocation (`python tiling/auto_tiling.py`) puts the script's
    # own directory at sys.path[0], shadowing the `tiling` package. Push the
    # repo root in front so `from tiling.tiling import ...` resolves as a
    # package. Running via `python -m tiling.auto_tiling` works without the
    # shim — both invocations are supported.
    import os
    import sys
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _repo_root)

    # Pull in the existing legacy table + plan factory so we can validate the
    # current production plans, not just synthetic ones.
    from tiling.tiling import _UNET_LAYER_TABLE, choose_tiling  # noqa: E402

    hw = HardwareSpec()
    checker = TilingConstraintChecker(hw)

    # Representative layers: encoder front (cin=1), encoder mid (g=2), and
    # encoder bottleneck (g=8).
    sample_layers = [
        # name, LayerDesc(...) — only the fields used by the checker.
        ("conv1",  LayerDesc(op="conv2d", idx=0,
                             h_in=144, w_in=256, cin=1,  cout=4,
                             k_h=3, k_w=3, groups=1)),
        ("conv6",  LayerDesc(op="conv2d", idx=7,
                             h_in=18,  w_in=32,  cin=16, cout=64,
                             k_h=3, k_w=3, groups=2)),
        ("conv10", LayerDesc(op="conv2d", idx=10,
                             h_in=9,   w_in=16,  cin=64, cout=256,
                             k_h=3, k_w=3, groups=8)),
    ]

    for name, layer in sample_layers:
        print("=" * 72)
        print(f"Layer {name}: h={layer.h_in} w={layer.w_in} "
              f"cin={layer.cin} cout={layer.cout} "
              f"k={layer.k_h} g={layer.groups}")
        print("-" * 72)

        # 1) Mechanical inference.
        derived = infer_template_params(layer, hw)
        print("infer_template_params() →")
        for key, val in derived.items():
            print(f"  {key:24s} = {val}")

        # 2) Validate the legacy table entry (when present) AND the plan
        #    that choose_tiling() actually produces.
        key = (layer.h_in, layer.w_in, layer.cin, layer.cout,
               layer.k_h, layer.groups)
        table_entry = _UNET_LAYER_TABLE.get(key)
        print(f"\n_UNET_LAYER_TABLE[{key}] present: {table_entry is not None}")
        if table_entry is not None:
            shown = {k: table_entry[k] for k in (
                "h_out_per_step", "cin_group", "weight_transnum_base",
            ) if k in table_entry}
            print(f"  table fields of interest: {shown}")

        plan = choose_tiling(layer, tile_h=None)  # SD-UNet full-height mode
        print(f"\nchoose_tiling() → plan summary:")
        print(f"  h_out_per_step       = {plan.h_out_per_step}")
        print(f"  cin_group            = {plan.cin_group}")
        print(f"  weight_transnum_base = {plan.weight_transnum_base}")
        print(f"  load_total_num       = {plan.load_total_num}")
        print(f"  line_buffer_rows     = {plan.line_buffer_rows}")
        last_step = getattr(plan, "last_step_transnum", None)
        print(f"  last_step_transnum   = {last_step}")

        # 3) Run the constraint checker on the produced plan.
        violations = checker.check(layer, plan)
        print(f"\nTilingConstraintChecker.check(): {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v}")

    print("=" * 72)
    print("auto_tiling.py self-test complete.")
