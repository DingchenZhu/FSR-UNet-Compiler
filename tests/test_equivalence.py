"""
Pytest regression for tools/equivalence_check.py.

Two layers of testing:

1. Unit tests for the normalisation rules (datapath_tuple): given small
   handcrafted instructions, assert that scheduling-state fields are
   stripped and datapath fields are kept.

2. End-to-end on actual ours / golden files:
     - SD-UNet: must be DATAPATH_EQUIVALENT (19/19 layers PASS).
     - FSRCNN: must be DATAPATH_DIVERGENT with the *known* placement-shift
       pattern (WL bas_addr fixed offset per layer). The exact diff count
       is pinned so that any new divergence flips the test red.
"""
from __future__ import annotations

import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from tools.equivalence_check import (
    UNIVERSAL_SKIP_FIELDS,
    SCHEDULING_STATE_FIELDS_BY_OP,
    compare,
    datapath_tuple,
)


UNET_OURS = os.path.join(REPO_ROOT, "output", "unet", "pseudo_instructions.txt")
UNET_GOLDEN = "/home/hansz/scratch-data/design/tvm-tiling/golden/pseudo_code_load_next_mid.txt"
FSRCNN_OURS = os.path.join(REPO_ROOT, "output", "fsrcnn", "pseudo_instructions.txt")
FSRCNN_GOLDEN = "/home/hansz/scratch-data/design/tvm-tiling/references/sr_inst_golden.txt"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_universal_skip_includes_post_pass_fields():
    """Post-pass / dependency fields must be skipped."""
    for f in ["code_num", "dependency", "dest", "src1", "src2", "src3", "src4",
              "is_offset", "quant_config_idx", "layer_idx",
              "is_compression", "offchip_read_mode", "is_skip"]:
        assert f in UNIVERSAL_SKIP_FIELDS, f"{f} must be universally skipped"


def test_weightloader_scheduling_state_stripped():
    """is_new / acc_reg_comp_idx / line_buffer_idx must drop out of the tuple."""
    assert "is_new" in SCHEDULING_STATE_FIELDS_BY_OP["WeightLoader"]
    assert "acc_reg_comp_idx" in SCHEDULING_STATE_FIELDS_BY_OP["WeightLoader"]
    assert "line_buffer_idx" in SCHEDULING_STATE_FIELDS_BY_OP["WeightLoader"]


def test_weightloader_datapath_fields_retained():
    """bas_addr / kernal_size / transnum must remain in the datapath tuple."""
    a = {
        "op_code": "WeightLoader", "code_num": [0], "dependency": [],
        "is_new": 0, "acc_reg_comp_idx": 0, "line_buffer_idx": 0,
        "bas_addr": 100, "kernal_size": 1, "transnum": 9,
        "weight_parall_mode": 0, "is_padding_col": 1,
        "line_buffer_row_shift": 0, "is_bilinear_bicubic": 0,
        "offset_reg_idx": 0, "is_skip": 2,
        "dest": 1, "src1": 2, "src2": 0, "src3": 0, "src4": 0,
    }
    b = {
        "op_code": "WeightLoader", "code_num": [99], "dependency": [3],
        "is_new": 1, "acc_reg_comp_idx": 1, "line_buffer_idx": 1,
        "bas_addr": 100, "kernal_size": 1, "transnum": 9,
        "weight_parall_mode": 0, "is_padding_col": 1,
        "line_buffer_row_shift": 0, "is_bilinear_bicubic": 0,
        "offset_reg_idx": 0, "is_skip": 2,
        "dest": 7, "src1": 5, "src2": 6, "src3": 0, "src4": 0,
    }
    # Two WLs differing ONLY in scheduling-state fields must be datapath-equal.
    assert datapath_tuple(a) == datapath_tuple(b)


def test_weightloader_bas_addr_diff_breaks_equality():
    """Changing bas_addr is a real datapath diff and must show up."""
    a = {"op_code": "WeightLoader", "bas_addr": 100, "kernal_size": 1,
         "transnum": 9, "is_new": 0}
    b = {"op_code": "WeightLoader", "bas_addr": 200, "kernal_size": 1,
         "transnum": 9, "is_new": 0}
    assert datapath_tuple(a) != datapath_tuple(b)


def test_quantloader_quant_reg_load_idx_stripped():
    """QL.quant_reg_load_idx must be stripped (proven non-functional)."""
    a = {"op_code": "QuantLoader", "quant_reg_load_idx": 0,
         "quant_mode": 3, "transnum": 32, "bas_addr": 0}
    b = {"op_code": "QuantLoader", "quant_reg_load_idx": 1,
         "quant_mode": 3, "transnum": 32, "bas_addr": 0}
    assert datapath_tuple(a) == datapath_tuple(b)


def test_datastorer_pool_addr_kept():
    """DS.base_addr_pooling and base_addrs_res must remain in the tuple."""
    a = {"op_code": "DataStorer", "reg_out_idx": 0, "pooling_out_new": 0,
         "base_addrs_res": 0, "base_addr_pooling": 0, "transfer_num": 1}
    b = {"op_code": "DataStorer", "reg_out_idx": 0, "pooling_out_new": 0,
         "base_addrs_res": 64, "base_addr_pooling": 0, "transfer_num": 1}
    assert datapath_tuple(a) != datapath_tuple(b)


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.isfile(UNET_OURS) and os.path.isfile(UNET_GOLDEN)),
    reason="SD-UNet ours/golden files not available",
)
def test_unet_datapath_equivalent():
    """SD-UNet must produce DATAPATH_EQUIVALENT verdict (19/19 layers PASS)."""
    report = compare(UNET_OURS, UNET_GOLDEN)
    assert report["overall_verdict"] == "DATAPATH_EQUIVALENT", (
        f"SD-UNet datapath divergence detected: "
        f"{report['layers_fail']} layers failed, "
        f"{report['total_datapath_diff']} total datapath diffs. "
        "This is a regression — the compiler now emits hardware datapath "
        "instructions that no longer multiset-match golden."
    )
    assert report["layers_pass"] == 19
    assert report["layers_fail"] == 0
    assert report["total_datapath_diff"] == 0


@pytest.mark.skipif(
    not (os.path.isfile(FSRCNN_OURS) and os.path.isfile(FSRCNN_GOLDEN)),
    reason="FSRCNN ours/golden files not available",
)
def test_fsrcnn_known_placement_divergence():
    """FSRCNN diverges due to weight-placement offset (placement-only).

    The earlier project memory claimed "0 functional diffs" for FSRCNN,
    but this checker exposes a real datapath divergence: WeightLoader
    bas_addr is shifted by a fixed per-layer offset (-576 / -792 / +320)
    relative to the golden. This is most likely an addr_alloc choice
    difference (placement-only, not a runtime bug) but it is now
    surfaced explicitly so future regressions cannot hide behind the
    incorrect "0 diff" headline.

    The expected total diff (1410) is pinned. If a future change either
    fixes the placement (reducing diff) or introduces a *new* divergence
    (increasing diff) this test will fail and force a deliberate update.
    """
    report = compare(FSRCNN_OURS, FSRCNN_GOLDEN)
    assert report["overall_verdict"] == "DATAPATH_DIVERGENT", (
        "FSRCNN unexpectedly converged — this is good news but the test "
        "needs to be updated to assert DATAPATH_EQUIVALENT instead."
    )
    # Layer 0 had 0 diff in our run; pin that.
    layer0 = next(r for r in report["per_layer"] if r["layer"] == 0)
    assert layer0["datapath_diff"] == 0
    # Pin the total to detect new regressions.
    assert report["total_datapath_diff"] == 1410, (
        f"FSRCNN datapath diff drifted: was 1410, now "
        f"{report['total_datapath_diff']}. Investigate before updating."
    )


@pytest.mark.skipif(
    not (os.path.isfile(UNET_OURS) and os.path.isfile(UNET_GOLDEN)),
    reason="SD-UNet ours/golden files not available",
)
def test_unet_per_layer_counts_match():
    """SD-UNet ours and golden must have identical per-layer instruction counts."""
    report = compare(UNET_OURS, UNET_GOLDEN)
    for row in report["per_layer"]:
        assert row["ours_count"] == row["golden_count"], (
            f"Layer {row['layer']} count mismatch: ours={row['ours_count']} "
            f"vs golden={row['golden_count']}"
        )
