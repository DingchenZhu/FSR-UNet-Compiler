"""OffsetGenerator subgraph fusion pass.

OffsetGenerator (models_new_930.py) = AvgPool2d(reuse) + Conv2d(C, 2*9, 3, padding=1).
After relay.frontend.from_pytorch, this appears in the LayerDesc list as:
    pool2d (k=reuse×reuse, cin=C) → conv2d (cout=18) → deformable_conv2d

The pool2d + conv2d pair is fused into a single op='offset_gen' LayerDesc.
This triggers a dedicated hardware template that writes results to dest_buffer_idx='offset_reg'
instead of buffer a/b — the offset register consumed directly by the subsequent OffsetLoader.

Without this pass:
  - pool2d emits a PseudoOp (hardware ignores it)
  - conv2d emits a generic standard-conv sequence writing to buffer a (wrong destination)
  - OffsetLoader reads stale/uninitialized offset_reg data

With this pass:
  - offset_gen emits QuantLoader + 3×(DataLoader + WeightLoader) + DataStorer(dest='offset_reg')
  - OffsetLoader reads freshly computed offsets from offset_reg

Position in pipeline:  after extract_layer_descs(), before plan_all().
"""
from __future__ import annotations

from typing import List

from ir.layer_desc import LayerDesc

# Historical note: for a 3×3 deformable kernel the offset-gen conv has cout=18
# (2 × 3 × 3). For a 5×5 deformable kernel it would be cout=50 (2 × 5 × 5).
# We no longer hardcode 18 — the expected cout is derived from the following
# deformable_conv2d's kernel size so other kernels fuse correctly.
_OFFSET_GEN_COUT_3x3 = 18   # kept only for documentation / legacy reference


def fuse_offset_generators(layers: List[LayerDesc]) -> List[LayerDesc]:
    """Replace each (pool2d + conv2d) pair preceding a deformable_conv2d with a
    single op='offset_gen' LayerDesc. Layer indices are reassigned after fusion.

    Recognition rule (structural, not magic-constant based):
        layers[i].op   == 'pool2d'
        layers[i+1].op == 'conv2d'
        layers[i+2].op == 'deformable_conv2d'
        layers[i+1].cout == 2 * layers[i+2].k_h * layers[i+2].k_w
            (each deformable sample needs an (x, y) offset, hence the factor 2)

    If a pool2d+conv2d pair appears with the expected cout shape but no
    deformable_conv2d follows, we emit a warning and skip fusion so callers
    can investigate instead of silently generating wrong code.

    Returned list has the conv2d entry removed; pool2d becomes offset_gen.
    Indices are renumbered sequentially starting from 0.
    """
    n = len(layers)
    fused: List[LayerDesc] = []
    i = 0
    while i < n:
        L = layers[i]
        if (
            L.op == "pool2d"
            and i + 2 < n
            and layers[i + 1].op == "conv2d"
            and layers[i + 2].op == "deformable_conv2d"
            and layers[i + 1].cout
                == 2 * layers[i + 2].k_h * layers[i + 2].k_w
        ):
            conv = layers[i + 1]
            fused.append(LayerDesc(
                op="offset_gen",
                idx=L.idx,
                h_in=conv.h_in,
                w_in=conv.w_in,
                cin=conv.cin,
                cout=conv.cout,
                k_h=conv.k_h,
                k_w=conv.k_w,
                stride_h=conv.stride_h,
                stride_w=conv.stride_w,
                pad_top=conv.pad_top,
                pad_left=conv.pad_left,
                pad_bottom=conv.pad_bottom,
                pad_right=conv.pad_right,
                extra={"pool_stride": L.k_h},
            ))
            i += 2  # consume pool2d (i) and conv2d (i+1); deformable_conv2d processed next
        else:
            # Detect "looks like an offset generator but no deformable follows":
            # pool2d + conv2d(cout == 18 or any even cout==2*k*k) with NO
            # deformable_conv2d directly after. Warn so the user can investigate.
            if (
                L.op == "pool2d"
                and i + 1 < n
                and layers[i + 1].op == "conv2d"
                and layers[i + 1].cout % 2 == 0
                and not (
                    i + 2 < n and layers[i + 2].op == "deformable_conv2d"
                )
            ):
                print(
                    f"[WARNING] fuse_offset_generators: found pool2d+conv2d"
                    f"(cout={layers[i + 1].cout}) at layer idx={L.idx} but no "
                    f"deformable_conv2d follows — skipping fusion"
                )
            fused.append(L)
            i += 1

    for new_idx, layer in enumerate(fused):
        layer.idx = new_idx

    return fused


def fuse_activations(layers: List[LayerDesc]) -> List[LayerDesc]:
    """Merge (conv2d/offset_gen/deformable_conv2d → relu/prelu) pairs.

    Sets LayerDesc.activation on the preceding conv layer and drops the
    standalone activation LayerDesc.  Indices are renumbered after fusion.
    """
    fused: List[LayerDesc] = []
    i = 0
    while i < len(layers):
        L = layers[i]
        if (
            i + 1 < len(layers)
            and L.op in ("conv2d", "offset_gen", "deformable_conv2d")
            and layers[i + 1].op in ("relu", "prelu")
        ):
            L.activation = layers[i + 1].op
            fused.append(L)
            i += 2
        else:
            fused.append(L)
            i += 1

    for new_idx, layer in enumerate(fused):
        layer.idx = new_idx

    return fused
