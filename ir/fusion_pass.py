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

_OFFSET_GEN_COUT = 18   # 2 × 9 offsets for a 3×3 deformable kernel


def fuse_offset_generators(layers: List[LayerDesc]) -> List[LayerDesc]:
    """Replace each (pool2d + conv2d(cout=18)) pair preceding a deformable_conv2d
    with a single op='offset_gen' LayerDesc.  Layer indices are reassigned after fusion.

    Recognition rule:
        layers[i].op   == 'pool2d'
        layers[i+1].op == 'conv2d' and layers[i+1].cout == 18
        layers[i+2].op == 'deformable_conv2d'

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
            and layers[i + 1].cout == _OFFSET_GEN_COUT
            and layers[i + 2].op == "deformable_conv2d"
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
            fused.append(L)
            i += 1

    for new_idx, layer in enumerate(fused):
        layer.idx = new_idx

    return fused
