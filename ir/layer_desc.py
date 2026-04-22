"""Relay IR → neutral LayerDesc representations.

DeformableConv2d stays as one logical op (deformable=True flag).
Hardware lowers it via OffsetLoader + bilinear WeightLoader — NOT via TVM TE schedules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import tvm
from tvm import relay
from tvm.ir import Op


def _int_tuple(x) -> Tuple[int, ...]:
    out = []
    for v in x:
        if hasattr(v, "value"):
            out.append(int(v.value))
        else:
            out.append(int(v))
    return tuple(out)


def _tensor_shape(expr: relay.Expr) -> List[int]:
    tt = expr.checked_type
    if not isinstance(tt, relay.TensorType):
        raise TypeError(f"Expected TensorType, got {type(tt)}")
    shape = []
    for s in tt.shape:
        if isinstance(s, tvm.tir.IntImm):
            shape.append(int(s.value))
        else:
            shape.append(int(s))
    return shape


@dataclass
class LayerDesc:
    """Single compute step visible to tiling + codegen backend."""

    op: str
    idx: int
    h_in: int
    w_in: int
    cin: int
    cout: int
    k_h: int = 1
    k_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_left: int = 0
    pad_bottom: int = 0
    pad_right: int = 0
    groups: int = 1
    deformable: bool = False
    deformable_groups: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    needs_pixel_shuffle: bool = False
    upscale_factor: int = 1
    pool_type: Optional[str] = None
    pool_size: Tuple[int, int] = (1, 1)
    extra: Dict[str, Any] = field(default_factory=dict)


def _collect_calls_exec_order(
    expr: relay.Expr,
    out: List[relay.Call],
    visited: Optional[set] = None,
) -> None:
    """Walk call-graph DAG and collect Call nodes in approximate execution order.

    Uses a visited set keyed on TVM's stable hash (the underlying C++ handle)
    to avoid exponential re-traversal of shared subgraphs.
    IMPORTANT: Do NOT use id(expr) — TVM ObjectRef creates a new Python wrapper
    on each access, so id() is unstable even for the same underlying node.
    """
    if visited is None:
        visited = set()
    # TVM's __hash__ is based on the C++ object pointer — stable across wrappers.
    # TVM's __eq__ (same_as) is consistent with __hash__, so sets work correctly.
    if expr in visited:
        return
    visited.add(expr)

    if isinstance(expr, relay.Let):
        _collect_calls_exec_order(expr.value, out, visited)
        _collect_calls_exec_order(expr.body, out, visited)
    elif isinstance(expr, relay.Call):
        for a in expr.args:
            _collect_calls_exec_order(a, out, visited)
        out.append(expr)
    elif isinstance(expr, relay.Tuple):
        for f in expr.fields:
            _collect_calls_exec_order(f, out, visited)
    elif isinstance(expr, relay.TupleGetItem):
        _collect_calls_exec_order(expr.tuple_value, out, visited)
    elif isinstance(expr, relay.Function):
        _collect_calls_exec_order(expr.body, out, visited)


def _call_op_name(call: relay.Call) -> Optional[str]:
    fn = call.op
    if isinstance(fn, Op):
        return fn.name
    return None


def _conv_like_from_call(call: relay.Call, idx: int, deformable: bool) -> LayerDesc:
    attrs = call.attrs
    k = _int_tuple(attrs.kernel_size)
    strides = _int_tuple(attrs.strides)
    dil = _int_tuple(getattr(attrs, "dilation", (1, 1)))
    padding = _int_tuple(attrs.padding)
    if len(padding) == 2:
        ph, pw = int(padding[0]), int(padding[1])
        pad_top = pad_bottom = ph
        pad_left = pad_right = pw
    else:
        pad_top, pad_left, pad_bottom, pad_right = (int(x) for x in padding)
    groups = int(attrs.groups)
    data = call.args[0]
    weight = call.args[2] if deformable else call.args[1]
    dshape = _tensor_shape(data)
    wshape = _tensor_shape(weight)
    _, cin, h_in, w_in = dshape[0], dshape[1], dshape[2], dshape[3]
    cout, _, kh, kw = wshape[0], wshape[1], wshape[2], wshape[3]
    dg = int(getattr(attrs, "deformable_groups", 1)) if deformable else 1
    return LayerDesc(
        op="deformable_conv2d" if deformable else "conv2d",
        idx=idx,
        h_in=h_in,
        w_in=w_in,
        cin=cin,
        cout=cout,
        k_h=kh,
        k_w=kw,
        stride_h=strides[0],
        stride_w=strides[1],
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        groups=groups,
        deformable=deformable,
        deformable_groups=dg,
        dilation_h=dil[0],
        dilation_w=dil[1],
    )


def _pool_from_call(call: relay.Call, idx: int, pool_type: str) -> LayerDesc:
    attrs = call.attrs
    pool_size = _int_tuple(attrs.pool_size)
    strides = _int_tuple(attrs.strides)
    padding = _int_tuple(attrs.padding)
    if len(padding) == 2:
        ph, pw = int(padding[0]), int(padding[1])
        pad_top = pad_bottom = ph
        pad_left = pad_right = pw
    else:
        pad_top, pad_left, pad_bottom, pad_right = (int(x) for x in padding)
    data = call.args[0]
    dshape = _tensor_shape(data)
    _, cin, h_in, w_in = dshape[0], dshape[1], dshape[2], dshape[3]
    return LayerDesc(
        op="pool2d",
        idx=idx,
        h_in=h_in,
        w_in=w_in,
        cin=cin,
        cout=cin,
        k_h=pool_size[0],
        k_w=pool_size[1],
        stride_h=strides[0],
        stride_w=strides[1],
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        pool_type=pool_type,
        pool_size=(pool_size[0], pool_size[1]),
    )


def extract_layer_descs(mod: tvm.ir.IRModule) -> List[LayerDesc]:
    """Walk main() and return ordered LayerDesc list for supported ops."""
    mod = relay.transform.InferType()(mod)
    fn = mod["main"]
    calls: List[relay.Call] = []
    _collect_calls_exec_order(fn.body, calls)

    descs: List[LayerDesc] = []
    idx = 0
    for call in calls:
        name = _call_op_name(call)
        if name == "nn.conv2d":
            descs.append(_conv_like_from_call(call, idx, deformable=False))
            idx += 1
        elif name == "nn.deformable_conv2d":
            descs.append(_conv_like_from_call(call, idx, deformable=True))
            idx += 1
        elif name == "nn.max_pool2d":
            descs.append(_pool_from_call(call, idx, "max"))
            idx += 1
        elif name == "nn.avg_pool2d":
            descs.append(_pool_from_call(call, idx, "avg"))
            idx += 1
        elif name in ("nn.relu", "nn.prelu"):
            data = call.args[0]
            dshape = _tensor_shape(data)
            _, c, h, w = dshape[0], dshape[1], dshape[2], dshape[3]
            descs.append(LayerDesc(op=name.split(".")[-1], idx=idx, h_in=h, w_in=w, cin=c, cout=c))
            idx += 1
        # nn.batch_norm, nn.layer_norm, add (residual), etc. can be added here
    return descs
