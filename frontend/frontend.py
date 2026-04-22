"""ONNX / PyTorch → Relay IRModule.

Phase 1 policy: only InferType is run automatically (no fusion, no constant folding).
Callers can opt-in to FoldConstant via the pipeline config.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import tvm
from tvm import relay

try:
    import onnx
except ImportError:
    onnx = None

try:
    import torch
except ImportError:
    torch = None


def load_onnx(
    model_path: str,
    shape_dict: Dict[str, Tuple[int, ...]],
    freeze_params: bool = True,
    dtype: Optional[Union[str, Dict[str, str]]] = None,
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """Load ONNX model and convert to Relay IR."""
    if onnx is None:
        raise RuntimeError("onnx package required for load_onnx")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    model_proto = onnx.load(model_path)
    init_names = {t.name for t in model_proto.graph.initializer}
    inames = [i.name for i in model_proto.graph.input if i.name not in init_names]
    if dtype is None:
        dtype = {n: "float32" for n in inames}
    elif isinstance(dtype, str):
        dtype = {n: dtype for n in inames}
    mod, params = relay.frontend.from_onnx(
        model_proto,
        shape=shape_dict,
        dtype=dtype,
        freeze_params=freeze_params,
    )
    return mod, params


def load_pytorch(
    model: Any,
    example_inputs: Any,
    input_names: Optional[List[str]] = None,
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """Trace a PyTorch model and convert to Relay IR via torch.jit.trace."""
    if torch is None:
        raise RuntimeError("PyTorch is required for load_pytorch")
    model.eval()
    traced = torch.jit.trace(model, example_inputs)
    input_names = input_names or ["input"]
    if isinstance(example_inputs, torch.Tensor):
        input_infos = [(input_names[0], list(example_inputs.shape))]
    else:
        names = input_names
        if len(names) != len(example_inputs):
            names = [f"input{i}" for i in range(len(example_inputs))]
        input_infos = [(n, list(t.shape)) for n, t in zip(names, example_inputs)]
    mod, params = relay.frontend.from_pytorch(
        traced,
        input_infos,
        default_dtype="float32",
        use_parser_friendly_name=True,
    )
    return mod, params


def onnx_input_names(model_path: str) -> List[str]:
    """Return non-initializer input names from an ONNX model."""
    if onnx is None:
        raise RuntimeError("onnx package required")
    m = onnx.load(model_path)
    init_names = {t.name for t in m.graph.initializer}
    return [i.name for i in m.graph.input if i.name not in init_names]


def dump_relay(mod: tvm.ir.IRModule, path: str) -> None:
    """Write human-readable Relay IR text to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(mod.astext(show_meta_data=False))
