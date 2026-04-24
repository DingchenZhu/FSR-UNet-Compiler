"""Top-level compiler pipeline orchestration.

Usage:
    python3 pipeline.py --model path/to/model.onnx --type onnx \
        --input-shape 1 1 144 256 --output-dir output/

    python3 pipeline.py --model path/to/model.py --type pytorch \
        --input-shape 1 1 144 256 --output-dir output/

Each stage dumps to output/:
    relay_ir.txt          Relay IR text
    layer_descs.json      extracted LayerDesc list
    tiling_plan.json      TilingPlan list
    pseudo_instructions.txt  final pseudo-instructions (golden format, one dict/line)
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import tvm
from tvm import relay

# Add tvm-design root to path so relative imports work when run as script
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from backend.emitter import emit_program
from frontend.frontend import dump_relay, load_onnx, load_pytorch
from ir.fusion_pass import fuse_activations, fuse_offset_generators
from ir.layer_desc import LayerDesc, extract_layer_descs
from tiling.tiling import TilingPlan, plan_all


@dataclass
class PipelineConfig:
    output_dir: str = "output"
    dump_relay: bool = True
    dump_layers: bool = True
    dump_tiling: bool = True
    dump_instructions: bool = True
    finalize_instructions: bool = True
    fold_constant: bool = False   # Phase 1: off by default
    # load_next scheduling flags (mirrors sd_inst parameters)
    is_first: bool = False
    load_next: bool = False
    emit_image_load: bool = True       # False for FSRCNN-only golden (image pre-loaded by UNet)
    image_transnum: int = 576          # 144×4 for UNet 144-row input tile
    inter_layer_transnum: Optional[int] = None  # 64 (32×2) for UNet→FSRCNN boundary
    inter_layer_bas_addr: int = 576
    # load_next OffchipDataLoader parameters (sr_inst golden: transnum=64, load_model=1, bas_addr=576)
    load_next_transnum: int = 64
    load_next_load_model: int = 1
    load_next_bas_addr: int = 576
    # Terminal OffchipDataStorer (FSRCNN sr_inst() tail write-back).
    # Defaults match sr_inst(): src_buffer='fsrcnn_output_buffer', transnum=1024, base_addr=0.
    emit_offchip_store: bool = True
    offchip_store_src_buffer: str = "fsrcnn_output_buffer"
    offchip_store_transnum: int = 1024
    offchip_store_base_addr: int = 0
    # dest_buffer_idx for the terminal conv/deformable_conv layer's DataStorer.
    # All preceding layers ping-pong between 'a' and 'b'; the terminal layer
    # targets this named buffer which the final OffchipDataStorer drains to DDR.
    last_layer_dest_buffer: str = "fsrcnn_output_buffer"
    verbose: bool = False


@dataclass
class PipelineResult:
    mod: tvm.ir.IRModule
    params: Dict[str, Any]
    layers: List[LayerDesc] = field(default_factory=list)
    tilings: List[TilingPlan] = field(default_factory=list)
    instructions: List[Dict[str, Any]] = field(default_factory=list)


def run_pipeline(
    model_path: str,
    model_type: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Full compilation pipeline: model file → pseudo-instructions.

    model_type: 'onnx' or 'pytorch'
    input_shapes: dict mapping input name(s) to shape tuples
    """
    cfg = config or PipelineConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Stage 1: Frontend ──────────────────────────────────────────────────
    if cfg.verbose:
        print(f"[1/4] Loading {model_type} model: {model_path}")

    if model_type == "onnx":
        mod, params = load_onnx(model_path, input_shapes)
    elif model_type == "pytorch":
        import torch
        # Dynamically load model class from file
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_model", model_path)
        user_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_mod)
        model = user_mod.get_model()   # convention: model file exposes get_model()
        example = {n: torch.zeros(s) for n, s in input_shapes.items()}
        if len(example) == 1:
            example_tensor = list(example.values())[0]
        else:
            example_tensor = tuple(example.values())
        mod, params = load_pytorch(model, example_tensor, input_names=list(input_shapes.keys()))
    else:
        raise ValueError(f"model_type must be 'onnx' or 'pytorch', got {model_type!r}")

    # Phase 1: only InferType (no fusion, no constant folding unless configured)
    with tvm.transform.PassContext(opt_level=0):
        mod = relay.transform.InferType()(mod)
        if cfg.fold_constant:
            mod = relay.transform.FoldConstant()(mod)

    if cfg.dump_relay:
        dump_relay(mod, os.path.join(cfg.output_dir, "relay_ir.txt"))
        if cfg.verbose:
            print(f"  Relay IR → {cfg.output_dir}/relay_ir.txt")

    # ── Stage 2: LayerDesc extraction + fusion ────────────────────────────
    if cfg.verbose:
        print("[2/4] Extracting layer descriptors")
    layers = extract_layer_descs(mod)
    layers = fuse_offset_generators(layers)
    layers = fuse_activations(layers)

    # ── Stage 3: Tiling ───────────────────────────────────────────────────
    if cfg.verbose:
        print("[3/4] Computing tiling plans")
    tilings = plan_all(layers)

    if cfg.dump_layers:
        payload = [
            {**dataclasses.asdict(L), "tiling": dataclasses.asdict(T)}
            for L, T in zip(layers, tilings)
        ]
        layers_path = os.path.join(cfg.output_dir, "layer_descs.json")
        with open(layers_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        if cfg.verbose:
            print(f"  LayerDescs + TilingPlans → {layers_path}")

    if cfg.dump_tiling:
        tiling_path = os.path.join(cfg.output_dir, "tiling_plan.json")
        with open(tiling_path, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(T) for T in tilings], f, indent=2, default=str)

    # ── Stage 4: Emit + PostPass ──────────────────────────────────────────
    if cfg.verbose:
        print("[4/4] Emitting pseudo-instructions")
    instructions = emit_program(
        layers,
        tilings,
        is_first=cfg.is_first,
        load_next=cfg.load_next,
        emit_image_load=cfg.emit_image_load,
        image_transnum=cfg.image_transnum,
        inter_layer_transnum=cfg.inter_layer_transnum,
        inter_layer_bas_addr=cfg.inter_layer_bas_addr,
        load_next_transnum=cfg.load_next_transnum,
        load_next_load_model=cfg.load_next_load_model,
        load_next_bas_addr=cfg.load_next_bas_addr,
        emit_offchip_store=cfg.emit_offchip_store,
        offchip_store_src_buffer=cfg.offchip_store_src_buffer,
        offchip_store_transnum=cfg.offchip_store_transnum,
        offchip_store_base_addr=cfg.offchip_store_base_addr,
        last_layer_dest_buffer=cfg.last_layer_dest_buffer,
        finalize=cfg.finalize_instructions,
    )

    if cfg.dump_instructions:
        instr_path = os.path.join(cfg.output_dir, "pseudo_instructions.txt")
        with open(instr_path, "w", encoding="utf-8") as f:
            for inst in instructions:
                f.write(str(inst) + "\n")
        if cfg.verbose:
            print(f"  {len(instructions)} instructions → {instr_path}")

    return PipelineResult(
        mod=mod,
        params=params,
        layers=layers,
        tilings=tilings,
        instructions=instructions,
    )


def diff_with_golden(output_path: str, golden_path: str) -> int:
    """Compare output against golden file. Returns count of mismatched lines."""
    import ast
    with open(output_path, encoding="utf-8") as f:
        output_lines = [ast.literal_eval(l.strip()) for l in f if l.strip()]
    with open(golden_path, encoding="utf-8") as f:
        golden_lines = [ast.literal_eval(l.strip()) for l in f if l.strip()]

    mismatches = 0
    for i, (o, g) in enumerate(zip(output_lines, golden_lines)):
        if o != g:
            mismatches += 1
            print(f"[MISMATCH] instruction {i}")
            print(f"  output: {o}")
            print(f"  golden: {g}")
            if mismatches >= 10:
                print("... (stopping after 10 mismatches)")
                break
    if len(output_lines) != len(golden_lines):
        print(f"Length mismatch: output={len(output_lines)}, golden={len(golden_lines)}")
        mismatches += abs(len(output_lines) - len(golden_lines))
    return mismatches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TVM compiler frontend pipeline")
    parser.add_argument("--model", required=True, help="Model file path (.onnx or .py)")
    parser.add_argument("--type", required=True, choices=["onnx", "pytorch"], dest="model_type")
    parser.add_argument("--input-shape", nargs="+", type=int, default=[1, 1, 144, 256])
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--golden", default=None, help="Golden file for diff comparison")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    shape = tuple(args.input_shape)
    cfg = PipelineConfig(
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    result = run_pipeline(
        model_path=args.model,
        model_type=args.model_type,
        input_shapes={args.input_name: shape},
        config=cfg,
    )
    print(f"Done: {len(result.layers)} layers, {len(result.instructions)} instructions")
    print(f"Output: {args.output_dir}/")

    if args.golden:
        instr_path = os.path.join(args.output_dir, "pseudo_instructions.txt")
        n_bad = diff_with_golden(instr_path, args.golden)
        if n_bad == 0:
            print("GOLDEN MATCH: output matches golden exactly")
        else:
            print(f"GOLDEN MISMATCH: {n_bad} differences found")
        sys.exit(0 if n_bad == 0 else 1)
