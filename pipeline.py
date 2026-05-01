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
from ir.addr_alloc import allocate_addresses
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
    emit_image_load_at_end: bool = False  # True for SD-UNet archived golden bucket alignment (emits at end of L=0)
    # image_transnum is the word count for ONE input image of the FIRST model in
    # the multi-model pipeline (e.g. UNet for the SD+SR cascade). Default 576 =
    # 144×4 for UNet 144-row tile (h_in=144, w_in=256 → 144 rows × 256/64 words/row).
    # When None, the pipeline auto-derives it from layers[0] via
    # _derive_image_transnum() — only safe when emit_image_load=True (i.e. layers[0]
    # really is the first model's first layer). For FSRCNN-only with emit_image_load=False,
    # set this explicitly (defaults retain the legacy 576 to preserve UNet→FSRCNN behavior).
    image_transnum: Optional[int] = None
    inter_layer_transnum: Optional[int] = None  # 64 (32×2) for UNet→FSRCNN boundary
    # inter_layer_bas_addr / load_next_bas_addr default to image_transnum when None:
    # the next-image / inter-layer load lands AFTER the first model's image in the
    # on-chip feature buffer, so its base address equals the size of that image.
    inter_layer_bas_addr: Optional[int] = None
    # load_next OffchipDataLoader parameters (sr_inst golden: transnum=64, load_model=1,
    # bas_addr=image_transnum=576).
    load_next_transnum: int = 64
    load_next_load_model: int = 1
    load_next_bas_addr: Optional[int] = None
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
    # FSRCNN-only sr_inst() seed offsets — when None, defaults are derived
    # from emit_image_load (None and emit_image_load=False → sr_inst() values
    # 665 / [1737, 792, 1152] / image_transnum; otherwise 0 / [0,0,0] / 0).
    # These mirror sd_sr_codegen.py sr_inst() init at line 2490-2491:
    #   weightloadermanager.bas_addr_cur = [1737, 792, 1152]
    #   quantloadermanager.bas_addr_cur  = 665
    # The upstream UNet has consumed those buffer regions; FSRCNN starts above.
    initial_quant_bas_addr: Optional[int] = None
    initial_weight_bas_addr: Optional[List[int]] = None
    # Override layer-0's resolved input base address. Defaults to image_transnum
    # in FSRCNN-only mode (the FSRCNN input image is placed AFTER the UNet image,
    # at offset = image_transnum in the on-chip feature buffer).
    initial_layer0_input_bas_addr: Optional[int] = None
    # Feature-buffer address allocation solver. 'linear' (default) uses Linear
    # Scan — optimal for sequential models and UNet-style nested skips. 'ilp'
    # uses scipy.optimize.milp for exact optimality on arbitrary topologies
    # (requires scipy >= 1.7; auto-falls back to linear on timeout).
    alloc_solver: str = "linear"
    verbose: bool = False
    # Spatial tile height for the tiling pass.
    #   32   (default): FSRCNN tiled mode — 32-row hardware tiles.
    #   None: full-height mode for SD-UNet — each layer processes its entire
    #         h_in in one pass (cal_total_num = h_in // h_out_per_step).
    tile_h: Optional[int] = 32


# Hardware DataLoader transfers data in 64-pixel words (one MAC-array column burst
# loads 64 pixels along W). The on-chip image footprint is therefore
# h_in * ceil(w_in / 64) words. This matches the golden sd_sr_codegen.py convention
# `transnum = 144*4` for UNet h_in=144, w_in=256 → 144 * (256//64) = 576.
_WORDS_PER_ROW_DENOM = 64


def _derive_image_transnum(layer0: LayerDesc) -> int:
    """Compute the input-image word count for the first model from its first layer.

    Formula: h_in * max(1, w_in // _WORDS_PER_ROW_DENOM).
    The max(1, ...) guard handles small inputs (e.g. FSRCNN's 36×64 standalone case
    yields 36 instead of 0); however auto-derivation from layers[0] is only correct
    when layers[0] is the FIRST model's first layer (i.e. emit_image_load=True).
    """
    h_in = int(layer0.h_in)
    w_in = int(layer0.w_in)
    return h_in * max(1, w_in // _WORDS_PER_ROW_DENOM)


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

    # Resolve image_transnum / inter_layer_bas_addr / load_next_bas_addr defaults.
    # These three quantities all describe the same hardware fact: the on-chip
    # footprint of the FIRST model's input image. Keeping them coupled removes
    # the prior triple-hardcoded "576" magic number in PipelineConfig.
    #   - image_transnum:     auto-derived from layers[0] when emit_image_load=True
    #                         (then layers[0] IS the first model's first layer);
    #                         falls back to legacy 576 otherwise (FSRCNN-only,
    #                         where layers[0] is FSRCNN's first layer and the
    #                         image is implicit/pre-loaded by UNet upstream).
    #   - inter_layer_bas_addr / load_next_bas_addr: default to image_transnum
    #                         since the next-image / inter-model image is placed
    #                         right after the first model's image in the buffer.
    if cfg.image_transnum is None:
        if cfg.emit_image_load and layers:
            cfg.image_transnum = _derive_image_transnum(layers[0])
        else:
            # FSRCNN-only / no-image-load path: layers[0] is NOT the upstream model's
            # first layer, so we cannot derive from it. Fall back to the UNet legacy
            # value of 576 (= the upstream model's image footprint that this stage
            # must skip past in the on-chip buffer).
            cfg.image_transnum = 576
    if cfg.inter_layer_bas_addr is None:
        cfg.inter_layer_bas_addr = cfg.image_transnum
    if cfg.load_next_bas_addr is None:
        cfg.load_next_bas_addr = cfg.image_transnum

    # FSRCNN-only sr_inst() seed offsets. When emit_image_load=False (the
    # FSRCNN sr_inst() invocation), pre-seed the emitter's running-base
    # counters so QL/WL/DL bas_addrs start at the golden offsets that account
    # for already-consumed UNet quant/weight/feature regions:
    #   QL.bas_addr starts at 665   (sd_sr_codegen.py line 2491)
    #   WL.bas_addr (slot 0)         starts at 1152 — matches golden
    #     ``weightloadermanager.bas_addr_cur[2] = 1152`` since FSRCNN's first
    #     WL uses parall_mode=2. The compiler routes all standard-conv WL
    #     dispatches through slot 0, so we seed slot 0 with the value golden
    #     reads from its slot 2 for the very first emission.
    #   DL.bas_addr (layer 0) starts at image_transnum (576)
    # When emit_image_load=True (UNet standalone), all seeds remain 0.
    if cfg.initial_quant_bas_addr is None:
        cfg.initial_quant_bas_addr = 0 if cfg.emit_image_load else 665
    if cfg.initial_weight_bas_addr is None:
        cfg.initial_weight_bas_addr = (
            [0, 0, 0] if cfg.emit_image_load else [1152, 0, 0]
        )
    if cfg.initial_layer0_input_bas_addr is None:
        cfg.initial_layer0_input_bas_addr = (
            0 if cfg.emit_image_load else cfg.image_transnum
        )

    # ── Stage 3: Tiling ───────────────────────────────────────────────────
    if cfg.verbose:
        print("[3/4] Computing tiling plans")
    tilings = plan_all(layers, tile_h=cfg.tile_h)

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

    # ── Stage 3.5: Feature-buffer address allocation ──────────────────────
    # Assigns starting word-addresses to each layer's output tensor within
    # its ping-pong buffer (a or b), ensuring simultaneously live tensors
    # (skip connections) do not overlap.  For sequential models (FSRCNN),
    # all addresses resolve to 0 and the emitter is unchanged.
    #
    # Phase 25: returns AddrResult(addr_map, buf_map). addr_map drives
    # DS.base_addrs_res via emitter.layer_output_bas_addr; buf_map carries
    # the logical buffer class ('a'/'b'/'offset_reg') for diagnostics and
    # downstream coordination with emitter parity.
    addr_result = allocate_addresses(layers, solver=cfg.alloc_solver)
    addr_map = addr_result.addr_map
    buf_map = addr_result.buf_map
    skip_layers = {idx: addr for idx, addr in addr_map.items() if addr > 0}
    if skip_layers and cfg.verbose:
        print(f"  Address allocation: {len(skip_layers)} non-zero base addrs "
              f"(skip tensors): {skip_layers}")
        print(f"  Buffer class map: {buf_map}")

    # ── Stage 4: Emit + PostPass ──────────────────────────────────────────
    if cfg.verbose:
        print("[4/4] Emitting pseudo-instructions")
    instructions = emit_program(
        layers,
        tilings,
        addr_map=addr_map,
        buf_map=buf_map,
        is_first=cfg.is_first,
        load_next=cfg.load_next,
        emit_image_load=cfg.emit_image_load,
        emit_image_load_at_end=cfg.emit_image_load_at_end,
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
        initial_quant_bas_addr=cfg.initial_quant_bas_addr,
        initial_weight_bas_addr=cfg.initial_weight_bas_addr,
        initial_layer0_input_bas_addr=cfg.initial_layer0_input_bas_addr,
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
    """Compare output against golden file. Returns count of mismatched lines.

    Comparison is subset-mode: every field present in the golden must appear in
    our output with the same value, but our output may carry additional constant
    fields (e.g. dependency/dest/src1-4 from post_pass, is_offset, offchip_read_mode)
    that older goldens predate.  This lets FSRCNN and SD-UNet goldens stay valid
    even as the output format gains new bookkeeping fields.
    """
    import ast
    with open(output_path, encoding="utf-8") as f:
        output_lines = [ast.literal_eval(l.strip()) for l in f if l.strip()]
    with open(golden_path, encoding="utf-8") as f:
        golden_lines = [ast.literal_eval(l.strip()) for l in f if l.strip()]

    mismatches = 0
    for i, (o, g) in enumerate(zip(output_lines, golden_lines)):
        # Report only golden fields that differ; ignore extra fields in our output.
        bad = {k: (o.get(k), v) for k, v in g.items() if o.get(k) != v}
        if bad:
            mismatches += 1
            print(f"[MISMATCH] instruction {i}")
            for k, (got, want) in bad.items():
                print(f"  field '{k}': output={got!r}  golden={want!r}")
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
    # Pipeline-stage flags (mirror PipelineConfig fields). Default behavior is
    # the standalone UNet case (emit_image_load=True, load_next=False); pass
    # --no-emit-image-load + --no-load-next to reproduce the FSRCNN-only golden.
    parser.add_argument(
        "--no-emit-image-load",
        dest="emit_image_load",
        action="store_false",
        default=True,
        help="Skip the layer-0 OffchipDataLoader for the input image — used when "
             "the image is pre-loaded by an upstream pipeline stage (e.g. UNet → "
             "FSRCNN sr_inst() golden).",
    )
    parser.add_argument(
        "--no-load-next",
        dest="load_next",
        action="store_false",
        default=False,
        help="Disable load_next prefetch of the next frame's image after layer 0. "
             "Combined with --no-emit-image-load, reproduces the standalone "
             "FSRCNN sr_inst() golden invocation.",
    )
    parser.add_argument(
        "--load-next",
        dest="load_next",
        action="store_true",
        help="Enable load_next prefetch of the next frame's image after layer 0.",
    )
    args = parser.parse_args()

    shape = tuple(args.input_shape)
    cfg = PipelineConfig(
        output_dir=args.output_dir,
        verbose=args.verbose,
        emit_image_load=args.emit_image_load,
        load_next=args.load_next,
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
