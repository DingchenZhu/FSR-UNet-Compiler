"""SD-UNet (USR_Net_109) loader for pipeline.py.

Model: USR_Net_109_nopad.onnx  (zero-pad nodes removed; see Phase 11 record)
Input: 'data', shape (1, 1, 144, 256) — 256×144 video frame, single channel
Output buffer: 'unet_output_reg'  (sd_inst() convention)

Programmatic use:
    from frontend.unet_loader import MODEL_PATH, INPUT_SHAPES, make_config
    result = run_pipeline(MODEL_PATH, 'onnx', INPUT_SHAPES, config=make_config())

CLI use (equivalent):
    python3 pipeline.py \\
        --model /home/scratch.hansz_coreai/design/USR_Net_109_nopad.onnx \\
        --type onnx --input-shape 1 1 144 256 --input-name data \\
        --output-dir output/unet/ \\
        --golden golden/pseudo_code_load_next_mid.txt

Direct run:
    python3 frontend/unet_loader.py [--output-dir output/unet/] [--load-next] [--golden PATH]
"""
from __future__ import annotations

import os
import sys

MODEL_PATH = "/home/scratch.hansz_coreai/design/USR_Net_109_nopad.onnx"
INPUT_SHAPES = {"data": (1, 1, 144, 256)}


def make_config(
    output_dir: str = "output/unet/",
    is_first: bool = False,
    load_next: bool = False,
    emit_image_load: bool = True,
    verbose: bool = False,
):
    """Return a PipelineConfig pre-tuned for SD-UNet.

    Key differences from the FSRCNN defaults:
      - last_layer_dest_buffer = 'unet_output_reg'  (sd_inst() output register)
      - emit_image_load = True   (sd_inst starts with OffchipDataLoader)
      - tile_h = None            (full-height streaming; SD-UNet is not tiled)
    """
    # Import here to avoid circular imports when this module is loaded standalone
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline import PipelineConfig

    return PipelineConfig(
        output_dir=output_dir,
        is_first=is_first,
        load_next=load_next,
        emit_image_load=emit_image_load,
        # Phase 31: archived golden `pseudo_code_load_next_mid.txt` emits the
        # image OffchipDataLoader at the END of L=0 (not the start). Setting
        # `emit_image_load_at_end=True` matches that bucket layout, which the
        # `tools/layer_diff.py` per-layer multiset diff uses for correctness.
        emit_image_load_at_end=True,
        last_layer_dest_buffer="unet_output_reg",
        offchip_store_src_buffer="unet_output_reg",
        offchip_store_transnum=18,
        tile_h=None,   # full-height streaming: effective_tile_h = h_in per layer
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile SD-UNet via TVM frontend pipeline")
    parser.add_argument("--output-dir", default="output/unet/")
    parser.add_argument("--is-first", action="store_true", default=False,
                        help="Emit 5-instruction DDR preamble (matches golden first.txt)")
    parser.add_argument("--load-next", action="store_true", default=False)
    parser.add_argument("--no-emit-image-load", dest="emit_image_load", action="store_false", default=True)
    parser.add_argument("--golden", default=None, help="Optional golden file for diff")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from pipeline import run_pipeline, diff_with_golden

    cfg = make_config(
        output_dir=args.output_dir,
        is_first=args.is_first,
        load_next=args.load_next,
        emit_image_load=args.emit_image_load,
        verbose=args.verbose,
    )

    result = run_pipeline(MODEL_PATH, "onnx", INPUT_SHAPES, config=cfg)
    print(f"Done: {len(result.layers)} layers, {len(result.instructions)} instructions")
    print(f"Output: {args.output_dir}/")

    if args.golden:
        instr_path = os.path.join(args.output_dir, "pseudo_instructions.txt")
        n_bad = diff_with_golden(instr_path, args.golden)
        if n_bad == 0:
            print("GOLDEN MATCH")
        else:
            print(f"GOLDEN MISMATCH: {n_bad} differences")
        sys.exit(0 if n_bad == 0 else 1)
