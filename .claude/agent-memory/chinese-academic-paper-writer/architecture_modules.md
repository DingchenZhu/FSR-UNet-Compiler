---
name: Architecture Modules
description: Key module locations, data structures, and ISA instruction types
type: project
---

**Module locations**:
- pipeline.py: Top-level orchestration, PipelineConfig, PipelineResult
- frontend/frontend.py: load_onnx(), load_pytorch(), dump_relay()
- ir/layer_desc.py: LayerDesc dataclass, extract_layer_descs(), _collect_calls_exec_order()
- ir/fusion_pass.py: fuse_offset_generators()
- tiling/tiling.py: TilingPlan dataclass, choose_tiling(), plan_all()
- backend/emitter.py: InstructionEmitter, EmitterState, emit_program()
- backend/post_pass.py: finalize_instructions(), add_instruction_dependencies(), assign_dependency_registers()
- backend/isa.py: 7 ISA instruction classes

**LayerDesc fields**: op, idx, h_in, w_in, cin, cout, k_h, k_w, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right, groups, deformable, deformable_groups, dilation_h, dilation_w, needs_pixel_shuffle, upscale_factor, pool_type, pool_size, extra

**TilingPlan fields**: layer_idx, h_out_per_step, load_total_num, padding_num, line_buffer_rows, line_buffer_reshape, w_macro_tiles, w_micro_tile, cin_group, cout_group, weight_parall_mode, weight_transnum_base, read_mode, use_bilinear_weights, ky_outer, ic_inner, acc_mode, store_mode, quant_mode, quant_transnum, data_bas_addr, notes

**EmitterState fields**: dataloader_bas_addr, weight_bas_addr[3], quant_bas_addr, storer_bas_addr, line_buffer_idx, acc_reg_idx, quant_config_idx, offset_reg_idx, conv_layer_counter

**ISA instruction types** (7 total):
1. OffchipDataLoader: transnum, load_model, src_buffer_idx, bas_addr
2. DataLoader: layer_idx, line_buffer_reshape, is_padding_row, read_mode, transnum, line_buffer_idx, src_buffer_idx, bas_addr
3. WeightLoader: acc_reg_comp_idx, kernal_size, line_buffer_row_shift, line_buffer_idx, is_padding_col, weight_parall_mode, is_new, transnum, is_bilinear_bicubic, offset_reg_idx, bas_addr
4. OffsetLoader: offset_reg_idx, bas_addr
5. QuantLoader: quant_reg_load_idx, quant_mode, layer_idx, transnum, bas_addr
6. DataStorer: many fields including dest_buffer_idx, acc_mode, store_mode, pooling_out_mode, is_pooling, stride, base_addr_pooling
7. OffchipDataStorer: src_buffer, transnum, base_addr

**PipelineConfig key fields**:
- is_first: emit 5-instruction DDR preamble (quant×2 + weight×3)
- load_next: after layer-0 tiles, emit next-frame prefetch
- image_transnum: 576 (144×4 for UNet 144-row input)
- inter_layer_transnum: 64 (32×2) for UNet→FSRCNN boundary
- inter_layer_bas_addr: 576

**How to apply**: Use these as ground truth when describing the system architecture in the paper.
