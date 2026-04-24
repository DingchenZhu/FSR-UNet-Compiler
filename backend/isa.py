"""Hardware ISA wrappers — golden-format compatible.

All 7 instruction types match the field layout in:
  tvm-tiling/references/instruction.py
  tvm-tiling/vis_compiler/emit/isa.py

Golden format: one Python dict per line, code_num as a single-element list.
Post-pass fields (dependency, dest, src1-4) are added by post_pass.py.
"""
from typing import Any, Dict, List


class Inst:
    current_code_num: int = 0
    code_list: List[Dict[str, Any]] = []


def reset_instruction_stream() -> None:
    Inst.current_code_num = 0
    Inst.code_list = []


class OffchipDataLoader:
    @staticmethod
    def dispatch(transnum: Any, load_model: int, src_buffer_idx: Any, bas_addr: int) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "OffchipDataLoader",
            "transnum": transnum,
            "load_model": load_model,
            "src_buffer_idx": src_buffer_idx,
            "bas_addr": bas_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class DataLoader:
    @staticmethod
    def dispatch(
        layer_idx: int,
        line_buffer_reshape: int,
        is_padding_row: int,
        read_mode: int,
        transnum: int,
        line_buffer_idx: int,
        src_buffer_idx: Any,
        bas_addr: Any,
    ) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "DataLoader",
            "layer_idx": layer_idx,
            "line_buffer_reshape": line_buffer_reshape,
            "is_padding_row": is_padding_row,
            "read_mode": read_mode,
            "transnum": transnum,
            "line_buffer_idx": line_buffer_idx,
            "src_buffer_idx": src_buffer_idx,
            "bas_addr": bas_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class WeightLoader:
    @staticmethod
    def dispatch(
        acc_reg_comp_idx: int,
        kernal_size: int,
        line_buffer_row_shift: int,
        line_buffer_idx: int,
        is_padding_col: int,
        weight_parall_mode: int,
        is_new: int,
        transnum: int,
        bas_addr: int,
        is_bilinear_bicubic: int,
        offset_reg_idx: int,
    ) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "WeightLoader",
            "acc_reg_comp_idx": acc_reg_comp_idx,
            "kernal_size": kernal_size,
            "line_buffer_row_shift": line_buffer_row_shift,
            "line_buffer_idx": line_buffer_idx,
            "is_padding_col": is_padding_col,
            "weight_parall_mode": weight_parall_mode,
            "is_new": is_new,
            "transnum": transnum,
            "is_bilinear_bicubic": is_bilinear_bicubic,
            "offset_reg_idx": offset_reg_idx,
            "bas_addr": bas_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class OffsetLoader:
    @staticmethod
    def dispatch(offset_reg_idx: int, bas_addr: Any) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "OffsetLoader",
            "offset_reg_idx": offset_reg_idx,
            "bas_addr": bas_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class QuantLoader:
    @staticmethod
    def dispatch(
        quant_reg_load_idx: int,
        quant_mode: int,
        layer_idx: int,
        transnum: int,
        bas_addr: int,
    ) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "QuantLoader",
            "quant_reg_load_idx": quant_reg_load_idx,
            "quant_mode": quant_mode,
            "layer_idx": layer_idx,
            "transnum": transnum,
            "bas_addr": bas_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class DataStorer:
    @staticmethod
    def dispatch(
        quant_config_idx: int,
        pixelshuffle_out_mode: int,
        is_pixelshuffle: int,
        pooling_out_mode: int,
        pooling_out_new: int,
        is_pooling: int,
        reg_out_idx: int,
        acc_mode: int,
        transfer_num: int,
        store_mode: int,
        stride: int,
        base_addr_pooling: int,
        base_addrs_res: Any,
        is_bicubic_add: int,
        is_first_or_last_row: int,
        is_mask: int,
        is_new: int,
        dest_buffer_idx: Any,
    ) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "DataStorer",
            "pooling_out_new": pooling_out_new,
            "quant_config_idx": quant_config_idx,
            "pixelshuffle_out_mode": pixelshuffle_out_mode,
            "is_pixelshuffle": is_pixelshuffle,
            "pooling_out_mode": pooling_out_mode,
            "is_pooling": is_pooling,
            "reg_out_idx": reg_out_idx,
            "acc_mode": acc_mode,
            "transfer_num": transfer_num,
            "store_mode": store_mode,
            "stride": stride,
            "is_bicubic_add": is_bicubic_add,
            "is_first_or_last_row": is_first_or_last_row,
            "is_offset": 0,
            "is_mask": is_mask,
            "is_new": is_new,
            "dest_buffer_idx": dest_buffer_idx,
            "base_addr_pooling": base_addr_pooling,
            "base_addrs_res": base_addrs_res,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class OffchipDataStorer:
    @staticmethod
    def dispatch(src_buffer: Any, transnum: int, base_addr: int) -> Dict:
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "OffchipDataStorer",
            "src_buffer": src_buffer,
            "transnum": transnum,
            "base_addr": base_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code
