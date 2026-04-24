"""Post-process raw ISA dicts: field alignment, dependency edges, virtual registers.

Rules are ported verbatim from sd_sr_codegen.py __main__ block.
Do NOT generalize the dependency rules — any deviation breaks golden parity.

Critical: src4 quirk (line 256 in original) assigns src_code[2] not src_code[3].
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Dict, List

# DataLoader dependency threshold: distinguishes layer-0 DataLoaders reading
# from the offchip DDR preload (load_model==0, bas_addr < threshold) from
# subsequent rows (load_model==1, bas_addr >= threshold).
#
# 576 = 144 (UNet input image height in rows) * 4 (offchip transnum per row).
# This is UNet-specific. For other input resolutions the threshold must be
# updated — FSRCNN's 36×64 input happens to still satisfy the layer-0 branch
# because its layer-0 bas_addr values stay well below 576.
#
# TODO: make this configurable (e.g. derive from the first layer's h_in and
# offchip transnum) when supporting different input resolutions generically.
_LAYER0_DL_BAS_ADDR_THRESHOLD = 144 * 4  # UNet-specific; update for other resolutions


def align_instruction_fields(code_list: List[Dict[str, Any]]) -> None:
    """Add golden/encoder fields that are missing from minimal isa.dispatch records."""
    for c in code_list:
        op = c.get("op_code")
        if op == "DataStorer":
            c.setdefault("is_offset", 0)


def add_instruction_dependencies(code_list: List[Dict[str, Any]]) -> int:
    """
    Fill 'dependency' with producer instruction indices.
    Returns max_gap (largest i - dep_index) used by register allocator.

    Rules ported verbatim from sd_sr_codegen.py.
    """
    n = len(code_list)
    for i in range(n):
        code_list[i]["dependency"] = []
        code_list[i]["dest"] = 0
        code_list[i]["src1"] = 0
        code_list[i]["src2"] = 0
        code_list[i]["src3"] = 0
        code_list[i]["src4"] = 0

        op = code_list[i]["op_code"]

        if op == "OffchipDataLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataLoader":
                    if code_list[d].get("layer_idx") == 0:
                        code_list[i]["dependency"].append(d)
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "OffchipDataLoader":
                    code_list[i]["dependency"].append(d)
                    break

        elif op == "WeightLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataLoader":
                    if code_list[d].get("line_buffer_idx") == code_list[i].get("line_buffer_idx"):
                        code_list[i]["dependency"].append(d)
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d].get("reg_out_idx") == code_list[i].get("acc_reg_comp_idx"):
                        code_list[i]["dependency"].append(d)
                        break
                if code_list[d]["op_code"] == "WeightLoader":
                    if code_list[d].get("acc_reg_comp_idx") == code_list[i].get("acc_reg_comp_idx"):
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    code_list[i]["dependency"].append(d)
                    break
            # Bilinear WeightLoader needs its OffsetLoader
            if code_list[i].get("is_bilinear_bicubic") == 1 and code_list[i]["dependency"]:
                last = code_list[i]["dependency"][-1]
                last_w = code_list[last]
                if last_w.get("is_bilinear_bicubic") == 0 or (
                    last_w.get("is_bilinear_bicubic") == 1
                    and last_w.get("offset_reg_idx") != code_list[i].get("offset_reg_idx")
                ):
                    for d in range(i - 1, -1, -1):
                        if code_list[d]["op_code"] == "OffsetLoader":
                            if code_list[d].get("offset_reg_idx") == code_list[i].get("offset_reg_idx"):
                                code_list[i]["dependency"].append(d)
                                break

        elif op == "DataLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    if code_list[d].get("line_buffer_idx") == code_list[i].get("line_buffer_idx"):
                        code_list[i]["dependency"].append(d)
                        break
            if code_list[i].get("layer_idx") == 0:
                dataloader_count = 0
                for d in range(i - 1, -1, -1):
                    if code_list[d]["op_code"] == "DataLoader" and dataloader_count < 2:
                        if code_list[d].get("layer_idx") == 0:
                            dataloader_count += 1
                    if dataloader_count == 2 or code_list[d]["op_code"] == "OffchipDataStorer":
                        break
                if dataloader_count < 2:
                    if code_list[i].get("bas_addr", 0) < _LAYER0_DL_BAS_ADDR_THRESHOLD:
                        for d in range(i - 1, -1, -1):
                            if code_list[d]["op_code"] == "OffchipDataLoader":
                                if code_list[d].get("src_buffer_idx") == 0 and code_list[d].get("load_model") == 0:
                                    code_list[i]["dependency"].append(d)
                                    break
                    if code_list[i].get("bas_addr", 0) >= _LAYER0_DL_BAS_ADDR_THRESHOLD:
                        for d in range(i - 1, -1, -1):
                            if code_list[d]["op_code"] == "OffchipDataLoader":
                                if code_list[d].get("src_buffer_idx") == 0 and code_list[d].get("load_model") == 1:
                                    code_list[i]["dependency"].append(d)
                                    break
            else:
                dataloader_count = 0
                dataloader_idx: List[int] = []
                datastorer_count = 0
                datastorer_idx: List[int] = []
                for d in range(i - 1, -1, -1):
                    if code_list[d]["op_code"] == "DataLoader" and dataloader_count < 2:
                        dataloader_count += 1
                        dataloader_idx.append(d)
                    if code_list[d]["op_code"] == "DataStorer" and datastorer_count < 1:
                        datastorer_count += 1
                        datastorer_idx.append(d)
                    if dataloader_count == 2 and datastorer_count == 1:
                        break
                if (
                    len(dataloader_idx) >= 2
                    and len(datastorer_idx) >= 1
                    and (
                        code_list[dataloader_idx[0]].get("layer_idx") != code_list[i].get("layer_idx")
                        or code_list[dataloader_idx[1]].get("layer_idx") != code_list[i].get("layer_idx")
                    )
                ):
                    code_list[i]["dependency"].append(datastorer_idx[0])

        elif op == "QuantLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d].get("quant_config_idx") == code_list[i].get("quant_reg_load_idx"):
                        code_list[i]["dependency"].append(d)
                        break
            quantloader_count = 0
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "OffchipDataLoader":
                    if code_list[d].get("src_buffer_idx") == 2:
                        code_list[i]["dependency"].append(d)
                        break
                if code_list[d]["op_code"] == "QuantLoader":
                    quantloader_count += 1
                if quantloader_count == 2:
                    break

        elif op == "DataStorer":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "QuantLoader":
                    if code_list[d].get("quant_reg_load_idx") == code_list[i].get("quant_config_idx"):
                        code_list[i]["dependency"].append(d)
                        break
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d].get("quant_config_idx") == code_list[i].get("quant_config_idx"):
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    if code_list[d].get("acc_reg_comp_idx") == code_list[i].get("reg_out_idx"):
                        code_list[i]["dependency"].append(d)
                        break
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    code_list[i]["dependency"].append(d)
                    break
            if code_list[i].get("dest_buffer_idx") in ("fsrcnn_output_buffer", "unet_output_reg"):
                for d in range(i - 1, -1, -1):
                    if code_list[d]["op_code"] == "OffchipDataStorer":
                        code_list[i]["dependency"].append(d)
                        break

        elif op == "OffsetLoader":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "WeightLoader":
                    if (
                        code_list[d].get("offset_reg_idx") == code_list[i].get("offset_reg_idx")
                        and code_list[d].get("is_bilinear_bicubic") == 1
                    ):
                        code_list[i]["dependency"].append(d)
                        break
                    if code_list[d].get("is_bilinear_bicubic") == 0:
                        break
            offsetloader_count = 0
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "OffsetLoader":
                    offsetloader_count += 1
                    if offsetloader_count == 2:
                        break
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d].get("dest_buffer_idx") == "offset_reg":
                        code_list[i]["dependency"].append(d)
                        break

        elif op == "OffchipDataStorer":
            for d in range(i - 1, -1, -1):
                if code_list[d]["op_code"] == "DataStorer":
                    if code_list[d].get("dest_buffer_idx") == code_list[i].get("src_buffer"):
                        code_list[i]["dependency"].append(d)
                        break

    max_gap = 0
    for i in range(n):
        for t in code_list[i]["dependency"]:
            if i - t > max_gap:
                max_gap = i - t
    return max_gap


def assign_dependency_registers(code_list: List[Dict[str, Any]], max_gap: int) -> int:
    """
    Assign dest and src1..src4 from dependency graph.
    Uses virtual register pool 1..15 with LIFO release.
    Returns peak live register count.

    CRITICAL: src4 = src_code[2] (not src_code[3]) — matches sd_sr_codegen quirk.
    """
    idle_reg_id = list(range(1, 16))[::-1]
    init_len = len(idle_reg_id)
    reg_used_count_max = 0
    occupy_list: List[List[int]] = []

    for i, code_dict in enumerate(code_list):
        dest_code: List[int] = []
        src_code: List[int] = []

        for one_code_num in code_dict["code_num"]:
            reg_id = idle_reg_id.pop()
            occupy_list.append([one_code_num, reg_id])
            dest_code.append(reg_id)

        assert 0 < len(dest_code) <= 2
        code_dict["dest"] = dest_code[0]

        reg_used_count = init_len - len(idle_reg_id)
        reg_used_count_max = max(reg_used_count_max, reg_used_count)

        for dep_ref in code_dict["dependency"]:
            for j in range(i - 1, -1, -1):
                if dep_ref in code_list[j]["code_num"]:
                    src_code.append(code_list[j]["dest"])
                    break

        assert len(src_code) <= 4
        code_dict["src1"] = src_code[0] if len(src_code) > 0 else 0
        code_dict["src2"] = src_code[1] if len(src_code) > 1 else 0
        code_dict["src3"] = src_code[2] if len(src_code) > 2 else 0
        code_dict["src4"] = src_code[2] if len(src_code) > 3 else 0  # intentional quirk

        for occ_pair in reversed(occupy_list):
            required = False
            upper = min(len(code_list), i + max_gap + 10)
            for k in range(i + 1, upper):
                if occ_pair[0] in code_list[k]["dependency"]:
                    required = True
                    break
            if not required:
                idle_reg_id.append(occ_pair[1])
                occupy_list.remove(occ_pair)

    return reg_used_count_max


def finalize_instructions(
    code_list: List[Dict[str, Any]],
    *,
    align_fields: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Mutates code_list in place. Returns stats dict."""
    if align_fields:
        align_instruction_fields(code_list)
    max_gap = add_instruction_dependencies(code_list)
    reg_max = assign_dependency_registers(code_list, max_gap)
    stats = {
        "max_gap": max_gap,
        "reg_used_count_max": reg_max,
        "num_instructions": len(code_list),
    }
    if verbose:
        print(stats)
    return stats


def strip_post_pass_fields(code_list: List[Dict[str, Any]]) -> None:
    """Remove post-pass fields for golden replay tests."""
    for c in code_list:
        for k in ("dependency", "dest", "src1", "src2", "src3", "src4"):
            c.pop(k, None)


def load_golden_file(path: str) -> List[Dict[str, Any]]:
    """Load golden pseudo-code file: one Python dict per line."""
    text = Path(path).read_text(encoding="utf-8")
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(ast.literal_eval(line))
    return out


def prepend_leading_code_num_padding(code_list: List[Dict[str, Any]]) -> int:
    """
    Golden files sometimes start at code_num > 0 (earlier ops omitted).
    Prepend stub rows so dependency indices resolve. Returns count prepended.
    """
    if not code_list:
        return 0
    nums = code_list[0].get("code_num") or [0]
    n0 = int(nums[0]) if nums else 0
    if n0 <= 0:
        return 0
    stubs = [{"code_num": [k], "op_code": "PaddingNoOp"} for k in range(n0)]
    code_list[:] = stubs + code_list
    return n0
