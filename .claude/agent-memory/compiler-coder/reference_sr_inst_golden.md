---
name: sr_inst() golden reference for FSRCNN
description: How to regenerate sr_inst() standalone golden and the key shape/state assumptions it makes
type: reference
---

`sr_inst()` lives in `references/sd_sr_codegen.py` lines 2488–3676. It emits the FSRCNN instruction stream for a single 32×32 spatial block on top of the module-level `Inst.code_list`.

Key assumptions (important for any comparison):
- Input spatial size: **32×32** (cal_total_num in simple-conv layers == 32; deformable-conv layers use cal_total_num = 32//4 = 8). DataLoader bas_addr starts at `144*4 = 576` (past the preceding UNet/offchip region).
- 12 hardware layers, numbered 0–11 in the golden: L0 first_part (3×3, 1→32), L1 mid_part_1[0] 1×1 (32→8), L2/L4/L6/L8 offset-gen, L3/L5/L7/L9 deformable-conv, L10 mid_part_2[-1] 1×1 (8→32), L11 last_part 3×3 (32→4 + pixelshuffle).
- `load_next` only gates a single `OffchipDataLoader` at the end (load the next tile's image) and is NOT a preamble — sr_inst() has no first/mid/last variants.
- Init state assumed at entry: `weightloadermanager.bas_addr_cur = [1737, 792, 1152]`, `quantloadermanager.bas_addr_cur = 665`. These are the post-UNet weight/quant offsets.
- Final instruction is always `OffchipDataStorer(src_buffer='fsrcnn_output_buffer', transnum=1024, base_addr=0)`.

Regeneration recipe (standalone, no sd_inst contamination):
```bash
cd /home/hansz/scratch-data/design/tvm-tiling/references && \
PYTHONPATH=/home/hansz/scratch-data/design/tvm/python:/home/hansz/scratch-data/design/tvm-tiling \
/home/hansz/scratch-data/tools/miniconda3/envs/hhb/bin/python -c "
import sd_sr_codegen
sd_sr_codegen.sr_inst(load_next=False)
from instruction import Inst
with open('/tmp/sr_inst_golden.txt','w') as f:
    for c in Inst.code_list: f.write(str(c)+'\n')
"
```

Expected totals with load_next=False: 1273 instructions. Opcodes: QuantLoader 12, DataLoader 524, WeightLoader 524, DataStorer 116, OffsetLoader 96, OffchipDataStorer 1. (With load_next=True there is also 1 OffchipDataLoader.)
