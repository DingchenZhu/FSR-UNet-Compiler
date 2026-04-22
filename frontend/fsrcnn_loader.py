"""FSRCNN model loader — exposes get_model() for pipeline.py's pytorch path.

Usage in pipeline:
    python3 pipeline.py --model frontend/fsrcnn_loader.py --type pytorch \
        --input-shape 1 1 36 64 --output-dir output/fsrcnn/

The model file for FSRCNN (models_new_930.py) uses torchvision.ops.deform_conv2d
in its forward() — already traceable, no patching needed.
TVM relay.frontend.from_pytorch converts torchvision::deform_conv2d to
nn.deformable_conv2d automatically; layer_desc.py extracts it as op='deformable_conv2d'.
"""
import sys
import os

# Add the references directory to path so models_new_930 can be imported
_REFS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "tvm-tiling", "references",
)
if _REFS_DIR not in sys.path:
    sys.path.insert(0, _REFS_DIR)

from models_new_930 import FSRCNN


def get_model(scale_factor: int = 2, num_channels: int = 1, d: int = 32, s: int = 8, m: int = 4):
    """Return an eval-mode FSRCNN model instance."""
    model = FSRCNN(
        scale_factor=scale_factor,
        num_channels=num_channels,
        d=d,
        s=s,
        m=m,
    )
    model.eval()
    return model
