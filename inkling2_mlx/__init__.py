"""MLX runtime for Thinking Machines' Inkling 2.

    convert   repack a released checkpoint (NVFP4 / MXFP8 experts) into MLX safetensors, bit for bit,
              or re-quantize the experts to fewer bits
    prune     keep only the experts each layer's router uses most, fetching nothing else
    generate  chat / image+text generation
    validate  teacher-forced check of a converted checkpoint, one layer in memory at a time

Status: the architecture, cached decoding, vision tower, prompt construction and both tools are tested
against the Hugging Face Transformers reference on a small random model (see tests/). Nothing here has
been run on the full released weights yet.
"""

from .cache import Cache
from .config import Inkling2Config, TextConfig, VisionConfig
from .model import Model, load_model

__all__ = ["Cache", "Inkling2Config", "TextConfig", "VisionConfig", "Model", "load_model"]
__version__ = "0.1.0"
