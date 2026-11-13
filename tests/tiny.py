"""A small random Inkling 2 built with the transformers reference, exported for the MLX port."""
import os
import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
from transformers.models.inkling2.modeling_inkling2 import Inkling2ForConditionalGeneration

TEXT = dict(
    vocab_size=512, unpadded_vocab_size=500, hidden_size=64, num_hidden_layers=12,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    swa_num_attention_heads=8, swa_num_key_value_heads=4, swa_head_dim=8,
    sliding_window_size=8, d_rel=4, rel_extent=6, log_scaling_n_floor=4, log_scaling_alpha=0.1,
    conv_kernel_size=4, moe_intermediate_size=32, n_routed_experts=16, num_experts_per_tok=4,
    n_layers_per_attn_res_block=6, qk_l2_clip_threshold=1.0, rel_l2_clip_threshold=0.5,
    mlp_gate_softcap=0.5, mlp_up_softcap=0.8, final_logit_softcapping=2.0, route_scale=8.0,
)
VISION = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, patch_size=4, spatial_merge_size=2,
              posemb_grid_size=4, projection_hidden_dim=48, text_hidden_size=64)
IMAGE_TOKEN = 7

# the same model in the original checkpoint's config vocabulary, as the MLX port reads it
MLX_CONFIG = {
    "model_type": "inkling2",
    "eos_token_id": 200006,
    "image_token_id": IMAGE_TOKEN,
    "text_config": {
        "hidden_size": 64, "num_hidden_layers": 12, "vocab_size": 512, "unpadded_vocab_size": 500,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "swa_num_attention_heads": 8, "swa_num_key_value_heads": 4, "swa_head_dim": 8,
        "sliding_window_size": 8, "d_rel": 4, "rel_extent": 6, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
        "sconv_kernel_size": 4, "intermediate_size": 32, "n_routed_experts": 16, "num_experts_per_tok": 4,
        "n_layers_per_attn_res_block": 6, "qk_l2_clip_threshold": 1.0, "rel_l2_clip_threshold": 0.5,
        "mlp_gate_softcap": 0.5, "mlp_up_softcap": 0.8, "final_logit_softcapping": 2.0, "route_scale": 8.0,
        "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
    },
    "vision_config": {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "native_patch_size": 4,
                      "adapter_merge_size": 2, "posemb_grid_size": 4, "projection_hidden_dim": 48},
}


def build_reference(seed: int = 0, smooth: bool = False):
    """`smooth` shrinks the routed/queried components so the random network is not chaotic, which is what
    a comparison under bf16 rounding needs."""
    torch.manual_seed(seed)
    config = Inkling2Config(text_config=TEXT, vision_config=VISION, image_token_id=IMAGE_TOKEN)
    model = Inkling2ForConditionalGeneration(config).eval().float()
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if "inv_freq" in name:
                continue
            leaf = name.split(".")[-1]
            if "norm" in name and leaf == "weight" and p.ndim == 1:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif leaf == "scale":
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif leaf in ("lambda_1", "lambda_2"):
                p.copy_(0.5 + 0.2 * torch.randn_like(p))
            elif leaf == "query":
                p.copy_((0.2 if smooth else 2.0) * torch.randn_like(p))
            elif leaf == "e_score_correction_bias":
                p.copy_(0.2 * torch.randn_like(p))
            elif leaf in ("gate_up_proj", "down_proj"):
                p.copy_((0.03 if smooth else 0.3) * torch.randn_like(p))
            elif smooth and leaf == "weight" and name.endswith("mlp.gate.weight"):
                p.copy_(4.0 * torch.randn_like(p) / np.sqrt(p.shape[-1]))  # decisive routing
            elif p.ndim == 1:
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (1.0 / np.sqrt(p.shape[-1])))
    return model


def export_state(model, directory):
    import json
    os.makedirs(directory, exist_ok=True)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if "inv_freq" not in k}
    save_file(state, os.path.join(directory, "model.safetensors"))
    json.dump(MLX_CONFIG, open(os.path.join(directory, "config.json"), "w"), indent=1)
    return state
