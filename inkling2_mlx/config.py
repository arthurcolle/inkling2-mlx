"""Configuration for Inkling 2, parsed from the original checkpoint's config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


def _pick(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TextConfig:
    hidden_size: int = 4608
    num_hidden_layers: int = 48
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 48
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 128000
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    local_layer_ids: list | None = None
    sconv_kernel_size: int = 4
    final_logit_softcapping: float = 30.0
    n_layers_per_attn_res_block: int = 6
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8
    intermediate_size: int = 2304
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    model_max_length: int = 1048576

    def __post_init__(self):
        if self.local_layer_ids is None:
            self.local_layer_ids = [i for i in range(self.num_hidden_layers) if (i + 1) % 6 != 0]
        self.local_layer_ids = [int(i) for i in self.local_layer_ids]

    def is_sliding(self, layer_idx: int) -> bool:
        return layer_idx in (self.local_layer_ids or ())


@dataclass
class VisionConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    native_patch_size: int = 20
    adapter_merge_size: int = 2
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-5
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    rope_theta: float = 10000.0

    @property
    def intermediate_size(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)


# Architecture switches the original config carries. This port implements exactly one setting of each.
_EXPECTED_TEXT_FLAGS = {
    "q_bias": False,
    "o_bias": False,
    "pre_norm_pre_affine": False,
    "final_norm_pre_affine": False,
    "use_value_norm_no_gamma": True,
    "attn_res_normalize_values": True,
    "n_shared_experts": 0,
    "use_gate_bias": True,
}


@dataclass
class Inkling2Config:
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    eos_token_id: int = 200006
    image_token_id: int = 200054
    # per-layer storage format of the routed experts: "float", "nvfp4", "mxfp8" or "affine"
    expert_formats: list | None = None
    affine_bits: int | None = None
    affine_group_size: int | None = None
    # emulate the activation quantization the checkpoint was trained with (see moe.py); the format
    # of each layer is the one its experts were released in
    activation_quantization: bool = True
    activation_formats: list | None = None
    # gate_up_proj rows stored gate_0, up_0, gate_1, ... (original layout) instead of all gates then all ups
    gate_up_interleaved: bool = False
    # experts kept in each layer when the checkpoint was pruned (None: all of them)
    experts_per_layer: list | None = None
    # affine quantization of the large non-expert matrices (attention projections, embeddings, LM head)
    linear_bits: int | None = None
    linear_group_size: int = 64

    @classmethod
    def from_dict(cls, d: dict) -> "Inkling2Config":
        text = d.get("text_config", {})
        for key, want in _EXPECTED_TEXT_FLAGS.items():
            if key in text and text[key] != want:
                raise ValueError(f"text_config.{key}={text[key]!r} is not supported (expected {want!r})")
        if d.get("vision_config", {}).get("projector_bias", False):
            raise ValueError("vision_config.projector_bias=True is not supported")
        q = d.get("quantization", {}) or {}
        n_layers = text.get("num_hidden_layers", 48)
        return cls(
            text_config=_pick(TextConfig, text),
            vision_config=_pick(VisionConfig, d.get("vision_config", {})),
            eos_token_id=d.get("eos_token_id", 200006),
            image_token_id=d.get("image_token_id", 200054),
            expert_formats=q.get("expert_formats") or ["float"] * n_layers,
            affine_bits=q.get("affine_bits"),
            affine_group_size=q.get("affine_group_size"),
            activation_quantization=q.get("activation_quantization", True),
            activation_formats=q.get("activation_formats"),
            gate_up_interleaved=q.get("gate_up_interleaved", False),
            experts_per_layer=q.get("experts_per_layer"),
            linear_bits=q.get("linear_bits"),
            linear_group_size=q.get("linear_group_size", 64),
        )

    def experts_in_layer(self, layer_idx: int) -> int:
        if self.experts_per_layer:
            return int(self.experts_per_layer[layer_idx])
        return self.text_config.n_routed_experts

    def activation_format(self, layer_idx: int):
        if not self.activation_quantization or not self.activation_formats:
            return None
        return self.activation_formats[layer_idx]

    @classmethod
    def from_pretrained(cls, path) -> "Inkling2Config":
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text()))
