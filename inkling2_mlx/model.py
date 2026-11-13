"""Inkling 2 for MLX: vision tower + text decoder + LM head."""

from __future__ import annotations

import glob
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .cache import Cache
from .config import Inkling2Config
from .layers import F32, softcap
from .text import TextModel
from .vision import VisionModel

# weights the reference keeps in fp32 even when the model runs in bf16
_FP32_SUFFIXES = (
    "mlp.gate.weight",
    "e_score_correction_bias",
    "conv1d.weight",
    "attention_input.query",
    "mlp_input.query",
    "final_input.query",
)


class _Body(nn.Module):
    def __init__(self, config: Inkling2Config):
        super().__init__()
        self.language_model = TextModel(config)
        self.vision_tower = VisionModel(config.vision_config, config.text_config.hidden_size)


class Model(nn.Module):
    def __init__(self, config: Inkling2Config):
        super().__init__()
        self.config = config
        self.model = _Body(config)
        text = config.text_config
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def make_cache(self) -> Cache:
        return Cache(self.config.text_config.num_hidden_layers)

    def get_image_features(self, pixel_values: mx.array, image_grid_thw) -> mx.array:
        return self.model.vision_tower(pixel_values, image_grid_thw)

    def embed(self, input_ids: mx.array, pixel_values=None, image_grid_thw=None) -> mx.array:
        """Token embeddings [B, T, H] with image features written over the image placeholder tokens."""
        embeds = self.model.language_model.embed_tokens(input_ids)
        if pixel_values is None:
            return embeds
        features = self.get_image_features(pixel_values, image_grid_thw).astype(embeds.dtype)
        mask = input_ids == self.config.image_token_id
        count = int(mask.sum().item())
        if count != features.shape[0]:
            raise ValueError(f"{count} image placeholder tokens but {features.shape[0]} image features")
        # position of each placeholder among the placeholders, in reading order
        slot = mx.cumsum(mask.reshape(-1).astype(mx.int32)) - 1
        gathered = features[mx.maximum(slot, 0)].reshape(*input_ids.shape, -1)
        return mx.where(mask[..., None], gathered, embeds)

    def __call__(self, input_ids: mx.array | None = None, cache: Cache | None = None, pixel_values=None,
                 image_grid_thw=None, inputs_embeds: mx.array | None = None) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids, pixel_values, image_grid_thw)
        hidden = self.model.language_model(inputs_embeds, cache)
        logits = softcap(self.lm_head(hidden).astype(F32), self.config.text_config.final_logit_softcapping)
        return logits[..., : self.config.text_config.unpadded_vocab_size]

    def prepare_quantized(self, names) -> None:
        """Swap in quantized modules for every matrix that `names` (weight keys) stores quantized."""
        names = set(names)
        cfg = self.config
        if cfg.linear_bits is None:
            return
        nn.quantize(self, group_size=cfg.linear_group_size, bits=cfg.linear_bits,
                    class_predicate=lambda path, module: isinstance(module, nn.Linear) and f"{path}.scales" in names)
        embed = self.model.language_model.embed_tokens
        if "model.language_model.embed_tokens.scales" in names and embed.quant is None:
            embed.to_quantized(cfg.linear_group_size, cfg.linear_bits)

    def load_safetensors(self, path, strict: bool = True, max_experts: int | None = None):
        weights = {}
        for file in sorted(glob.glob(str(Path(path) / "model-*.safetensors"))) or sorted(glob.glob(str(Path(path) / "*.safetensors"))):
            weights.update(mx.load(file))
        for key in list(weights):
            if key.endswith(_FP32_SUFFIXES) and weights[key].dtype != F32:
                weights[key] = weights[key].astype(F32)
        if max_experts is not None:
            # experts are stored most-important first, so a prefix is a smaller model
            for key in list(weights):
                if ".mlp.experts." in key or ".mlp.gate." in key:
                    weights[key] = weights[key][:max_experts]
        self.prepare_quantized(weights)
        self.load_weights(list(weights.items()), strict=strict)
        self.eval()
        return self


def load_model(path, max_experts: int | None = None) -> Model:
    """Build the model described by `path/config.json` and attach the weights (lazily).

    `max_experts` keeps only the first N experts of every layer, trading quality for memory.
    """
    config = Inkling2Config.from_pretrained(path)
    if max_experts is not None:
        counts = [config.experts_in_layer(i) for i in range(config.text_config.num_hidden_layers)]
        config.experts_per_layer = [min(c, max_experts) for c in counts]
    return Model(config).load_safetensors(path, max_experts=max_experts)
