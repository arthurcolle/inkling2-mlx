"""Inkling 2 text decoder: hybrid sliding/full attention layers with attention residuals and MoE."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .cache import Cache, LayerCache
from .config import Inkling2Config, TextConfig
from .layers import (
    F32,
    BlockAttentionResidual,
    DynamicShortConvolution,
    L2Clip,
    RelativeLogits,
    RMSNorm,
    ScaledL2Clip,
    ShortConvolution,
    unweighted_rms_norm,
)
from .moe import SparseMoeBlock


class Attention(nn.Module):
    """Grouped-query attention with a content-conditioned relative bias, a per-query sink logit and
    an output gate. Sliding layers see the last `sliding_window_size` tokens; full layers see everything
    and take separate key/value inputs with dynamic short convolutions."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.window = config.sliding_window_size if self.is_sliding else None
        rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.scaling = self.head_dim**-0.5
        self.d_rel = config.d_rel
        self.log_n_floor = config.log_scaling_n_floor
        self.log_alpha = config.log_scaling_alpha
        self.eps = config.rms_norm_eps

        hidden, kv_dim = config.hidden_size, self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.r_proj = nn.Linear(hidden, self.num_heads * config.d_rel, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        if self.is_sliding:
            self.k_sconv = ShortConvolution(kv_dim, config.sconv_kernel_size)
            self.v_sconv = ShortConvolution(kv_dim, config.sconv_kernel_size)
        else:
            self.k_sconv = DynamicShortConvolution(kv_dim, hidden, self.num_kv_heads, config.sconv_kernel_size)
            self.v_sconv = DynamicShortConvolution(kv_dim, hidden, self.num_kv_heads, config.sconv_kernel_size)
        self.rel_logits_proj = RelativeLogits(config.d_rel, rel_extent)
        self.gate_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.q_l2_clip = ScaledL2Clip(self.num_heads, self.head_dim, config.qk_l2_clip_threshold)
        self.k_l2_clip = L2Clip(config.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(self.num_heads, config.d_rel, config.rel_l2_clip_threshold)

    def __call__(self, x: mx.array, key_x: mx.array, value_x: mx.array, cache: LayerCache | None, offset: int):
        batch, q_len, _ = x.shape
        heads, kv_heads, head_dim = self.num_heads, self.num_kv_heads, self.head_dim
        k_state = cache.conv[0] if cache is not None else None
        v_state = cache.conv[1] if cache is not None else None

        dtype = x.dtype  # quantized projections may return another float type
        queries = self.q_proj(x).astype(dtype)
        if self.is_sliding:
            keys, k_state = self.k_sconv(self.k_proj(key_x).astype(dtype), k_state)
            values, v_state = self.v_sconv(self.v_proj(value_x).astype(dtype), v_state)
        else:
            keys, k_state = self.k_sconv(self.k_proj(key_x).astype(dtype), key_x, k_state)
            values, v_state = self.v_sconv(self.v_proj(value_x).astype(dtype), value_x, v_state)
        relative = self.r_proj(x).astype(dtype)
        gate = self.gate_proj(x)
        sinks = self.sink_proj(x)

        q_pos = mx.arange(q_len) + offset
        queries = self.q_l2_clip(queries.reshape(batch, q_len, heads, head_dim))
        relative = self.rel_l2_clip(relative.reshape(batch, q_len, heads, self.d_rel))
        if not self.is_sliding:
            # log scaling of the queries and relative states beyond `log_scaling_n_floor` tokens, in fp32
            tau = 1.0 + self.log_alpha * mx.log(mx.maximum((q_pos + 1).astype(F32) / self.log_n_floor, 1.0))
            tau = tau[None, :, None, None]
            queries = (queries.astype(F32) * tau).astype(queries.dtype)
            relative = (relative.astype(F32) * tau).astype(relative.dtype)
        queries = queries.transpose(0, 2, 1, 3)
        keys = self.k_l2_clip(keys.reshape(batch, q_len, kv_heads, head_dim)).transpose(0, 2, 1, 3)
        values = unweighted_rms_norm(values.reshape(batch, q_len, kv_heads, head_dim), self.eps)
        values = values.transpose(0, 2, 1, 3)

        kv_offset = offset
        if cache is not None:
            if cache.keys is not None:
                keys = mx.concatenate([cache.keys, keys], axis=2)
                values = mx.concatenate([cache.values, values], axis=2)
                kv_offset = cache.kv_offset
        kv_len = keys.shape[2]
        kv_pos = mx.arange(kv_len) + kv_offset

        bias = self.rel_logits_proj(relative, q_pos, kv_pos)

        groups = heads // kv_heads
        grouped_q = queries.reshape(batch, kv_heads, groups, q_len, head_dim)
        scores = (grouped_q @ keys[:, :, None].swapaxes(-1, -2)) * self.scaling
        scores = scores.reshape(batch, heads, q_len, kv_len) + bias.astype(scores.dtype)
        if q_len > 1 or (self.window is not None and kv_len > self.window):
            allowed = kv_pos[None, :] <= q_pos[:, None]
            if self.window is not None:
                allowed = allowed & (kv_pos[None, :] > q_pos[:, None] - self.window)
            scores = mx.where(allowed[None, None], scores, mx.array(-mx.inf, dtype=scores.dtype))
        # the sink is one extra logit per query with a zero value: it only absorbs probability mass
        combined = mx.concatenate([scores, sinks.transpose(0, 2, 1)[..., None].astype(scores.dtype)], axis=-1)
        probs = mx.softmax(combined.astype(F32), axis=-1)[..., :-1].astype(queries.dtype)
        out = probs.reshape(batch, kv_heads, groups, q_len, kv_len) @ values[:, :, None]
        out = out.reshape(batch, heads, q_len, head_dim).transpose(0, 2, 1, 3).reshape(batch, q_len, -1)
        out = (out.astype(F32) * mx.sigmoid(gate.astype(F32))).astype(out.dtype)

        if cache is not None:
            if self.window is not None and kv_len > self.window - 1:
                drop = kv_len - (self.window - 1)
                keys, values, kv_offset = keys[:, :, drop:], values[:, :, drop:], kv_offset + drop
            cache.keys, cache.values, cache.kv_offset = keys, values, kv_offset
            cache.conv[0], cache.conv[1] = k_state, v_state
        return self.o_proj(out).astype(dtype)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, **expert_kwargs):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.is_sliding = config.is_sliding(layer_idx)
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        if layer_idx > 0:
            self.attention_input = BlockAttentionResidual(hidden, num_queries=1 if self.is_sliding else 3)
        self.has_attention_input = layer_idx > 0
        self.mlp_input = BlockAttentionResidual(hidden)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoeBlock(config, **expert_kwargs)
        if self.is_sliding:
            self.input_layernorm = RMSNorm(hidden, eps)
        else:
            self.q_input_layernorm = RMSNorm(hidden, eps)
            self.k_input_layernorm = RMSNorm(hidden, eps)
            self.v_input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.post_attention_convolution = ShortConvolution(hidden, config.sconv_kernel_size)
        self.post_mlp_convolution = ShortConvolution(hidden, config.sconv_kernel_size)

    def __call__(self, hidden: mx.array, block_states: list, cache: LayerCache | None, offset: int) -> mx.array:
        residual, mlp_in = self.attend(hidden, block_states, cache, offset)
        return self.finish(residual, self.mlp(mlp_in), cache)

    def attend(self, hidden: mx.array, block_states: list, cache: LayerCache | None, offset: int):
        """Attention half of the layer. Returns the fp32 residual stream and the input of the MoE block."""
        dtype = hidden.dtype
        block_residual = hidden
        if self.is_sliding:
            if self.has_attention_input:
                hidden = self.attention_input(block_states, hidden)[0]
            normed = self.input_layernorm(hidden).astype(dtype)
            q_in = k_in = v_in = normed
        else:
            if self.has_attention_input:
                q_h, k_h, v_h = self.attention_input(block_states, hidden)
            else:
                q_h = k_h = v_h = hidden
            q_in = self.q_input_layernorm(q_h).astype(dtype)
            k_in = self.k_input_layernorm(k_h).astype(dtype)
            v_in = self.v_input_layernorm(v_h).astype(dtype)

        hidden = self.self_attn(q_in, k_in, v_in, cache, offset)
        hidden, state = self.post_attention_convolution(hidden, cache.conv[2] if cache is not None else None)
        if cache is not None:
            cache.conv[2] = state

        hidden = hidden.astype(F32)
        if self.is_block_boundary:
            # the residual stream restarts here; the layer input becomes a block state instead
            block_states = block_states + [block_residual]
        else:
            hidden = block_residual.astype(F32) + hidden
        residual = hidden

        hidden = self.mlp_input(block_states, hidden)[0]
        return residual, self.post_attention_layernorm(hidden).astype(dtype)

    def finish(self, residual: mx.array, mlp_out: mx.array, cache: LayerCache | None = None) -> mx.array:
        """MoE half of the layer: short convolution over the expert output, then the residual."""
        dtype = mlp_out.dtype
        hidden, state = self.post_mlp_convolution(mlp_out, cache.conv[3] if cache is not None else None)
        if cache is not None:
            cache.conv[3] = state
        return (residual + hidden.astype(F32)).astype(dtype)


class NormedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden))
        self.embed_norm = RMSNorm(hidden, eps)
        self.quant = None  # (group_size, bits) when the table is stored affine-quantized

    def to_quantized(self, group_size: int, bits: int):
        rows, cols = self.weight.shape
        self.quant = (group_size, bits)
        self.weight = mx.zeros((rows, cols * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((rows, cols // group_size), dtype=mx.float16)
        self.biases = mx.zeros((rows, cols // group_size), dtype=mx.float16)

    def __call__(self, input_ids: mx.array) -> mx.array:
        if self.quant is None:
            return self.embed_norm(self.weight[input_ids])
        group_size, bits = self.quant
        flat = input_ids.reshape(-1)
        rows = mx.dequantize(self.weight[flat], self.scales[flat], self.biases[flat], group_size=group_size, bits=bits)
        rows = rows.reshape(*input_ids.shape, -1).astype(self.embed_norm.weight.dtype)
        return self.embed_norm(rows)


class TextModel(nn.Module):
    def __init__(self, config: Inkling2Config):
        super().__init__()
        text = config.text_config
        self.embed_tokens = NormedEmbedding(text.vocab_size, text.hidden_size, text.rms_norm_eps)
        self.layers = [
            DecoderLayer(
                text,
                i,
                fmt=config.expert_formats[i],
                affine_bits=config.affine_bits,
                affine_group_size=config.affine_group_size,
                act_format=config.activation_format(i),
                interleaved=config.gate_up_interleaved,
                num_experts=config.experts_in_layer(i),
            )
            for i in range(text.num_hidden_layers)
        ]
        self.norm = RMSNorm(text.hidden_size, text.rms_norm_eps)
        self.final_input = BlockAttentionResidual(text.hidden_size)

    def __call__(self, inputs_embeds: mx.array, cache: Cache | None = None) -> mx.array:
        offset = cache.offset if cache is not None else 0
        hidden = inputs_embeds
        block_states: list = []
        for i, layer in enumerate(self.layers):
            new_hidden = layer(hidden, block_states, cache.layers[i] if cache is not None else None, offset)
            if layer.is_block_boundary:
                block_states = block_states + [hidden]
            hidden = new_hidden
        if cache is not None:
            cache.offset = offset + inputs_embeds.shape[1]
        hidden = self.final_input(block_states, hidden)[0]
        return self.norm(hidden).astype(inputs_embeds.dtype)
