"""Sparse mixture-of-experts feed-forward block.

The routed experts are stored in one of four formats:

* ``float``  - plain floating point (testing).
* ``nvfp4``  - E2M1 codes, one E4M3 scale per 16 inputs, one fp32 outer scale per expert. This is the
               native storage of layers 0-39 of the released checkpoint and is repacked bit-for-bit.
* ``mxfp8``  - E4M3 values, one E8M0 scale per 32 inputs. Native storage of layers 40-47.
* ``affine`` - MLX affine quantization (a lossy re-quantization to fewer bits).

The checkpoint was trained with the inputs of every expert matmul quantized to the same format as that
layer's weights. ``act_format`` reproduces that arithmetic, whatever format the weights are stored in.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import TextConfig
from .layers import F32, softcap

# routes (tokens x experts-per-token) up to which experts are gathered into one batched matmul
GATHER_LIMIT = 64


def _round_half_even(x: mx.array) -> mx.array:
    r = mx.floor(x + 0.5)
    tie = (x + 0.5) == r
    odd = mx.remainder(r, 2.0) == 1.0
    return mx.where(tie & odd, r - 1.0, r)


def _e4m3_round(v: mx.array) -> mx.array:
    """Round non-negative fp32 to the nearest E4M3 value, saturating at 448."""
    v = mx.minimum(v, 448.0)
    e = mx.clip(mx.floor(mx.log2(mx.maximum(v, 2.0**-9))), -6, 8)
    step = mx.power(2.0, e - 3)
    return mx.minimum(_round_half_even(v / step) * step, 448.0)


def _e2m1_round(a: mx.array) -> mx.array:
    """Round non-negative fp32 to the nearest E2M1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}."""
    a = mx.minimum(a, 6.0)
    low = _round_half_even(a * 2.0) * 0.5
    mid = 2.0 + _round_half_even(a - 2.0)
    high = 4.0 + 2.0 * _round_half_even((a - 4.0) * 0.5)
    return mx.where(a < 2.0, low, mx.where(a < 4.0, mid, high))


def nvfp4_fake_quant(x: mx.array) -> mx.array:
    """Quantize-dequantize rows of `x` to NVFP4 with a per-row outer scale and the four-over-six rule:
    each block of 16 keeps whichever of "block max -> 6" and "block max -> 4" has the lower L1 error."""
    n, k = x.shape
    blocks = x.reshape(n, k // 16, 16)
    mag = mx.abs(blocks)
    outer = mx.max(mag, axis=(1, 2), keepdims=True) / 1536.0
    block_max = mx.max(mag, axis=-1, keepdims=True)
    safe_outer = mx.where(outer == 0, 1.0, outer)

    def candidate(target: float):
        scale = outer * _e4m3_round(mx.where(outer == 0, 0.0, (block_max / target) / safe_outer))
        codes = mx.where(scale == 0, 0.0, _e2m1_round(mag / mx.where(scale == 0, 1.0, scale)))
        value = codes * scale
        return value, mx.sum(mx.abs(value - mag), axis=-1, keepdims=True)

    six, err_six = candidate(6.0)
    four, err_four = candidate(4.0)
    out = mx.where(err_four < err_six, four, six)
    return (mx.sign(blocks) * out).reshape(n, k)


def mxfp8_fake_quant(x: mx.array) -> mx.array:
    """Quantize-dequantize rows of `x` to MXFP8: E4M3 values with a power-of-two scale per 32 inputs."""
    n, k = x.shape
    blocks = x.reshape(n, k // 32, 32)
    mag = mx.abs(blocks)
    block_max = mx.max(mag, axis=-1, keepdims=True)
    scale = mx.power(2.0, mx.floor(mx.log2(mx.where(block_max > 0, block_max, 1.0))) - 8.0)
    return (mx.sign(blocks) * _e4m3_round(mag / scale) * scale).reshape(n, k)


class TopKRouter(nn.Module):
    def __init__(self, config: TextConfig, num_experts: int | None = None):
        super().__init__()
        num_experts = num_experts or config.n_routed_experts
        self.top_k = min(config.num_experts_per_tok, num_experts)
        self.route_scale = config.route_scale
        self.weight = mx.zeros((num_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((num_experts,))

    def __call__(self, x: mx.array):
        scores = mx.sigmoid(x.astype(F32) @ self.weight.astype(F32).T)
        biased = scores + self.e_score_correction_bias.astype(F32)
        index = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        weights = mx.take_along_axis(scores, index, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return index, weights * self.route_scale


_QUANT = {
    "nvfp4": {"mode": "nvfp4", "group_size": 16, "bits": 4},
    "mxfp8": {"mode": "mxfp8", "group_size": 32, "bits": 8},
}


class Experts(nn.Module):
    """Stacked expert weights: gate_up_proj [E, 2I, H] and down_proj [E, H, I]."""

    def __init__(self, config: TextConfig, fmt: str = "float", affine_bits=None, affine_group_size=None,
                 act_format: str | None = None, interleaved: bool = False, num_experts: int | None = None):
        super().__init__()
        e, h, i = num_experts or config.n_routed_experts, config.hidden_size, config.intermediate_size
        self.format = fmt
        # original checkpoints store gate_up_proj rows as gate_0, up_0, gate_1, up_1, ...
        self.interleaved = interleaved
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        # format the inputs of the expert matmuls are quantized to: "nvfp4", "mxfp8" or None
        self.act_quant = act_format
        if fmt == "float":
            self.gate_up_proj = mx.zeros((e, 2 * i, h))
            self.down_proj = mx.zeros((e, h, i))
            return
        if fmt in _QUANT:
            self.q = dict(_QUANT[fmt])
        elif fmt == "affine":
            self.q = {"mode": "affine", "group_size": affine_group_size, "bits": affine_bits}
        else:
            raise ValueError(f"unknown expert format {fmt!r}")
        bits, group = self.q["bits"], self.q["group_size"]
        scale_dtype = mx.uint8 if fmt in _QUANT else mx.float16
        for name, rows, cols in (("gate_up_proj", 2 * i, h), ("down_proj", h, i)):
            setattr(self, name, mx.zeros((e, rows, cols * bits // 32), dtype=mx.uint32))
            setattr(self, name + "_scales", mx.zeros((e, rows, cols // group), dtype=scale_dtype))
            if fmt == "affine":
                setattr(self, name + "_biases", mx.zeros((e, rows, cols // group), dtype=scale_dtype))
            if fmt == "nvfp4":
                setattr(self, name + "_outer_scale", mx.ones((e,), dtype=F32))

    def _quantize_input(self, x: mx.array) -> mx.array:
        if self.act_quant == "nvfp4":
            return nvfp4_fake_quant(x)
        if self.act_quant == "mxfp8":
            return mxfp8_fake_quant(x)
        return x

    def _matmul(self, name: str, x: mx.array, index) -> mx.array:
        """x @ W[index].T. `index` is one expert (x [n, in]) or an array of experts (x [m, 1, in])."""
        weight = self[name][index]
        if self.format == "float":
            return x @ weight.astype(F32).swapaxes(-1, -2)
        biases = self[name + "_biases"][index] if self.format == "affine" else None
        out = mx.quantized_matmul(x, weight, scales=self[name + "_scales"][index], biases=biases,
                                  transpose=True, **self.q)
        if self.format == "nvfp4":
            outer = self[name + "_outer_scale"][index]
            out = out * (outer[:, None, None] if outer.ndim else outer)
        return out

    def _expert(self, x: mx.array, index) -> mx.array:
        """`x` is the (already input-quantized) hidden state of the tokens routed to `index`."""
        gate_up = self._matmul("gate_up_proj", x, index).astype(F32)
        if self.interleaved:
            pairs = gate_up.reshape(*gate_up.shape[:-1], -1, 2)
            gate, up = pairs[..., 0], pairs[..., 1]
        else:
            gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = softcap(gate, self.gate_cap) * mx.sigmoid(gate) * softcap(up, self.up_cap)
        hshape = hidden.shape
        hidden = self._quantize_input(hidden.reshape(-1, hshape[-1])).reshape(hshape)
        return self._matmul("down_proj", hidden, index).astype(F32)

    def __call__(self, x: mx.array, top_k_index: mx.array, top_k_weights: mx.array) -> mx.array:
        # x [N, H]; top_k_index, top_k_weights [N, k]. Expert outputs are accumulated in fp32.
        n, k = top_k_index.shape
        x = self._quantize_input(x.astype(F32))
        if n * k <= GATHER_LIMIT:
            routed = mx.repeat(x, k, axis=0)[:, None, :]  # [N*k, 1, H]
            out = self._expert(routed, top_k_index.reshape(-1))[:, 0, :]
            return mx.sum(out.reshape(n, k, -1) * top_k_weights[..., None], axis=1)

        # group routes by expert so each expert runs once over all of its tokens
        flat = np.array(top_k_index).reshape(-1)
        order = np.argsort(flat, kind="stable")
        experts = flat[order]
        tokens = order // k
        starts = np.concatenate([[0], np.flatnonzero(np.diff(experts)) + 1, [len(experts)]])
        parts = []
        for a, b in zip(starts[:-1], starts[1:]):
            parts.append(self._expert(x[mx.array(tokens[a:b])], int(experts[a])))
        out = mx.concatenate(parts, axis=0)[mx.array(np.argsort(order))]
        return mx.sum(out.reshape(n, k, -1) * top_k_weights[..., None], axis=1)


class SparseMoeBlock(nn.Module):
    def __init__(self, config: TextConfig, **expert_kwargs):
        super().__init__()
        self.experts = Experts(config, **expert_kwargs)
        self.gate = TopKRouter(config, expert_kwargs.get("num_experts"))

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        index, weights = self.gate(flat)
        return self.experts(flat, index, weights).reshape(batch, seq_len, hidden).astype(x.dtype)
