"""Building blocks shared by the Inkling 2 decoder layers."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

F32 = mx.float32


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(F32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (self.weight.astype(F32) * x).astype(dtype)


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    xf = x.astype(F32)
    return x * mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + eps).astype(x.dtype)


def l2_clip(x: mx.array, threshold: float) -> mx.array:
    """Scale each vector down to `threshold` when its L2 norm is larger. Input and output are fp32."""
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return x * mx.minimum(threshold / (norm + 1e-6), 1.0)


class L2Clip(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, x: mx.array) -> mx.array:
        return l2_clip(x.astype(F32), self.threshold).astype(x.dtype)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        return l2_clip(x.astype(F32) * self.scale.astype(F32), self.threshold).astype(x.dtype)


def softcap(x: mx.array, cap: float) -> mx.array:
    return mx.tanh(x / cap) * cap


class RelativeLogits(nn.Module):
    """Content-conditioned relative position bias: each query mixes a bank of bias-vs-distance profiles."""

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states: mx.array, q_pos: mx.array, kv_pos: mx.array) -> mx.array:
        # relative_states [B, Q, H, d_rel] -> bias [B, H, Q, K]
        rel = (relative_states @ self.proj.astype(relative_states.dtype)).transpose(0, 2, 1, 3)
        distance = q_pos[:, None] - kv_pos[None, :]
        index = mx.clip(distance, 0, self.rel_extent - 1)
        index = mx.broadcast_to(index[None, None], (*rel.shape[:2], *index.shape))
        bias = mx.take_along_axis(rel, index, axis=-1)
        valid = (distance >= 0) & (distance < self.rel_extent)
        return mx.where(valid[None, None], bias, mx.zeros_like(bias))


class _ConvWeight(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = mx.zeros((channels, 1, kernel_size))  # torch depthwise conv1d layout


def _history(state, batch: int, kernel_size: int, channels: int) -> mx.array:
    return state if state is not None else mx.zeros((batch, kernel_size - 1, channels), dtype=F32)


class ShortConvolution(nn.Module):
    """Causal depthwise convolution over the last `kernel_size` tokens, plus a residual.

    `state` holds the previous `kernel_size - 1` inputs (zeros at the start of a sequence).
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = _ConvWeight(channels, kernel_size)

    def __call__(self, x: mx.array, state=None):
        dtype = x.dtype
        batch, seq_len, channels = x.shape
        k = self.kernel_size
        xf = x.astype(F32)
        padded = mx.concatenate([_history(state, batch, k, channels), xf], axis=1)
        w = self.conv1d.weight.astype(F32)
        out = padded[:, 0:seq_len, :] * w[:, 0, 0]
        for j in range(1, k):  # tap k - 1 weights the current token
            out = out + padded[:, j : j + seq_len, :] * w[:, 0, j]
        out = (out + xf) / math.sqrt(k + 1)
        return out.astype(dtype), padded[:, -(k - 1) :, :]


class DynamicShortConvolution(nn.Module):
    """Short causal convolution whose taps are static per-channel weights plus per-token weights
    projected from `dynamic_input` and shared within each key/value head."""

    def __init__(self, channels: int, input_size: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, channels))  # tap index = backward distance
        self.weight_proj = nn.Linear(input_size, num_heads * kernel_size, bias=False)

    def __call__(self, x: mx.array, dynamic_input: mx.array, state=None):
        dtype = x.dtype
        batch, seq_len, channels = x.shape
        k, heads = self.kernel_size, self.num_heads
        dynamic = self.weight_proj(dynamic_input).astype(F32).reshape(batch, seq_len, heads, 1, k)
        weights = self.static_weight.astype(F32).T.reshape(heads, channels // heads, k) + dynamic
        xf = x.astype(F32)
        padded = mx.concatenate([_history(state, batch, k, channels), xf], axis=1)
        # windows[..., lag] is the token `lag` steps back
        windows = mx.stack([padded[:, k - 1 - lag : k - 1 - lag + seq_len, :] for lag in range(k)], axis=-1)
        windows = windows.reshape(batch, seq_len, heads, channels // heads, k)
        out = (windows * weights).sum(axis=-1).reshape(batch, seq_len, channels)
        out = (out + xf) / math.sqrt(2 * k + 1)
        return out.astype(dtype), padded[:, -(k - 1) :, :]


class BlockAttentionResidual(nn.Module):
    """Learned queries attend over the RMS-normalised block states and the running hidden state."""

    def __init__(self, hidden_size: int, num_queries: int = 1):
        super().__init__()
        self.num_queries = num_queries
        self.query = mx.zeros((hidden_size,)) if num_queries == 1 else mx.zeros((num_queries, hidden_size))

    def __call__(self, block_states: list, partial: mx.array) -> list:
        stacked = mx.stack([s.astype(F32) for s in block_states] + [partial.astype(F32)], axis=-2)
        stacked = unweighted_rms_norm(stacked)  # [B, T, S, H]
        hidden = stacked.shape[-1]
        query = self.query.astype(F32).reshape(-1, hidden)  # [Q, H]
        scores = (stacked @ query.T) * hidden**-0.5  # [B, T, S, Q]
        probs = mx.softmax(scores, axis=-2)
        out = probs.swapaxes(-1, -2) @ stacked  # [B, T, Q, H]
        return [out[..., i, :] for i in range(self.num_queries)]
