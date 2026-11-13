"""Inkling 2 vision tower: a ViT over native-resolution patches with axial RoPE, a class token per
image, and a merger that concatenates each 2x2 block of patch states into one language-model token."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig
from .layers import F32


def _axis_taps(index: np.ndarray, size: np.ndarray, side: int):
    """Bilinear taps (align_corners=False, border padding) into a `side`-long table."""
    src = (index.astype(np.float32) + 0.5) * side / size.astype(np.float32) - 0.5
    floor = np.floor(src)
    offsets = np.arange(2)
    raw = floor.astype(np.int64)[:, None] + offsets
    distance = np.abs(src[:, None] - floor[:, None] - offsets)
    return np.clip(raw, 0, side - 1), np.clip(1 - distance, 0, None).astype(np.float32)


def _patch_rows_cols(grid_thw: np.ndarray, merge: int):
    """(row, col, height, width) of every native patch, images concatenated, in merge-block order."""
    rows, cols, heights, widths = [], [], [], []
    for t, h, w in grid_thw.tolist():
        r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        block = (h // merge, merge, w // merge, merge)
        r = r.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
        c = c.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
        rows.append(np.tile(r, t))
        cols.append(np.tile(c, t))
        heights.append(np.full(t * h * w, h))
        widths.append(np.full(t * h * w, w))
    return (np.concatenate(a) for a in (rows, cols, heights, widths))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, bounds: list, cos: mx.array, sin: mx.array) -> mx.array:
        seq_len = x.shape[0]
        shape = (seq_len, self.num_heads, self.head_dim)
        q, k, v = self.q_proj(x).reshape(shape), self.k_proj(x).reshape(shape), self.v_proj(x).reshape(shape)

        def rotate(t):
            dtype = t.dtype
            t = t.astype(F32)
            pairs = t.reshape(seq_len, self.num_heads, -1, 2)
            rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(shape)
            return (t * cos[:, None, :] + rotated * sin[:, None, :]).astype(dtype)

        q, k = rotate(q), rotate(k)
        outputs = []
        for start, end in bounds:  # images do not attend to each other
            qi, ki, vi = (t[start:end].transpose(1, 0, 2) for t in (q, k, v))
            weights = mx.softmax(((qi @ ki.swapaxes(-1, -2)) * self.scaling).astype(F32), axis=-1).astype(qi.dtype)
            outputs.append((weights @ vi).transpose(1, 0, 2))
        return self.out_proj(mx.concatenate(outputs, axis=0).reshape(seq_len, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lambda_1 = mx.ones((config.hidden_size,))
        self.lambda_2 = mx.ones((config.hidden_size,))

    def __call__(self, x: mx.array, bounds: list, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.lambda_1 * self.self_attn(self.layernorm_before(x), bounds, cos, sin)
        return x + self.lambda_2 * self.mlp(self.layernorm_after(x))


class _Embedding(nn.Module):
    def __init__(self, num: int, dim: int):
        super().__init__()
        self.weight = mx.zeros((num, dim))


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        patch = config.native_patch_size
        self.patch_embedding = nn.Linear(config.num_channels * patch * patch, config.hidden_size, bias=False)
        self.position_embedding = _Embedding(config.posemb_grid_size**2, config.hidden_size)
        self.class_embedding = mx.zeros((config.hidden_size,))
        self.cls_position_embedding = mx.zeros((config.hidden_size,))


class VisionMerger(nn.Module):
    def __init__(self, config: VisionConfig, text_hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(config.adapter_merge_size**2 * config.hidden_size, config.projection_hidden_dim, bias=False)
        self.linear_2 = nn.Linear(config.projection_hidden_dim, text_hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.gelu(self.linear_1(x)))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig, text_hidden_size: int):
        super().__init__()
        self.config = config
        self.embeddings = VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.merger = VisionMerger(config, text_hidden_size)

    def __call__(self, pixel_values: mx.array, image_grid_thw) -> mx.array:
        """pixel_values [num_image_tokens, merge*patch, merge*patch, C] in [-1, 1]; image_grid_thw
        [num_images, 3] counted in language-model tokens. Returns one embedding per image token."""
        cfg = self.config
        merge, patch, side = cfg.adapter_merge_size, cfg.native_patch_size, cfg.posemb_grid_size
        emb = self.embeddings
        dtype = emb.patch_embedding.weight.dtype
        grid = np.asarray(image_grid_thw, dtype=np.int64).reshape(-1, 3) * np.array([1, merge, merge])
        per_image = grid.prod(-1)
        num_patches = int(per_image.sum())

        # one native patch per row, ordered so the merge x merge patches of a token are consecutive
        pixels = pixel_values.astype(dtype).reshape(-1, merge, patch, merge, patch, pixel_values.shape[-1])
        pixels = pixels.transpose(0, 1, 3, 2, 4, 5).reshape(num_patches, -1)
        patches = emb.patch_embedding(pixels)

        rows, cols, heights, widths = _patch_rows_cols(grid, merge)
        h_taps, h_w = _axis_taps(rows, heights, side)
        w_taps, w_w = _axis_taps(cols, widths, side)
        index = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
        weight = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        position = (emb.position_embedding.weight[mx.array(index)].astype(F32) * mx.array(weight)[..., None]).sum(axis=1)
        patches = patches + position.astype(dtype)

        # each image is a class token followed by its patches
        cls = (emb.class_embedding + emb.cls_position_embedding).astype(dtype)
        starts = np.concatenate([[0], np.cumsum(per_image + 1)])
        bounds = [(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])]
        offsets = np.concatenate([[0], np.cumsum(per_image)])
        pieces = []
        for i in range(len(per_image)):
            pieces += [cls[None], patches[int(offsets[i]) : int(offsets[i + 1])]]
        hidden = self.pre_layernorm(mx.concatenate(pieces, axis=0))

        # axial RoPE: patches are 1-indexed (row, col) and class tokens sit at (0, 0)
        patch_index = np.arange(num_patches) + np.repeat(np.arange(1, len(per_image) + 1), per_image)
        pos = np.zeros((hidden.shape[0], 2), dtype=np.float32)
        pos[patch_index, 0], pos[patch_index, 1] = rows + 1, cols + 1
        spatial = (cfg.hidden_size // cfg.num_attention_heads) // 2
        inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))
        freqs = pos[..., None] * inv_freq  # [N, 2, F]
        angles = np.repeat(np.concatenate([freqs[:, 1], freqs[:, 0]], axis=-1), 2, axis=-1)
        cos, sin = mx.array(np.cos(angles)), mx.array(np.sin(angles))

        for layer in self.layers:
            hidden = layer(hidden, bounds, cos, sin)
        states = hidden[mx.array(patch_index)]
        return self.merger(states.reshape(-1, merge * merge * cfg.hidden_size))
