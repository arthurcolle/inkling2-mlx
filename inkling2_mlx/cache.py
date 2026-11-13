"""Incremental decoding state."""

from __future__ import annotations

import mlx.core as mx


class LayerCache:
    """Keys/values of one layer (already clipped/normalised) plus the inputs its four short
    convolutions still need: key conv, value conv, post-attention conv, post-MLP conv."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.kv_offset = 0  # absolute position of the first cached key
        self.conv = [None, None, None, None]

    @property
    def state(self):
        return [a for a in (self.keys, self.values, *self.conv) if a is not None]


class Cache:
    def __init__(self, num_layers: int):
        self.layers = [LayerCache() for _ in range(num_layers)]
        self.offset = 0  # tokens processed so far

    def eval(self):
        mx.eval([layer.state for layer in self.layers])
