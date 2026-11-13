"""Prompt construction: chat template, tokenization and image preprocessing for Inkling 2."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

IMAGE_PART_TYPES = ("image", "input_image", "image_url")


def composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    """Flatten an RGBA uint8 image onto the 8px grey checkerboard the model was trained with."""
    height, width = rgba.shape[:2]
    rows = np.arange(height)[:, None] // 8
    cols = np.arange(width)[None, :] // 8
    background = np.where((rows + cols) % 2 == 0, 204, 153)[..., None]
    rgba = rgba.astype(np.int32)
    alpha = rgba[..., 3:4]
    return ((rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def preprocess_image(image, patch: int = 40):
    """Image -> (pixel patches [rows * cols, patch, patch, 3] in [-1, 1], [1, rows, cols]).

    One `patch` x `patch` pixel tile becomes one language-model token. The image is not resized; it is
    padded on the bottom/right to a whole number of tiles.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.has_transparency_data:
        pixels = composite_on_checkerboard(np.asarray(image.convert("RGBA")))
    else:
        pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    rows, cols = math.ceil(height / patch), math.ceil(width / patch)
    padded = np.full((rows * patch, cols * patch, 3), -1.0, dtype=np.float32)  # -1 on the 0-255 scale
    padded[:height, :width] = pixels
    normalised = (padded / 255.0 - 0.5) / 0.5
    tiles = normalised.reshape(rows, patch, cols, patch, 3).transpose(0, 2, 1, 3, 4)
    return tiles.reshape(rows * cols, patch, patch, 3).astype(np.float32), [1, rows, cols]


class Processor:
    def __init__(self, path):
        from transformers import AutoTokenizer

        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        template = path / "chat_template.jinja"
        if template.exists():
            self.tokenizer.chat_template = template.read_text()
        cfg = json.loads((path / "processor_config.json").read_text()) if (path / "processor_config.json").exists() else {}
        self.image_token = cfg.get("image_token", "<|unused_200054|>")
        self.patch = cfg.get("image_processor", {}).get("size", {}).get("height", 40)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    @staticmethod
    def _images(messages):
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES:
                        yield part.get("image") or part.get("path") or part.get("url") or part.get("image_url")

    def __call__(self, messages, tools=None, reasoning_effort=None):
        """Chat messages -> {"input_ids": [T], "pixel_values": [N, p, p, 3] | None, "image_grid_thw": [[1, r, c], ...]}.

        Image parts are {"type": "image", "image": <PIL image or path>}. `reasoning_effort` is one of
        none/minimal/low/medium/high/xhigh/max or a number in [0, 0.99]; the template defaults to 0.9.
        """
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

        patches, grids = [], []
        for image in self._images(messages):
            tiles, grid = preprocess_image(image, self.patch)
            patches.append(tiles)
            grids.append(grid)
        if text.count(self.image_token) != len(grids):
            raise ValueError(f"{text.count(self.image_token)} image slots in the prompt but {len(grids)} images")
        counts = iter(g[1] * g[2] for g in grids)
        text = re.sub(re.escape(self.image_token), lambda _: self.image_token * next(counts), text)

        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": input_ids,
            "pixel_values": np.concatenate(patches) if patches else None,
            "image_grid_thw": grids or None,
        }

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


_SEGMENT = re.compile(r"(?:<\|message_model\|>[^<]*)?<\|content_(thinking|text|tool_error|invoke_tool_text|invoke_tool_json)\|>(.*?)(?:<\|end_message\|>|$)", re.S)


def parse_response(raw: str) -> dict:
    """Split raw generated text (special tokens kept) into reasoning, answer text and tool calls."""
    out = {"reasoning_content": "", "content": "", "tool_calls": []}
    for kind, body in _SEGMENT.findall(raw.split("<|content_model_end_sampling|>")[0]):
        if kind == "thinking":
            out["reasoning_content"] += body
        elif kind == "invoke_tool_json":
            try:
                call = json.loads(body)
                out["tool_calls"].append({"type": "function", "function": {"name": call.get("name"), "arguments": call.get("args")}})
            except json.JSONDecodeError:
                out["content"] += body
        else:
            out["content"] += body
    return out
