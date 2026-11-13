"""Check a converted checkpoint on real text without holding it in memory.

The prompt is pushed through the decoder one layer at a time; each layer's weights are loaded, used and
released, so peak memory is one layer (about 13 GB) however large the model is. The assistant turn is
teacher-forced: the report shows whether the model's own top prediction at each position is the token
that was actually there, and the average negative log-likelihood of the text.

    python -m inkling2_mlx.validate --model ./Inkling-2-Small-MLX
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import Inkling2Config
from .layers import F32, softcap
from .model import _FP32_SUFFIXES, Model
from .processing import Processor

DEFAULT_MESSAGES = [
    {
        "role": "user",
        "content": "List the capital cities of France, Germany, Italy, Spain and Japan, in that order, "
        "separated by commas. Reply with the five names only.",
    },
    {"role": "assistant", "content": "Paris, Berlin, Rome, Madrid, Tokyo"},
]


def _as_runtime_dtype(weights: dict) -> dict:
    return {k: (v.astype(F32) if k.endswith(_FP32_SUFFIXES) and v.dtype != F32 else v) for k, v in weights.items()}


def streaming_logits(path, input_ids, pixel_values=None, image_grid_thw=None, activation_modes=(True,), log=None):
    """Logits [T, vocab] for one sequence, per activation-quantization mode, loading one layer at a time."""
    path = Path(path)
    config = Inkling2Config.from_pretrained(path)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    layer_prefix = "model.language_model.layers."
    model = Model(config)
    layers = model.model.language_model.layers
    model.model.language_model.layers = []  # the layers are filled in one at a time below

    base_files = sorted({f for k, f in weight_map.items() if not k.startswith(layer_prefix)})
    base = {}
    for file in base_files:
        base.update({k: v for k, v in mx.load(str(path / file)).items() if not k.startswith(layer_prefix)})
    model.prepare_quantized(base)
    model.load_weights(list(_as_runtime_dtype(base).items()), strict=True)
    model.eval()

    ids = mx.array(input_ids)[None]
    embeds = model.embed(ids, mx.array(pixel_values) if pixel_values is not None else None, image_grid_thw)
    hidden = {mode: embeds for mode in activation_modes}
    blocks = {mode: [] for mode in activation_modes}
    for i, layer in enumerate(layers):
        started = time.perf_counter()
        prefix = f"{layer_prefix}{i}."
        weights = {}
        for file in sorted({f for k, f in weight_map.items() if k.startswith(prefix)}):
            weights.update({k[len(prefix) :]: v for k, v in mx.load(str(path / file)).items() if k.startswith(prefix)})
        if config.linear_bits:
            nn.quantize(layer, group_size=config.linear_group_size, bits=config.linear_bits,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in weights)
        layer.load_weights(list(_as_runtime_dtype(weights).items()), strict=True)
        layer.eval()
        for mode in activation_modes:
            layer.mlp.experts.act_quant = config.activation_formats[i] if (mode and config.activation_formats) else None
            new = layer(hidden[mode], blocks[mode], None, 0)
            if layer.is_block_boundary:
                blocks[mode] = blocks[mode] + [hidden[mode]]
            hidden[mode] = new
        mx.eval(list(hidden.values()))
        layers[i] = None
        del layer, weights
        mx.clear_cache()
        if log:
            log(f"layer {i:2d} done in {time.perf_counter() - started:6.1f}s (peak memory {mx.get_peak_memory() / 1e9:.1f} GB)")

    text = model.model.language_model
    out = {}
    for mode in activation_modes:
        final = text.norm(text.final_input(blocks[mode], hidden[mode])[0]).astype(embeds.dtype)
        logits = softcap(model.lm_head(final).astype(F32), config.text_config.final_logit_softcapping)
        out[mode] = np.array(logits[0, :, : config.text_config.unpadded_vocab_size])
    return out


def score(path, messages=None, reasoning_effort="none", activation_modes=(True, False), log=print) -> dict:
    """Teacher-forced report for `messages`, whose last message is the assistant reply to check."""
    messages = messages or DEFAULT_MESSAGES
    processor = Processor(path)
    inputs = processor(messages, reasoning_effort=reasoning_effort)
    ids = inputs["input_ids"]
    prompt_len = len(processor(messages[:-1], reasoning_effort=reasoning_effort)["input_ids"])
    logits = streaming_logits(path, ids, inputs["pixel_values"], inputs["image_grid_thw"], activation_modes, log)

    report = {"tokens": len(ids), "prompt_tokens": prompt_len, "modes": {}}
    targets = np.array(ids[1:])
    for mode, lg in logits.items():
        lg = lg[:-1].astype(np.float64)
        logp = lg - (np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) + lg.max(-1, keepdims=True))
        nll = -logp[np.arange(len(targets)), targets]
        top = lg.argmax(-1)
        reply = slice(prompt_len - 1, len(targets))  # predictions of the assistant turn
        rows = [
            {"position": int(t + 1), "target": processor.decode([int(targets[t])]), "predicted": processor.decode([int(top[t])]),
             "p_target": float(np.exp(-nll[t])), "correct": bool(top[t] == targets[t])}
            for t in range(reply.start, reply.stop)
        ]
        report["modes"]["activation_quantization" if mode else "float_activations"] = {
            "reply_top1_accuracy": float(np.mean([r["correct"] for r in rows])),
            "reply_nll": float(nll[reply].mean()),
            "prompt_nll": float(nll[: reply.start].mean()),
            "uniform_nll": float(np.log(lg.shape[-1])),
            "reply": rows,
        }
    if len(logits) == 2:
        a, b = (logits[m][:-1] for m in logits)
        report["mode_agreement_top1"] = float((a.argmax(-1) == b.argmax(-1)).mean())
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()
    report = score(args.model)
    for name, mode in report["modes"].items():
        print(f"\n{name}: reply top-1 accuracy {mode['reply_top1_accuracy']:.2f}, reply NLL {mode['reply_nll']:.3f}, "
              f"prompt NLL {mode['prompt_nll']:.3f} (uniform guessing would be {mode['uniform_nll']:.2f})")
        for row in mode["reply"]:
            print(f"  {row['target']!r:28} predicted {row['predicted']!r:28} p(target)={row['p_target']:.3f} {'ok' if row['correct'] else 'MISS'}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
