"""Text generation for Inkling 2 on MLX.

    python -m inkling2_mlx.generate --model ./Inkling-2-Small-MLX --prompt "Why is the sky blue?"
    python -m inkling2_mlx.generate --model ./Inkling-2-Small-MLX --image cat.png --prompt "What is this?"
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from .model import Model, load_model
from .processing import Processor, parse_response


def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_p < 1.0:
        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        cumulative = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        keep = (cumulative - mx.softmax(sorted_logits, axis=-1)) < top_p  # always keeps the top token
        choice = mx.random.categorical(mx.where(keep, sorted_logits, -mx.inf))
        return mx.take_along_axis(order, choice[..., None], axis=-1)[..., 0]
    return mx.random.categorical(logits)


def generate_tokens(model: Model, inputs: dict, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
                    prefill_step: int = 512, stop_ids=None):
    """Yield generated token ids. `inputs` is the dict returned by `Processor.__call__`."""
    stop_ids = set(stop_ids if stop_ids is not None else [model.config.eos_token_id])
    ids = mx.array(inputs["input_ids"])[None]
    pixels = inputs.get("pixel_values")
    embeds = model.embed(ids, mx.array(pixels) if pixels is not None else None, inputs.get("image_grid_thw"))
    cache = model.make_cache()

    logits = None
    for start in range(0, embeds.shape[1], prefill_step):  # prefill in chunks to bound attention memory
        logits = model(inputs_embeds=embeds[:, start : start + prefill_step], cache=cache)[:, -1]
        mx.eval(logits)
        cache.eval()
    for _ in range(max_tokens):
        token = _sample(logits, temperature, top_p)
        mx.eval(token)
        token_id = int(token.item())
        yield token_id
        if token_id in stop_ids:
            return
        logits = model(token[None], cache=cache)[:, -1]
        mx.eval(logits)
        cache.eval()


def generate(model: Model, processor: Processor, messages, max_tokens: int = 512, temperature: float = 0.0,
             top_p: float = 1.0, reasoning_effort=None, tools=None, stream=None) -> dict:
    """Run a chat completion. Returns the raw text plus parsed reasoning/content/tool calls and timings.

    `stream`, if given, is called with each new piece of raw text (special tokens included).
    """
    inputs = processor(messages, tools=tools, reasoning_effort=reasoning_effort)
    generated, emitted = [], ""
    started = time.perf_counter()
    first = None
    for token in generate_tokens(model, inputs, max_tokens, temperature, top_p):
        first = first or time.perf_counter()
        generated.append(token)
        text = processor.decode(generated)
        if stream and not text.endswith("�"):  # wait for complete UTF-8 sequences
            stream(text[len(emitted) :])
            emitted = text
    ended = time.perf_counter()
    raw = processor.decode(generated)
    return {
        "raw": raw,
        **parse_response(raw),
        "prompt_tokens": len(inputs["input_ids"]),
        "completion_tokens": len(generated),
        "prompt_seconds": (first or ended) - started,
        "tokens_per_second": (len(generated) - 1) / (ended - first) if first and ended > first and len(generated) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="directory of a converted checkpoint")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--image", action="append", default=[], help="image file to attach (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default=None, help="none|minimal|low|medium|high|xhigh|max or 0-0.99")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-activation-quantization", action="store_true",
                    help="feed the experts unquantized activations instead of emulating the trained arithmetic")
    args = ap.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)
    model = load_model(args.model)
    if args.no_activation_quantization:
        for layer in model.model.language_model.layers:
            layer.mlp.experts.act_quant = None
    processor = Processor(args.model)

    content = [{"type": "image", "image": path} for path in args.image] + [{"type": "text", "text": args.prompt}]
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": content}]
    effort = args.reasoning_effort
    if effort is not None:
        try:
            effort = float(effort)
        except ValueError:
            pass
    result = generate(model, processor, messages, args.max_tokens, args.temp, args.top_p, reasoning_effort=effort,
                      stream=lambda piece: (sys.stdout.write(piece), sys.stdout.flush()))
    print(f"\n\n[{result['prompt_tokens']} prompt tokens in {result['prompt_seconds']:.1f}s, "
          f"{result['completion_tokens']} generated at {result['tokens_per_second']:.2f} tok/s, "
          f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB]", file=sys.stderr)


if __name__ == "__main__":
    main()
