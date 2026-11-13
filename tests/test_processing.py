"""Prompt construction and image preprocessing must match the transformers processor; greedy decoding
must match transformers' `generate`. Needs the released tokenizer files (pass their directory)."""
import os, sys, tempfile
import numpy as np
import torch
import mlx.core as mx
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
from inkling2_mlx import load_model
from inkling2_mlx.generate import generate_tokens
from inkling2_mlx.processing import Processor, parse_response

if len(sys.argv) < 2:
    sys.exit("usage: test_processing.py <directory holding the released tokenizer and chat template files>")
TOKENIZER_DIR = sys.argv[1]


def main():
    from transformers import AutoProcessor

    ok = True
    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'ok ' if passed else 'FAIL'}] {name} {detail}")

    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, size=(93, 131, 3), dtype=np.uint8))          # not a multiple of 40
    rgba = Image.fromarray(rng.integers(0, 256, size=(40, 77, 4), dtype=np.uint8), "RGBA")  # transparency
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": [{"type": "image", "image": rgb}, {"type": "text", "text": "Compare with this one:"},
                                     {"type": "image", "image": rgba}, {"type": "text", "text": "Which is noisier? ünïcödé ✓"}]},
        {"role": "assistant", "content": "The first."},
        {"role": "user", "content": "Why?"},
    ]
    ours = Processor(TOKENIZER_DIR)
    theirs = AutoProcessor.from_pretrained(TOKENIZER_DIR)
    for effort in (None, "low", 0.5):
        kw = {} if effort is None else {"reasoning_effort": effort}
        want = theirs.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", **kw)
        got = ours(messages, reasoning_effort=effort)
        report(f"input_ids match (reasoning_effort={effort})", got["input_ids"] == want["input_ids"][0].tolist(),
               f"({len(got['input_ids'])} tokens)")
    report("image_grid_thw match", got["image_grid_thw"] == want["image_grid_thw"].tolist(), str(got["image_grid_thw"]))
    diff = float(np.abs(got["pixel_values"] - want["pixel_values"].numpy()).max())
    report("pixel_values match", diff < 1e-5, f"(max abs diff {diff:.2e}, shape {got['pixel_values'].shape})")

    text_only = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
    want = theirs.tokenizer.apply_chat_template(text_only, tokenize=True, return_dict=True)["input_ids"]
    report("text-only prompt matches", ours(text_only)["input_ids"] == list(want), repr(ours.decode(ours(text_only)["input_ids"]))[:150])

    parsed = parse_response("<|message_model|><|content_thinking|>hmm<|end_message|><|message_model|><|content_text|>Because Rayleigh."
                            "<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|end_message|><|content_model_end_sampling|>")
    report("response parsing", parsed["reasoning_content"] == "hmm" and parsed["content"] == "Because Rayleigh."
           and parsed["tool_calls"] == [{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}])

    # greedy decoding against transformers.generate on the small random model
    tmp = tempfile.mkdtemp(prefix="ink2gen_")
    ref = tiny.build_reference()
    tiny.export_state(ref, tmp)
    model = load_model(tmp)
    prompt = rng.integers(8, 500, size=23).tolist()
    with torch.no_grad():
        want = ref.generate(input_ids=torch.tensor([prompt]), max_new_tokens=40, do_sample=False, eos_token_id=None,
                            pad_token_id=0)[0, len(prompt):].tolist()
    got = list(generate_tokens(model, {"input_ids": prompt}, max_tokens=40, prefill_step=7, stop_ids=[]))
    same = sum(a == b for a, b in zip(got, want))
    report("greedy generation matches transformers.generate", got == want, f"({same}/40 tokens equal)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
