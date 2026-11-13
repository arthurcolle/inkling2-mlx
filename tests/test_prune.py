"""Router-guided expert pruning on a small random model: the pruned checkpoint must equal a transformers
reference in which the dropped experts are masked out of the router, the tool's own evaluation must equal
the loaded model, expert prefixes must work, and an interrupted run must resume to the same result."""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
from test_convert import deinterleave, write_original_quantized
from inkling2_mlx import load_model
from inkling2_mlx.convert import LocalSource
from inkling2_mlx.prune import prune

KEEP, PREFIX = 6, 4


def nll_of(model, ids):
    logits = np.array(model(mx.array(ids))).astype(np.float64)[:, :-1]
    logp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
    return float(-np.take_along_axis(logp, ids[:, 1:, None], axis=-1).mean())


def main():
    work = Path(tempfile.mkdtemp(prefix="ink2prune_"))
    ref = tiny.build_reference()
    ref.save_pretrained(work / "float")
    dense = write_original_quantized(work / "float", work / "orig")
    rng = np.random.default_rng(3)
    calib, evals = rng.integers(8, 500, size=(6, 24)), rng.integers(8, 500, size=(2, 24))
    source = LocalSource(work / "orig")
    results = []

    report = prune(source, work / "p", calib, evals, keep=KEEP, eval_keeps=(PREFIX,), linear_bits=None, batch=4, log=lambda *_: None)
    kept = [layer["experts"] for layer in report["layers"]]
    results.append(("every layer keeps the requested number of distinct experts", all(len(set(k)) == KEEP for k in kept) and len(kept) == 12))

    model = load_model(work / "p")
    d = abs(nll_of(model, evals) - report["evaluation"][str(KEEP)]["nll"])
    results.append((f"tool's evaluation equals the loaded pruned model (|dNLL| {d:.1e})", d < 1e-4))
    d = abs(nll_of(load_model(work / "p", max_experts=PREFIX), evals) - report["evaluation"][str(PREFIX)]["nll"])
    results.append((f"expert prefix via max_experts equals the tool's prefix stream (|dNLL| {d:.1e})", d < 1e-4))

    # transformers reference: same dense weights, dropped experts masked out of the router
    with torch.no_grad():
        for name, w in dense.items():
            layer = ref.model.language_model.layers[int(name.split(".")[3])]
            if name.endswith("w13_weight"):
                layer.mlp.experts.gate_up_proj.copy_(torch.tensor(deinterleave(w)))
            else:
                layer.mlp.experts.down_proj.copy_(torch.tensor(w))
        for i, layer in enumerate(ref.model.language_model.layers):
            mask = torch.full((16,), -1e9)
            mask[kept[i]] = 0.0
            layer.mlp.gate.e_score_correction_bias.add_(mask)
        want = ref(input_ids=torch.tensor(evals), use_cache=False).logits.numpy()
    for layer in model.model.language_model.layers:
        layer.mlp.experts.act_quant = None  # the reference feeds the experts unquantized activations
    d = float(np.abs(np.array(model(mx.array(evals))) - want).max())
    results.append((f"pruned model equals the reference with dropped experts masked (max diff {d:.1e})", d < 1e-3))

    # quantized projections + interrupted run resumed
    class Stop(Exception):
        pass
    calls = []
    def interrupting(msg):
        calls.append(msg)
        if msg.startswith("layer  5"):
            raise Stop()
    try:
        prune(source, work / "q", calib, evals, keep=KEEP, eval_keeps=(PREFIX,), linear_bits=8, batch=4, log=interrupting)
    except Stop:
        pass
    resumed = prune(source, work / "q", calib, evals, keep=KEEP, eval_keeps=(PREFIX,), linear_bits=8, batch=4, log=lambda *_: None)
    straight = prune(source, work / "q2", calib, evals, keep=KEEP, eval_keeps=(PREFIX,), linear_bits=8, batch=4, log=lambda *_: None)
    same = [l["experts"] for l in resumed["layers"]] == [l["experts"] for l in straight["layers"]]
    d = abs(resumed["evaluation"][str(KEEP)]["nll"] - straight["evaluation"][str(KEEP)]["nll"])
    results.append((f"interrupted run resumes to the same experts and loss (|dNLL| {d:.1e})", same and d < 1e-5))
    q = load_model(work / "q")
    d = abs(nll_of(q, evals) - resumed["evaluation"][str(KEEP)]["nll"])
    results.append((f"8-bit projections: loaded model equals the tool's evaluation (|dNLL| {d:.1e})", d < 1e-4))
    files = sorted(p.name for p in (work / "q").glob("model-*.safetensors"))
    results.append((f"{len(files)} shards written, config and report present",
                    len(files) == 13 and (work / "q" / "config.json").exists() and (work / "q" / "pruning.json").exists()))
    print("NLL float-projection %.4f | 8-bit projection %.4f | prefix-%d %.4f | uniform %.2f" % (
        report["evaluation"][str(KEEP)]["nll"], resumed["evaluation"][str(KEEP)]["nll"], PREFIX,
        report["evaluation"][str(PREFIX)]["nll"], np.log(500)))
    ok = True
    for name, passed in results:
        ok &= bool(passed)
        print(f"[{'ok ' if passed else 'FAIL'}] {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
