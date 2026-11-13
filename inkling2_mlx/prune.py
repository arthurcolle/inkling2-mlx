"""Shrink an Inkling 2 checkpoint by keeping only the experts its router uses most.

Almost all of the model's parameters sit in the routed experts (384 per layer, 8 used per token), so the
only way to make it much smaller is to drop experts. This tool decides which ones to keep one layer at a
time, without ever holding or downloading the full model:

1. Calibration text is pushed through the layers pruned so far, up to the input of the next router.
2. That layer's router (a few MB) scores all of its experts for every calibration token; the experts
   are ranked by the routing weight they receive.
3. Only the top `keep` experts are fetched, in their original NVFP4/MXFP8 bytes, and the layer is
   finished with them so the next layer sees what the pruned model will actually produce.

Experts are stored most-important first, so `load_model(path, max_experts=M)` gives a still smaller model
from the same files. Each finished layer is written as its own shard and can be uploaded and deleted
immediately, so the output never has to fit on local disk either.

    python -m inkling2_mlx.prune --source thinkingmachines/Inkling-2-Small --out ./pruned --keep 32 \\
        --calibration calibration.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import Inkling2Config
from .convert import _EXPERT, SIDE_FILES, HubSource, LocalSource, Source, TensorRef, expert_format, rename
from .layers import F32, softcap
from .model import _FP32_SUFFIXES, Model
from .text import DecoderLayer

PIECE = 4 << 20
LAYER_PREFIX = "model.language_model.layers."
QUANTIZED_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "r_proj", "o_proj", "gate_proj")


class Fetcher:
    """Reads byte ranges from a source over many connections at once."""

    def __init__(self, source: Source, streams: int = 16):
        self.source = source
        self.pool = futures.ThreadPoolExecutor(streams)
        self.bytes = 0

    def many(self, requests: list) -> list:
        """requests: [(TensorRef, start, length)] -> [bytes], fetched in parallel pieces."""
        jobs, spans = [], []
        for ref, start, length in requests:
            first = len(jobs)
            for s in range(start, start + length, PIECE):
                jobs.append(self.pool.submit(self.source.read, ref, s, min(PIECE, start + length - s)))
            spans.append((first, len(jobs)))
        parts = [j.result() for j in jobs]
        self.bytes += sum(len(p) for p in parts)
        return [b"".join(parts[a:b]) for a, b in spans]

    def arrays(self, refs: dict) -> dict:
        """{name: TensorRef} -> {name: mx.array}, whole tensors."""
        names = list(refs)
        blobs = self.many([(refs[n], 0, refs[n].nbytes) for n in names])
        return {n: _to_array(refs[n], blob) for n, blob in zip(names, blobs)}


def _to_array(ref: TensorRef, blob: bytes) -> mx.array:
    if ref.dtype == "BF16":
        return mx.array(np.frombuffer(blob, dtype="<u2").reshape(ref.shape)).view(mx.bfloat16)
    if ref.dtype == "F32":
        return mx.array(np.frombuffer(blob, dtype="<f4").reshape(ref.shape))
    raise ValueError(f"unexpected dtype {ref.dtype} for a non-expert tensor")


def _quantize(weight: mx.array, bits: int, group_size: int) -> dict:
    wq, scales, biases = mx.quantize(weight.astype(F32), group_size=group_size, bits=bits)
    return {"weight": wq, "scales": scales.astype(mx.float16), "biases": biases.astype(mx.float16)}


def _runtime(weights: dict) -> dict:
    return {k: (v.astype(F32) if k.endswith(_FP32_SUFFIXES) and v.dtype != F32 else v) for k, v in weights.items()}


def _batched(fn, arrays: list, batch: int):
    """Apply `fn` to slices of the leading (sequence) axis and concatenate each output."""
    n = arrays[0].shape[0]
    outs = []
    for start in range(0, n, batch):
        out = fn(*[a[start : start + batch] for a in arrays])
        out = out if isinstance(out, tuple) else (out,)
        mx.eval(*out)
        outs.append(out)
    merged = tuple(mx.concatenate([o[i] for o in outs], axis=0) for i in range(len(outs[0])))
    return merged if len(merged) > 1 else merged[0]


class Stream:
    """One set of sequences flowing through the layers with the first `keep` experts of each layer."""

    def __init__(self, hidden: mx.array, keep: int):
        self.hidden, self.blocks, self.keep = hidden, [], keep


def expert_requests(source: Source, layer: int, fmt: str, experts: list) -> tuple:
    """Byte ranges holding the given experts of one layer, and how to turn them into MLX tensors."""
    requests, layout = [], []
    base = f"model.llm.layers.{layer}.mlp.experts."
    for which, target in (("w13", "gate_up_proj"), ("w2", "down_proj")):
        for suffix, out_name, dtype in (("_weight", target, "<u4"), ("_weight_scale", target + "_scales", np.uint8)):
            ref = source.tensors[base + which + suffix]
            per = ref.nbytes // ref.shape[0]
            rows = ref.shape[1]
            for e in experts:
                requests.append((ref, e * per, per))
            layout.append((out_name, dtype, rows, len(experts)))
    return requests, layout


def fetch_experts(fetcher: Fetcher, layer: int, fmt: str, experts: list) -> dict:
    source = fetcher.source
    requests, layout = expert_requests(source, layer, fmt, experts)
    blobs = fetcher.many(requests)
    out, cursor = {}, 0
    for name, dtype, rows, count in layout:
        stacked = np.stack([np.frombuffer(blobs[cursor + i], dtype=dtype).reshape(rows, -1) for i in range(count)])
        out[name] = mx.array(stacked)
        cursor += count
    if fmt == "nvfp4":
        base = f"model.llm.layers.{layer}.mlp.experts."
        for which, target in (("w13", "gate_up_proj"), ("w2", "down_proj")):
            amax_ref = source.tensors[base + which + "_weight_amax"]
            den_ref = source.tensors[base + which + "_weight_outer_scale_denominator"]
            amax, den = fetcher.many([(amax_ref, 0, amax_ref.nbytes), (den_ref, 0, den_ref.nbytes)])
            outer = np.frombuffer(amax, dtype="<f4") / np.frombuffer(den, dtype="<f4")[0]
            out[target + "_outer_scale"] = mx.array(outer[np.array(experts)].astype(np.float32))
    return out


def prune(source: Source, out_dir, calibration: np.ndarray, evaluation: np.ndarray | None = None, keep: int = 32,
          eval_keeps: tuple = (), linear_bits: int | None = 6, group_size: int = 64, streams: int = 16,
          batch: int = 8, on_shard=None, package_dir=None, log=print) -> dict:
    """Build the pruned checkpoint in `out_dir`. Returns the report that is also written to pruning.json.

    calibration [C, T] and evaluation [E, T] are token ids; all sequences share one length. Evaluation
    sequences never influence which experts are kept.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(source, streams)
    config_json = json.loads(source.read_file("config.json"))
    num_layers = config_json["text_config"]["num_hidden_layers"]
    formats = [expert_format(source, i) for i in range(num_layers)]
    config_json["quantization"] = {
        "expert_formats": formats,
        "activation_formats": formats,
        "activation_quantization": True,
        "gate_up_interleaved": True,
        "experts_per_layer": [keep] * num_layers,
        **({"linear_bits": linear_bits, "linear_group_size": group_size} if linear_bits else {}),
    }
    config_json.setdefault("image_token_id", 200054)
    config = Inkling2Config.from_dict(config_json)
    text = config.text_config
    targets = {rename(n): n for n in source.tensors if not _EXPERT.match(n)}
    shard_name = lambda i: f"model-{i + 1:05d}-of-{num_layers + 1:05d}.safetensors"  # noqa: E731
    state_path, state_arrays = out_dir / "prune_state.json", out_dir / "prune_state.safetensors"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"next_layer": 0, "layers": [], "base_done": False}
    started = time.time()

    def emit(path: Path, weights: dict):
        mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})
        if on_shard:
            on_shard(path)

    # ---- everything outside the decoder layers: embeddings, LM head, vision tower, final norm
    base_refs = {t: source.tensors[n] for t, n in targets.items() if not t.startswith(LAYER_PREFIX)}
    base = fetcher.arrays(base_refs)
    if linear_bits:
        for name in ("model.language_model.embed_tokens", "lm_head"):
            for key, value in _quantize(base.pop(name + ".weight"), linear_bits, group_size).items():
                base[f"{name}.{key}"] = value
        mx.eval(base)
    model = Model(config)
    model.model.language_model.layers = []
    model.prepare_quantized(base)
    model.load_weights(list(_runtime(base).items()), strict=True)
    model.eval()
    if not state["base_done"]:
        emit(out_dir / shard_name(0), base)
        state["base_done"] = True
        state_path.write_text(json.dumps(state))
    log(f"base weights ready ({fetcher.bytes / 1e9:.2f} GB fetched, {time.time() - started:.0f}s)")
    del base

    # ---- token streams
    n_calib = calibration.shape[0]
    all_ids = calibration if evaluation is None else np.concatenate([calibration, evaluation])
    if state["next_layer"] > 0:
        saved = mx.load(str(state_arrays))
        streams_ = {}
        for name, k in state["streams"].items():
            s = Stream(saved[f"{name}.hidden"], k)
            s.blocks = [saved[f"{name}.block{j}"] for j in range(state["blocks"])]
            streams_[name] = s
    else:
        embed = lambda ids: _batched(model.model.language_model.embed_tokens, [mx.array(ids)], batch)  # noqa: E731
        streams_ = {"main": Stream(embed(all_ids), keep)}
        if evaluation is not None:
            for k in eval_keeps:
                streams_[f"keep{k}"] = Stream(embed(evaluation), k)

    # ---- one layer at a time
    for i in range(state["next_layer"], num_layers):
        t0, b0 = time.time(), fetcher.bytes
        prefix = f"{LAYER_PREFIX}{i}."
        weights = fetcher.arrays({t[len(prefix) :]: source.tensors[n] for t, n in targets.items() if t.startswith(prefix)})
        router_w, router_b = weights.pop("mlp.gate.weight"), weights.pop("mlp.gate.e_score_correction_bias")
        if linear_bits:
            for proj in QUANTIZED_PROJECTIONS:
                for key, value in _quantize(weights.pop(f"self_attn.{proj}.weight"), linear_bits, group_size).items():
                    weights[f"self_attn.{proj}.{key}"] = value
        layer = DecoderLayer(text, i, fmt=formats[i], act_format=formats[i], interleaved=True, num_experts=keep)
        if linear_bits:
            nn.quantize(layer, group_size=group_size, bits=linear_bits,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in weights)
        layer.load_weights(list(_runtime(weights).items()), strict=False)
        layer.eval()

        # attention half for every stream; the router then ranks all experts on the calibration tokens
        halves = {name: _batched(lambda h, *bl: layer.attend(h, list(bl), None, 0), [s.hidden, *s.blocks], batch)
                  for name, s in streams_.items()}
        mlp_in = halves["main"][1][:n_calib].reshape(-1, text.hidden_size).astype(F32)
        scores = mx.sigmoid(mlp_in @ router_w.astype(F32).T)
        top = mx.argpartition(-(scores + router_b.astype(F32)), kth=text.num_experts_per_tok - 1, axis=-1)[:, : text.num_experts_per_tok]
        share = mx.take_along_axis(scores, top, axis=-1)
        share = share / (mx.sum(share, axis=-1, keepdims=True) + 1e-20)
        mass = np.bincount(np.array(top).reshape(-1), weights=np.array(share).reshape(-1).astype(np.float64),
                           minlength=router_w.shape[0])
        order = np.argsort(-mass, kind="stable")[:keep]
        total = mass.sum()

        experts = fetch_experts(fetcher, i, formats[i], [int(e) for e in order])
        index = mx.array(order)
        gate = {"mlp.gate.weight": router_w[index], "mlp.gate.e_score_correction_bias": router_b[index]}
        final = {**weights, **gate, **{f"mlp.experts.{k}": v for k, v in experts.items()}}
        mx.eval(final)
        layer.load_weights(list(_runtime({k: v for k, v in final.items() if k.startswith("mlp.")}).items()), strict=False)

        for name, s in streams_.items():
            residual, x = halves[name]
            if s.keep < keep:  # a prefix of the kept experts: restrict the router, expert ids stay valid
                layer.mlp.gate.weight = _runtime(gate)["mlp.gate.weight"][: s.keep]
                layer.mlp.gate.e_score_correction_bias = gate["mlp.gate.e_score_correction_bias"][: s.keep]
                layer.mlp.gate.top_k = min(text.num_experts_per_tok, s.keep)
            else:
                layer.mlp.gate.weight = _runtime(gate)["mlp.gate.weight"]
                layer.mlp.gate.e_score_correction_bias = gate["mlp.gate.e_score_correction_bias"]
                layer.mlp.gate.top_k = min(text.num_experts_per_tok, keep)
            new = _batched(lambda r, v: layer.finish(r, layer.mlp(v)), [residual, x], batch)
            if layer.is_block_boundary:
                s.blocks = s.blocks + [s.hidden]
            s.hidden = new
        del halves

        emit(out_dir / shard_name(i + 1), {prefix + k: v for k, v in final.items()})
        captured = {str(k): float(mass[np.argsort(-mass)[:k]].sum() / total) for k in sorted({keep, *eval_keeps})}
        state["layers"].append({"layer": i, "format": formats[i], "experts": [int(e) for e in order],
                                "routing_mass_kept": captured, "experts_ever_routed": int((mass > 0).sum())})
        state.update(next_layer=i + 1, streams={n: s.keep for n, s in streams_.items()}, blocks=len(streams_["main"].blocks))
        arrays = {}
        for name, s in streams_.items():
            arrays[f"{name}.hidden"] = s.hidden
            arrays.update({f"{name}.block{j}": b for j, b in enumerate(s.blocks)})
        mx.save_safetensors(str(state_arrays) + ".tmp.safetensors", arrays)
        os.replace(str(state_arrays) + ".tmp.safetensors", state_arrays)
        state_path.write_text(json.dumps(state))
        del layer, weights, experts, final
        mx.clear_cache()
        done = i + 1 - 0
        log(f"layer {i:2d} [{formats[i]}] kept {keep}/{router_w.shape[0]} experts holding {captured[str(keep)]:.1%} of routing weight | "
            f"{(fetcher.bytes - b0) / 1e6:6.0f} MB in {time.time() - t0:5.0f}s | total {fetcher.bytes / 1e9:5.1f} GB, "
            f"{(time.time() - started) / 60:5.1f} min, eta {(time.time() - started) / max(done - state.get('resumed_at', 0), 1) * (num_layers - done) / 60:5.0f} min")

    # ---- evaluation: next-token loss of each stream on the held-out sequences
    report = {"keep": keep, "linear_bits": linear_bits, "calibration_tokens": int(calibration.size), "layers": state["layers"],
              "source": getattr(source, "repo", str(getattr(source, "dir", ""))), "evaluation": {}}
    lm = model.model.language_model
    if evaluation is not None:
        targets_ = evaluation[:, 1:]
        for name, s in streams_.items():
            hidden = s.hidden if name != "main" else s.hidden[n_calib:]
            blocks = s.blocks if name != "main" else [b[n_calib:] for b in s.blocks]

            def head(h, *bl):
                final = lm.norm(lm.final_input(list(bl), h)[0]).astype(h.dtype)
                logits = softcap(model.lm_head(final).astype(F32), text.final_logit_softcapping)[..., : text.unpadded_vocab_size]
                return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            logp = np.array(_batched(head, [hidden, *blocks], 2))[:, :-1]
            nll = -np.take_along_axis(logp, targets_[..., None], axis=-1)[..., 0]
            report["evaluation"][str(s.keep)] = {
                "nll": float(nll.mean()), "perplexity": float(np.exp(nll.mean())),
                "top1_accuracy": float((logp.argmax(-1) == targets_).mean()),
                "per_sequence_nll": [float(x) for x in nll.mean(axis=1)],
                "per_token_nll": [[round(float(v), 3) for v in row] for row in nll],
                "top1": [[int(v) for v in row] for row in logp.argmax(-1)],
            }

    # ---- metadata
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    (out_dir / "pruning.json").write_text(json.dumps(report, indent=1))
    for side in SIDE_FILES:
        data = source.read_file(side)
        if data is not None:
            (out_dir / side).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / side).write_bytes(data)
    if package_dir:
        shutil.copytree(package_dir, out_dir / "inkling2_mlx", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="local directory or Hub repo id of the original checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", required=True, help='JSON file: {"calibration": [[ids]], "evaluation": [[ids]]}')
    ap.add_argument("--keep", type=int, default=32, help="experts kept per layer")
    ap.add_argument("--eval-keeps", type=int, nargs="*", default=[8, 16, 24], help="also evaluate these prefixes")
    ap.add_argument("--linear-bits", type=int, default=6, choices=[0, 4, 5, 6, 8], help="0 keeps bf16 projections")
    ap.add_argument("--streams", type=int, default=16, help="parallel connections to the Hub")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--upload-repo", default=None, help="upload each shard to this Hub repo as it is finished")
    ap.add_argument("--remove-uploaded", action="store_true", help="delete each shard locally once uploaded")
    args = ap.parse_args()

    data = json.loads(Path(args.calibration).read_text())
    source = LocalSource(args.source) if os.path.isdir(args.source) else HubSource(args.source)
    on_shard, uploads = None, []
    if args.upload_repo:
        from huggingface_hub import HfApi

        api, pool = HfApi(), futures.ThreadPoolExecutor(1)

        def push(path: Path):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.upload_repo,
                            commit_message=f"Add {path.name}")
            if args.remove_uploaded:
                path.unlink()
            print(f"uploaded {path.name}", flush=True)

        def on_shard(path: Path):
            uploads.append(pool.submit(push, path))

    report = prune(source, args.out, np.array(data["calibration"]), np.array(data["evaluation"]) if data.get("evaluation") else None,
                   keep=args.keep, eval_keeps=tuple(k for k in args.eval_keeps if k < args.keep),
                   linear_bits=args.linear_bits or None, streams=args.streams, batch=args.batch, on_shard=on_shard,
                   package_dir=Path(__file__).parent, log=lambda m: print(m, flush=True))
    for u in uploads:
        u.result()
    for k, e in sorted(report["evaluation"].items(), key=lambda kv: int(kv[0])):
        print(f"{int(k):3d} experts/layer: held-out NLL {e['nll']:.3f} (perplexity {e['perplexity']:.1f}), top-1 accuracy {e['top1_accuracy']:.3f}")


if __name__ == "__main__":
    main()
