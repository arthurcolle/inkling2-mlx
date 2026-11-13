"""End-to-end converter test on a small random model.

transformers writes the model back out in the original checkpoint layout (its own reverse key mapping),
we block-quantize the experts exactly as the release stores them (NVFP4 for the lower layers, MXFP8 for
the top ones), convert with `inkling2_mlx.convert`, and require the MLX model to reproduce a transformers
model that holds the same (dequantized) weights.
"""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
import mlx.core as mx
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
from inkling2_mlx import load_model
from inkling2_mlx.convert import LocalSource, Out, convert, write_shard
from inkling2_mlx.formats import mxfp8_decode, nvfp4_decode

TOL = 1e-3
NVFP4_LAYERS = range(0, 10)  # the release stores layers 0-39 as NVFP4 and 40-47 as MXFP8
DEN = 2688.0
ST = {np.dtype("float32"): "F32", np.dtype("uint8"): "U8"}


def write_original_quantized(float_dir, out_dir):
    """Rewrite a float original-layout checkpoint with block-quantized experts, as released."""
    outs, dense = [], {}
    f = safe_open(os.path.join(float_dir, "model.safetensors"), "np")

    def add(name, array, dtype=None):
        array = np.ascontiguousarray(array)
        outs.append(Out(name, dtype or ST[array.dtype], array.shape, lambda a=array: iter([a.tobytes()])))

    for name in f.keys():
        w = f.get_tensor(name)
        if ".mlp.experts." not in name:
            add(name, w.astype(np.float32))
            continue
        layer = int(name.split(".")[3])
        if layer in NVFP4_LAYERS:
            amax = np.abs(w).max(axis=(1, 2)).astype(np.float32)
            outer = (amax / np.float32(DEN)).astype(np.float32)
            wq, sc = mx.quantize(mx.array(w / outer[:, None, None]), mode="nvfp4")[:2]
            codes = np.array(wq).view(np.uint8).reshape(*w.shape[:2], -1)
            scales = np.array(sc)
            add(name, codes)
            add(name + "_scale", scales, "F8_E4M3")
            add(name + "_amax", amax)
            add(name + "_outer_scale_denominator", np.array([DEN], dtype=np.float32))
            dense[name] = np.stack([nvfp4_decode(codes[e], scales[e], outer[e]) for e in range(w.shape[0])])
        else:
            wq, sc = mx.quantize(mx.array(w), mode="mxfp8")[:2]
            values = np.array(wq).view(np.uint8).reshape(*w.shape[:2], -1)
            scales = np.array(sc)
            add(name, values, "F8_E4M3")
            add(name + "_scale", scales)
            dense[name] = mxfp8_decode(values, scales)
        assert np.abs(dense[name] - w).max() < 0.2 * np.abs(w).max(), name
    os.makedirs(out_dir, exist_ok=True)
    write_shard(Path(out_dir) / "model-00001-of-00001.safetensors", outs)
    json.dump(tiny.MLX_CONFIG, open(os.path.join(out_dir, "config.json"), "w"))
    return dense


def deinterleave(w13):
    e, rows, cols = w13.shape
    return w13.reshape(e, rows // 2, 2, cols).transpose(0, 2, 1, 3).reshape(e, rows, cols)


def reference_logits(ref, expert_weights, ids):
    """Reference logits with the experts replaced by the given dense tensors in original layout."""
    with torch.no_grad():
        for name, w in expert_weights.items():
            layer = ref.model.language_model.layers[int(name.split(".")[3])]
            if name.endswith("w13_weight"):
                layer.mlp.experts.gate_up_proj.copy_(torch.tensor(deinterleave(w)))
            else:
                layer.mlp.experts.down_proj.copy_(torch.tensor(w))
        return ref(input_ids=torch.tensor(ids), use_cache=False).logits.numpy()


def converted_dense(directory):
    """Dense expert weights implied by a converted MLX checkpoint, keyed by original tensor name."""
    cfg = json.load(open(os.path.join(directory, "config.json")))["quantization"]
    weights = {}
    for file in sorted(Path(directory).glob("*.safetensors")):
        weights.update(mx.load(str(file)))
    out = {}
    for i, fmt in enumerate(cfg["expert_formats"]):
        for target, orig in (("gate_up_proj", "w13"), ("down_proj", "w2")):
            key = f"model.language_model.layers.{i}.mlp.experts.{target}"
            if fmt == "affine":
                # fp32 scales, as quantized_matmul computes (dequantize would otherwise round to fp16)
                w = mx.dequantize(weights[key], weights[key + "_scales"].astype(mx.float32),
                                  weights[key + "_biases"].astype(mx.float32),
                                  group_size=cfg["affine_group_size"], bits=cfg["affine_bits"])
            else:
                w = mx.dequantize(weights[key], weights[key + "_scales"], mode=fmt, dtype=mx.float32)
                if fmt == "nvfp4":
                    w = w * weights[key + "_outer_scale"][:, None, None]
            out[f"model.llm.layers.{i}.mlp.experts.{orig}_weight"] = np.array(w.astype(mx.float32))
    return out


def main():
    work = Path(tempfile.mkdtemp(prefix="ink2conv_"))
    ref = tiny.build_reference()
    ref.save_pretrained(work / "orig_float")
    dense = write_original_quantized(work / "orig_float", work / "orig")
    ids = np.random.default_rng(1).integers(8, 500, size=(2, 40))
    results = []

    def check(name, got, want, tol=TOL):
        d = float(np.abs(got - want).max())
        results.append((name, d, d < tol))

    # native repack: bit-identical weights, so the dequantized reference must match to rounding noise
    source = LocalSource(work / "orig")
    names, formats, _ = convert(source, work / "native", activation_quantization=False, log=lambda *_: None)
    assert formats == ["nvfp4"] * 10 + ["mxfp8"] * 2, formats
    want = reference_logits(ref, dense, ids)
    model = load_model(work / "native")
    check("native repack vs dequantized reference", np.array(model(mx.array(ids))), want)
    # the stored expert bytes must be exactly the source bytes (only the dtype label changes)
    converted = {}
    for file in sorted((work / "native").glob("*.safetensors")):
        converted.update(mx.load(str(file)))
    exact = True
    for name, ref_t in source.tensors.items():
        if ".mlp.experts." not in name or name.endswith(("_amax", "_outer_scale_denominator")):
            continue
        layer = name.split(".")[3]
        target = "gate_up_proj" if ".w13_" in name else "down_proj"
        key = f"model.language_model.layers.{layer}.mlp.experts.{target}" + ("_scales" if name.endswith("_scale") else "")
        exact &= np.array(converted[key]).tobytes() == source.read(ref_t, 0, ref_t.nbytes)
    results.append(("native repack: expert bytes identical to the source checkpoint", 0.0 if exact else 1.0, exact))

    # cached decoding through the quantized experts (single-token steps use the gathered path)
    cache, steps = model.make_cache(), []
    for t in range(40):
        steps.append(np.array(model(mx.array(ids[:, t : t + 1]), cache=cache)))
        cache.eval()
    check("native repack, token-by-token with cache", np.concatenate(steps, axis=1), want)

    # the layer-at-a-time path used to validate checkpoints too large for memory
    from inkling2_mlx.validate import streaming_logits
    stream = streaming_logits(work / "native", ids[0].tolist(), activation_modes=(False, True))
    check("streaming (one layer in memory) vs whole model", stream[False], np.array(model(mx.array(ids[:1])))[0])
    for layer in model.model.language_model.layers:
        layer.mlp.experts.act_quant = layer.mlp.experts.format
    check("streaming vs whole model, activation quantization on", stream[True], np.array(model(mx.array(ids[:1])))[0])

    # Activation quantization is a different, coarser computation (exact arithmetic is covered by
    # test_actquant.py). On a non-chaotic random model it must stay close to the unquantized-activation run.
    smooth = tiny.build_reference(smooth=True)
    smooth.save_pretrained(work / "smooth_float")
    write_original_quantized(work / "smooth_float", work / "smooth")
    outputs = {}
    for flag in (False, True):
        convert(LocalSource(work / "smooth"), work / f"smooth_aq{int(flag)}", activation_quantization=flag, log=lambda *_: None)
        outputs[flag] = np.array(load_model(work / f"smooth_aq{int(flag)}")(mx.array(ids)))
    agree = float((outputs[True].argmax(-1) == outputs[False].argmax(-1)).mean())
    drift = float(np.abs(outputs[True] - outputs[False]).mean())
    results.append((f"activation quantization on vs off (smooth model): argmax agreement {agree:.3f}", drift, agree > 0.9))

    # lossy affine re-quantization: must equal a reference holding exactly the re-quantized weights
    for bits in (8, 4, 2):
        out = work / f"affine{bits}"
        convert(source, out, expert_bits=bits, group_size=32, activation_quantization=False, log=lambda *_: None)
        implied = converted_dense(out)
        err = max(float(np.abs(implied[k] - dense[k]).max() / np.abs(dense[k]).max()) for k in dense)
        check(f"affine {bits}-bit vs reference with the same weights (weight err {err:.3f})",
              np.array(load_model(out)(mx.array(ids))), reference_logits(ref, implied, ids))

    ok = True
    for name, d, passed in results:
        ok &= passed
        print(f"[{'ok ' if passed else 'FAIL'}] {name:78s} {d:.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
