"""The conversion plan for the released checkpoint must cover the full-size MLX model exactly:
same parameter names, shapes and dtypes. Reads only the safetensors headers from the Hub."""
import json, os, sys
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inkling2_mlx import Inkling2Config, Model
from inkling2_mlx.convert import DTYPE_SIZE, HubSource, build_plan

MX_DTYPE = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "U8": mx.uint8, "U32": mx.uint32}


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "thinkingmachines/Inkling-2-Small"
    source = HubSource(repo)
    ok = True
    for bits in (None, 2, 3, 4):
        shards, formats = build_plan(source, expert_bits=bits)
        config = json.loads(source.read_file("config.json"))
        config["quantization"] = {"expert_formats": formats if bits is None else ["affine"] * len(formats),
                                  "activation_formats": formats, "gate_up_interleaved": True,
                                  "affine_bits": bits, "affine_group_size": 64}
        model = Model(Inkling2Config.from_dict(config))  # lazy: nothing is allocated
        params = dict(tree_flatten(model.parameters()))
        planned = {o.name: o for s in shards for o in s}
        missing, extra = sorted(set(params) - set(planned)), sorted(set(planned) - set(params))
        bad_shape = [(n, planned[n].shape, params[n].shape) for n in planned if n in params and tuple(planned[n].shape) != tuple(params[n].shape)]
        # quantized tensors must load with exactly the planned dtype; float ones may be cast at load
        bad_dtype = [(n, planned[n].dtype, params[n].dtype) for n in planned
                     if n in params and planned[n].dtype in ("U8", "U32") and MX_DTYPE[planned[n].dtype] != params[n].dtype]
        total = sum(o.nbytes for o in planned.values())
        n_params = sum(int(np.prod(o.shape)) * (32 // (bits or (4 if f == "nvfp4" else 8)) if o.dtype == "U32" else 1)
                       for s, f in zip(shards[1:], formats) for o in s if o.name.endswith(("gate_up_proj", "down_proj")))
        label = "native" if bits is None else f"affine {bits}-bit"
        passed = not (missing or extra or bad_shape or bad_dtype)
        ok &= passed
        print(f"[{'ok ' if passed else 'FAIL'}] {label:13s} {len(planned)} tensors in {len(shards)} shards, {total / 1e9:6.1f} GB, "
              f"{n_params / 1e9:.1f}B expert params, largest shard {max(sum(o.nbytes for o in s) for s in shards) / 1e9:.1f} GB")
        for title, items in (("missing from plan", missing), ("not in model", extra), ("shape mismatch", bad_shape), ("dtype mismatch", bad_dtype)):
            if items:
                print(f"       {title}: {len(items)} e.g. {items[:4]}")
        if bits is None:
            print("       expert formats:", {f: formats.count(f) for f in set(formats)},
                  "| source tensors:", len(source.tensors), "| bytes copied verbatim:",
                  f"{sum(o.nbytes for o in planned.values() if not o.name.endswith('_outer_scale')) / 1e9:.1f} GB")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
