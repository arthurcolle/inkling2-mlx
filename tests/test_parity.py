"""The MLX port must reproduce the transformers reference on a small random model."""
import os, sys, tempfile
import numpy as np
import torch
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny import IMAGE_TOKEN, build_reference, export_state
from inkling2_mlx import load_model
import inkling2_mlx.moe as moe

# the reference itself moves by up to 3e-4 when the same input is batched differently (fp32 rounding)
TOL = 1e-3


def _setup(tmp):
    ref = build_reference()
    export_state(ref, tmp)
    return ref, load_model(tmp)


def _diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main():
    tmp = tempfile.mkdtemp(prefix="ink2tiny_")
    ref, model = _setup(tmp)
    rng = np.random.default_rng(0)
    ids = rng.integers(8, 500, size=(2, 40))
    results = {}

    # 1. full-sequence logits, longer than the sliding window, rel_extent and log-scaling floor
    with torch.no_grad():
        want = ref(input_ids=torch.tensor(ids), use_cache=False).logits.numpy()
    for limit, name in ((0, "grouped"), (10**9, "gathered")):
        moe.GATHER_LIMIT = limit
        got = np.array(model(mx.array(ids)))
        results[f"text full sequence ({name} experts)"] = _diff(got, want)
    moe.GATHER_LIMIT = 64
    assert want.shape == (2, 40, 500), want.shape

    # 2. incremental decoding with a cache: prefill in uneven chunks, then token by token
    cache = model.make_cache()
    pieces, pos = [], 0
    for step in (5, 1, 11, 3, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1):
        pieces.append(np.array(model(mx.array(ids[:, pos : pos + step]), cache=cache)))
        cache.eval()
        pos += step
    assert pos == 40
    results["text cached (chunked prefill + stepwise)"] = _diff(np.concatenate(pieces, axis=1), want)

    # 3. the reference's own cached path, one token at a time
    with torch.no_grad():
        out = ref(input_ids=torch.tensor(ids[:, :17]), use_cache=True)
        ref_steps, past = [out.logits.numpy()], out.past_key_values
        for t in range(17, 40):
            out = ref(input_ids=torch.tensor(ids[:, t : t + 1]), past_key_values=past, use_cache=True)
            ref_steps.append(out.logits.numpy())
            past = out.past_key_values
    results["reference cached vs reference full (sanity)"] = _diff(np.concatenate(ref_steps, axis=1), want)

    # 4. images: two images of different sizes in one prompt
    grid = np.array([[1, 2, 3], [1, 1, 2]])
    n_img = int(grid.prod(-1).sum())
    pixels = rng.uniform(-1, 1, size=(n_img, 8, 8, 3)).astype(np.float32)
    img_ids = rng.integers(8, 500, size=(1, 30))
    img_ids[0, 3:9] = IMAGE_TOKEN
    img_ids[0, 15:17] = IMAGE_TOKEN
    with torch.no_grad():
        want_feat = torch.cat(ref.get_image_features(torch.tensor(pixels), torch.tensor(grid), return_dict=True).pooler_output).numpy()
        want_img = ref(input_ids=torch.tensor(img_ids), pixel_values=torch.tensor(pixels),
                       image_grid_thw=torch.tensor(grid), use_cache=False).logits.numpy()
    results["vision tower features"] = _diff(np.array(model.get_image_features(mx.array(pixels), grid)), want_feat)
    results["image + text logits"] = _diff(np.array(model(mx.array(img_ids), pixel_values=mx.array(pixels), image_grid_thw=grid)), want_img)

    ok = True
    for name, d in results.items():
        flag = "ok " if d < TOL else "FAIL"
        ok &= d < TOL
        print(f"[{flag}] {name:48s} max abs diff {d:.3e}")
    print("logit scale: max |logit| %.3f, std %.3f" % (np.abs(want).max(), want.std()))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
