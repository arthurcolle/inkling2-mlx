"""Activation quantization must match a scalar reference of the checkpoint's stated arithmetic:
NVFP4 with a per-row outer scale (amax / 1536) and the four-over-six rule; MXFP8 per MX spec."""
import os, sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inkling2_mlx.moe import mxfp8_fake_quant, nvfp4_fake_quant

f32 = np.float32
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# every non-negative finite E4M3FN value, in code order (0x7f is NaN)
E4M3 = [f32(m * 2.0**-9) for m in range(8)] + [f32((1 + m / 8) * 2.0 ** (e - 7)) for e in range(1, 16) for m in range(8)][:-1]


def nearest_even(table, v):
    """Index of the table entry nearest to v, ties to the even index, saturating at the top."""
    v = min(float(v), float(table[-1]))
    best = min(range(len(table)), key=lambda i: (abs(float(table[i]) - v), i % 2))
    return best


def nvfp4_row(x):
    out = np.zeros_like(x)
    outer = f32(np.abs(x).max()) / f32(1536.0)
    for b in range(0, len(x), 16):
        blk = x[b : b + 16]
        bmax = f32(np.abs(blk).max())
        best = None
        for target in (6.0, 4.0):
            s = f32(0) if outer == 0 else E4M3[nearest_even(E4M3, f32(bmax / f32(target)) / outer)]
            sf = f32(outer * s)
            q = np.array([0.0 if sf == 0 else np.sign(v) * E2M1[nearest_even(E2M1, abs(f32(v / sf)))] for v in blk], dtype=f32)
            deq = (q * sf).astype(f32)
            err = float(np.abs(deq - blk).astype(f32).sum(dtype=f32))
            if best is None or err < best[0]:  # ties keep the six candidate
                best = (err, deq)
        out[b : b + 16] = best[1]
    return out


def mxfp8_row(x):
    out = np.zeros_like(x)
    for b in range(0, len(x), 32):
        blk = x[b : b + 32]
        bmax = float(np.abs(blk).max())
        if bmax == 0:
            continue
        scale = f32(2.0 ** (np.floor(np.log2(bmax)) - 8))
        out[b : b + 32] = [np.sign(v) * E4M3[nearest_even(E4M3, abs(f32(v / scale)))] * scale for v in blk]
    return out


def main():
    rng = np.random.default_rng(0)
    rows = [rng.standard_normal(64) * s for s in (1.0, 1e-3, 37.0, 1e4)]
    rows += [rng.standard_normal(64) * np.exp(rng.standard_normal(64) * 2) for _ in range(20)]   # wide dynamic range
    rows += [np.zeros(64), np.r_[np.zeros(48), rng.standard_normal(16)], np.full(64, 0.3)]
    rows += [np.round(rng.standard_normal(64) * 4) / 4 for _ in range(8)]                         # many exact ties
    rows += [rng.standard_normal(64) * (rng.random(64) < 0.2) for _ in range(8)]                  # sparse
    x = np.stack(rows).astype(f32)

    ok = True
    for name, fake, ref in (("nvfp4 four-over-six", nvfp4_fake_quant, nvfp4_row), ("mxfp8", mxfp8_fake_quant, mxfp8_row)):
        got = np.array(fake(mx.array(x)))
        want = np.stack([ref(r) for r in x])
        mismatch = float((got != want).mean())
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        err = float(np.linalg.norm(want - x) / np.linalg.norm(x))
        passed = mismatch == 0.0
        ok &= passed
        print(f"[{'ok ' if passed else 'FAIL'}] {name:22s} elements differing from scalar reference: {mismatch:.5f} "
              f"(max rel diff {rel:.2e}); quantization error of the format itself {err:.3f}")
    # MLX's own MXFP8 quantizer is a second, independent implementation of the same spec
    wq, sc = mx.quantize(mx.array(x), mode="mxfp8")[:2]
    native = np.array(mx.dequantize(wq, sc, mode="mxfp8", dtype=mx.float32))
    same = float((native == np.array(mxfp8_fake_quant(mx.array(x)))).mean())
    print(f"       mxfp8 agreement with mx.quantize(mode='mxfp8'): {same:.5f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
