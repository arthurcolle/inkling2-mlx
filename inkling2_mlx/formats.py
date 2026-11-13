"""Reference (numpy) decoders for the two block-scaled formats used by Inkling 2 checkpoints."""
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_SIGNED = np.concatenate([E2M1, -E2M1]).astype(np.float32)  # code bit 3 = sign


def e4m3_to_f32(b):
    b = np.asarray(b, dtype=np.uint8)
    e = ((b >> 3) & 0xF).astype(np.int32)
    m = (b & 7).astype(np.float32)
    v = np.where(e == 0, m * np.float32(2.0 ** -9), (1.0 + m / 8.0) * np.exp2(e - 7).astype(np.float32))
    return np.where(b & 0x80, -v, v).astype(np.float32)


def nvfp4_decode(codes_u8, scales_e4m3, outer=1.0):
    """codes [..., K/2] bytes (element 2j low nibble, 2j+1 high nibble); scales [..., K/16] E4M3 bytes."""
    codes_u8 = np.asarray(codes_u8, dtype=np.uint8)
    nib = np.stack([codes_u8 & 0xF, codes_u8 >> 4], axis=-1).reshape(*codes_u8.shape[:-1], -1)
    vals = _E2M1_SIGNED[nib]
    sf = e4m3_to_f32(scales_e4m3) * np.float32(outer)
    return (vals.reshape(*sf.shape, 16) * sf[..., None]).reshape(vals.shape)


def mxfp8_decode(vals_e4m3, scales_e8m0):
    """values [..., K] E4M3 bytes; scales [..., K/32] E8M0 bytes (2 ** (b - 127))."""
    v = e4m3_to_f32(vals_e4m3)
    sf = np.exp2(np.asarray(scales_e8m0, dtype=np.int32) - 127).astype(np.float32)
    return (v.reshape(*sf.shape, 32) * sf[..., None]).reshape(v.shape)
