"""Convert an original Inkling 2 checkpoint (NVFP4 or MXFP8 release) to MLX safetensors.

Native mode keeps every weight bit-identical: the routed experts stay in the block-scaled format they
were released in (NVFP4 / MXFP8), relabelled as the uint32 codes + uint8 scales that
`mx.quantized_matmul` consumes, and everything else stays bf16/fp32. `--expert-bits N` instead
re-quantizes the experts to N-bit MLX affine quantization, which is lossy.

The source can be a local directory or a Hub repo id; a Hub source is read with HTTP range requests,
so the checkpoint never has to fit on local disk.

    python -m inkling2_mlx.convert --source thinkingmachines/Inkling-2-Small --out ./Inkling-2-Small-MLX
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import json
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "U32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "I32": 4, "I64": 8}
CHUNK = 32 << 20
SIDE_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
    "tiktoken/tokenizer.model",
]
USER_AGENT = "inkling2-mlx-convert/0.1"


# --------------------------------------------------------------------------------------------- sources


@dataclass
class TensorRef:
    file: str
    dtype: str
    shape: tuple
    offset: int  # absolute byte offset of the tensor data inside `file`
    nbytes: int


class Source:
    tensors: dict

    def read(self, ref: TensorRef, start: int, length: int) -> bytes:
        raise NotImplementedError

    def read_file(self, name: str) -> bytes | None:
        raise NotImplementedError

    def _index(self, file: str, header_len: int, header: dict):
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            self.tensors[name] = TensorRef(file, info["dtype"], tuple(info["shape"]), 8 + header_len + begin, end - begin)

    def iter_bytes(self, ref: TensorRef, chunk: int = CHUNK) -> Iterator[bytes]:
        for start in range(0, ref.nbytes, chunk):
            yield self.read(ref, start, min(chunk, ref.nbytes - start))


class LocalSource(Source):
    def __init__(self, directory):
        self.dir = Path(directory)
        self.tensors = {}
        self._local = threading.local()
        for path in sorted(glob.glob(str(self.dir / "model*.safetensors"))):
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._index(os.path.basename(path), n, json.loads(f.read(n)))

    def read(self, ref, start, length):
        handles = self._local.__dict__.setdefault("handles", {})
        f = handles.get(ref.file) or handles.setdefault(ref.file, open(self.dir / ref.file, "rb"))
        f.seek(ref.offset + start)
        data = f.read(length)
        if len(data) != length:
            raise IOError(f"short read from {ref.file}")
        return data

    def read_file(self, name):
        path = self.dir / name
        return path.read_bytes() if path.exists() else None


class HubSource(Source):
    def __init__(self, repo: str, revision: str = "main", token: str | None = None):
        self.repo, self.revision = repo, revision
        self.token = token or os.environ.get("HF_TOKEN")
        self.tensors = {}
        index = json.loads(self.read_file("model.safetensors.index.json"))
        files = sorted(set(index["weight_map"].values()))
        with futures.ThreadPoolExecutor(8) as pool:
            for file, (n, header) in zip(files, pool.map(self._header, files)):
                self._index(file, n, header)

    def _url(self, file):
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{file}"

    def _get(self, file, start=None, end=None, retries=8):
        headers = {"User-Agent": USER_AGENT}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end}"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        error = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(self._url(file), headers=headers), timeout=300) as r:
                    data = r.read()
                if start is not None and len(data) != end - start + 1:
                    raise IOError(f"short read: {len(data)} of {end - start + 1} bytes")
                return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise
                error = e
            except Exception as e:  # noqa: BLE001 - network errors are retried
                error = e
            time.sleep(min(60, 2.0 * (attempt + 1)))
        raise error

    def _header(self, file):
        n = struct.unpack("<Q", self._get(file, 0, 7))[0]
        return n, json.loads(self._get(file, 8, 8 + n - 1))

    def read(self, ref, start, length):
        return self._get(ref.file, ref.offset + start, ref.offset + start + length - 1)

    def read_file(self, name):
        try:
            return self._get(name)
        except urllib.error.HTTPError:
            return None


# ----------------------------------------------------------------------------------------------- naming

_ATTN = {
    "wq_du.weight": "q_proj.weight",
    "wk_dv.weight": "k_proj.weight",
    "wv_dv.weight": "v_proj.weight",
    "wr_du.weight": "r_proj.weight",
    "wo_ud.weight": "o_proj.weight",
    "gate_attn.weight": "gate_proj.weight",
    "attn_sink_proj.weight": "sink_proj.weight",
    "q_l2_clip.scale": "q_l2_clip.scale",
    "rel_l2_clip.scale": "rel_l2_clip.scale",
    "rel_logits_proj.proj": "rel_logits_proj.proj",
    "k_sconv.weight": "k_sconv.conv1d.weight",
    "v_sconv.weight": "v_sconv.conv1d.weight",
    "k_sconv.static_w": "k_sconv.static_weight",
    "v_sconv.static_w": "v_sconv.static_weight",
    "k_sconv.weight_proj": "k_sconv.weight_proj.weight",
    "v_sconv.weight_proj": "v_sconv.weight_proj.weight",
}
_LAYER = {
    "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
    "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "q_input_norm.weight": "q_input_layernorm.weight",
    "k_input_norm.weight": "k_input_layernorm.weight",
    "v_input_norm.weight": "v_input_layernorm.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
}
_TOP = {
    "model.llm.unembed.weight": "lm_head.weight",
    "model.llm.embed.weight": "model.language_model.embed_tokens.weight",
    "model.llm.embed_norm.weight": "model.language_model.embed_tokens.embed_norm.weight",
    "model.llm.norm.weight": "model.language_model.norm.weight",
    "model.llm.attn_res.output_query": "model.language_model.final_input.query",
    "model.visual.backbone.embeddings.position_embedding": "model.vision_tower.embeddings.position_embedding.weight",
}
_VISION_LAYER = {"ln_1.": "layernorm_before.", "ln_2.": "layernorm_after.", "mlp.c_fc.": "mlp.fc1.",
                 "mlp.c_proj.": "mlp.fc2.", "ls_1.gamma": "lambda_1", "ls_2.gamma": "lambda_2"}
_EXPERT = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight(_scale|_amax|_outer_scale_denominator)?$")
_EXPERT_TARGET = {"w13": "gate_up_proj", "w2": "down_proj"}


def rename(name: str) -> str:
    """Original checkpoint tensor name -> name in this package's module tree (every tensor but the experts)."""
    if name in _TOP:
        return _TOP[name]
    lm = "model.language_model."
    if m := re.fullmatch(r"model\.llm\.attn_res\.attn_queries\.(\d+)", name):
        return f"{lm}layers.{int(m.group(1)) + 1}.attention_input.query"  # layer 0 has no attention query
    if m := re.fullmatch(r"model\.llm\.attn_res\.mlp_queries\.(\d+)", name):
        return f"{lm}layers.{m.group(1)}.mlp_input.query"
    if m := re.fullmatch(r"model\.llm\.layers\.(\d+)\.attn\.(.+)", name):
        return f"{lm}layers.{m.group(1)}.self_attn.{_ATTN[m.group(2)]}"
    if m := re.fullmatch(r"model\.llm\.layers\.(\d+)\.(.+)", name):
        return f"{lm}layers.{m.group(1)}.{_LAYER[m.group(2)]}"
    vt = "model.vision_tower."
    if m := re.fullmatch(r"model\.visual\.backbone\.embeddings\.(.+)", name):
        return f"{vt}embeddings.{m.group(1)}"
    if m := re.fullmatch(r"model\.visual\.backbone\.ln_pre\.(.+)", name):
        return f"{vt}pre_layernorm.{m.group(1)}"
    if m := re.fullmatch(r"model\.visual\.backbone\.layers\.(\d+)\.(.+)", name):
        rest = m.group(2)
        for old, new in _VISION_LAYER.items():
            if rest.startswith(old):
                rest = new + rest[len(old) :]
                break
        return f"{vt}layers.{m.group(1)}.{rest}"
    if m := re.fullmatch(r"model\.visual\.proj_(in|out)\.(.+)", name):
        return f"{vt}merger.linear_{1 if m.group(1) == 'in' else 2}.{m.group(2)}"
    raise KeyError(f"unrecognised checkpoint tensor: {name}")


# ------------------------------------------------------------------------------------------------- plan


@dataclass
class Out:
    name: str
    dtype: str
    shape: tuple
    chunks: Callable[[], Iterator[bytes]]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * DTYPE_SIZE[self.dtype]


def _copy(source: Source, ref: TensorRef, name: str, dtype: str | None = None, shape: tuple | None = None) -> Out:
    out = Out(name, dtype or ref.dtype, tuple(shape or ref.shape), lambda: source.iter_bytes(ref))
    if out.nbytes != ref.nbytes:
        raise ValueError(f"{name}: {out.nbytes} bytes planned but the source tensor has {ref.nbytes}")
    return out


def expert_format(source: Source, layer: int) -> str:
    ref = source.tensors[f"model.llm.layers.{layer}.mlp.experts.w13_weight"]
    scale = source.tensors[f"model.llm.layers.{layer}.mlp.experts.w13_weight_scale"]
    if (ref.dtype, scale.dtype) == ("U8", "F8_E4M3"):
        return "nvfp4"
    if (ref.dtype, scale.dtype) == ("F8_E4M3", "U8"):
        return "mxfp8"
    raise ValueError(f"layer {layer}: experts stored as {ref.dtype} with {scale.dtype} scales")


def _native_expert_outs(source: Source, layer: int, which: str, fmt: str, prefix: str) -> list:
    base = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
    weight, scale = source.tensors[base], source.tensors[base + "_scale"]
    target = prefix + _EXPERT_TARGET[which]
    experts, rows, stored = weight.shape
    if stored % 4:
        raise ValueError(f"{base}: {stored} bytes per row is not a whole number of uint32")
    # same bytes, viewed as the little-endian uint32 words MLX packs quantized weights into
    outs = [_copy(source, weight, target, "U32", (experts, rows, stored // 4)),
            _copy(source, scale, target + "_scales", "U8", scale.shape)]
    if fmt == "nvfp4":
        amax_ref, den_ref = source.tensors[base + "_amax"], source.tensors[base + "_outer_scale_denominator"]

        def outer():
            amax = np.frombuffer(source.read(amax_ref, 0, amax_ref.nbytes), dtype="<f4")
            den = np.frombuffer(source.read(den_ref, 0, den_ref.nbytes), dtype="<f4")
            yield (amax / den[0]).astype("<f4").tobytes()

        outs.append(Out(target + "_outer_scale", "F32", (experts,), outer))
    return outs


class _AffineExpert:
    """Dequantize one stacked expert tensor and re-quantize it with MLX affine quantization."""

    def __init__(self, source: Source, layer: int, which: str, fmt: str, bits: int, group_size: int, batch: int = 16):
        self.source, self.fmt, self.bits, self.group_size, self.batch = source, fmt, bits, group_size, batch
        base = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
        self.weight, self.scale = source.tensors[base], source.tensors[base + "_scale"]
        self.amax = source.tensors.get(base + "_amax")
        self.den = source.tensors.get(base + "_outer_scale_denominator")
        self.experts, self.rows, stored = self.weight.shape
        self.cols = stored * 2 if fmt == "nvfp4" else stored
        self._result = None
        self._lock = threading.Lock()

    def shapes(self):
        groups = self.cols // self.group_size
        return ((self.experts, self.rows, self.cols * self.bits // 32), (self.experts, self.rows, groups))

    def _compute(self):
        import mlx.core as mx

        src, e_bytes = self.source, self.weight.nbytes // self.experts
        s_bytes = self.scale.nbytes // self.experts
        outer = None
        if self.fmt == "nvfp4":
            amax = np.frombuffer(src.read(self.amax, 0, self.amax.nbytes), dtype="<f4")
            outer = amax / np.frombuffer(src.read(self.den, 0, self.den.nbytes), dtype="<f4")[0]
        parts = ([], [], [])
        for start in range(0, self.experts, self.batch):
            n = min(self.batch, self.experts - start)
            codes = np.frombuffer(src.read(self.weight, start * e_bytes, n * e_bytes), dtype="<u4").reshape(n, self.rows, -1)
            scales = np.frombuffer(src.read(self.scale, start * s_bytes, n * s_bytes), dtype=np.uint8).reshape(n, self.rows, -1)
            dense = mx.dequantize(mx.array(codes), mx.array(scales), mode=self.fmt, dtype=mx.float32)
            if outer is not None:
                dense = dense * mx.array(outer[start : start + n])[:, None, None]
            wq, sc, bi = mx.quantize(dense, group_size=self.group_size, bits=self.bits)
            sc, bi = sc.astype(mx.float16), bi.astype(mx.float16)
            mx.eval(wq, sc, bi)
            for acc, arr in zip(parts, (wq, sc, bi)):
                acc.append(np.array(arr))
        return tuple(np.concatenate(p) for p in parts)

    def part(self, i: int) -> Iterator[bytes]:
        with self._lock:
            if self._result is None:
                self._result = list(self._compute())
            data = self._result[i]
            self._result[i] = None
            if all(r is None for r in self._result):
                self._result = None
        raw = memoryview(np.ascontiguousarray(data)).cast("B")
        for start in range(0, len(raw), CHUNK):
            yield raw[start : start + CHUNK]


def _affine_expert_outs(source, layer, which, fmt, prefix, bits, group_size) -> list:
    job = _AffineExpert(source, layer, which, fmt, bits, group_size)
    target = prefix + _EXPERT_TARGET[which]
    wshape, sshape = job.shapes()
    return [Out(target, "U32", wshape, lambda: job.part(0)),
            Out(target + "_scales", "F16", sshape, lambda: job.part(1)),
            Out(target + "_biases", "F16", sshape, lambda: job.part(2))]


def build_plan(source: Source, expert_bits: int | None = None, group_size: int = 64):
    """Returns (shards, native expert format of each layer). A shard is a list of `Out`."""
    layer_ids = sorted({int(m.group(1)) for n in source.tensors if (m := _EXPERT.match(n))})
    formats = [expert_format(source, i) for i in layer_ids]
    base, layers = [], {i: [] for i in layer_ids}
    for name, ref in sorted(source.tensors.items()):
        if _EXPERT.match(name):
            continue
        target = rename(name)
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", target)
        (layers[int(m.group(1))] if m else base).append(_copy(source, ref, target))
    for i, fmt in zip(layer_ids, formats):
        prefix = f"model.language_model.layers.{i}.mlp.experts."
        for which in ("w13", "w2"):
            if expert_bits is None:
                layers[i] += _native_expert_outs(source, i, which, fmt, prefix)
            else:
                layers[i] += _affine_expert_outs(source, i, which, fmt, prefix, expert_bits, group_size)
    return [base] + [layers[i] for i in layer_ids], formats


# ---------------------------------------------------------------------------------------------- writing


def write_shard(path: Path, outs: list, progress: Callable[[int], None] | None = None):
    header, offset = {"__metadata__": {"format": "mlx"}}, 0
    for out in outs:
        header[out.name] = {"dtype": out.dtype, "shape": list(out.shape), "data_offsets": [offset, offset + out.nbytes]}
        offset += out.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for out in outs:
            written = 0
            for chunk in out.chunks():
                f.write(chunk)
                written += len(chunk)
                if progress:
                    progress(len(chunk))
            if written != out.nbytes:
                raise IOError(f"{out.name}: wrote {written} bytes, expected {out.nbytes}")
    os.replace(tmp, path)
    return 8 + len(blob) + offset


def convert(source: Source, out_dir, expert_bits: int | None = None, group_size: int = 64, workers: int = 4,
            activation_quantization: bool = True, on_shard: Callable[[Path], None] | None = None,
            only_shards: set | None = None, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, formats = build_plan(source, expert_bits, group_size)
    total = sum(o.nbytes for s in shards for o in s)
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]

    config = json.loads(source.read_file("config.json"))
    config["quantization"] = {
        "expert_formats": formats if expert_bits is None else ["affine"] * len(formats),
        "activation_formats": formats,
        "activation_quantization": activation_quantization,
        "gate_up_interleaved": True,
        **({"affine_bits": expert_bits, "affine_group_size": group_size} if expert_bits is not None else {}),
    }
    config.setdefault("image_token_id", 200054)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    index = {"metadata": {"total_size": total}, "weight_map": {o.name: names[i] for i, s in enumerate(shards) for o in s}}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for side in SIDE_FILES:
        data = source.read_file(side)
        if data is not None:
            (out_dir / side).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / side).write_bytes(data)

    done, lock, started = [0], threading.Lock(), time.time()

    def progress(n):
        with lock:
            done[0] += n

    def run(i):
        path = out_dir / names[i]
        size = sum(o.nbytes for o in shards[i])
        marker = out_dir / (names[i] + ".done")
        if marker.exists() or (path.exists() and path.stat().st_size > size):
            progress(size)
            log(f"skip  {names[i]} (already converted)")
        else:
            write_shard(path, shards[i], progress)
            rate = done[0] / max(time.time() - started, 1e-6)
            log(f"wrote {names[i]} {size / 1e9:6.2f} GB | total {done[0] / 1e9:7.1f}/{total / 1e9:.1f} GB | {rate / 1e6:6.1f} MB/s")
        if on_shard and not marker.exists():
            on_shard(path)
            marker.write_text("uploaded\n")

    todo = [i for i in range(len(shards)) if only_shards is None or i in only_shards]
    with futures.ThreadPoolExecutor(workers) as pool:
        for f in futures.as_completed([pool.submit(run, i) for i in todo]):
            f.result()
    return names, formats, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="local directory or Hub repo id of the original checkpoint")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--expert-bits", type=int, default=None, choices=[2, 3, 4, 5, 6, 8],
                    help="re-quantize the experts to N-bit MLX affine quantization (lossy); default keeps them native")
    ap.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-activation-quantization", action="store_true")
    ap.add_argument("--upload-repo", default=None, help="upload each shard to this Hub repo as it is finished")
    ap.add_argument("--remove-uploaded", action="store_true", help="delete each shard locally once uploaded")
    args = ap.parse_args()

    source = LocalSource(args.source) if os.path.isdir(args.source) else HubSource(args.source, args.revision)
    on_shard = None
    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()

        def on_shard(path: Path):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.upload_repo,
                            commit_message=f"Add {path.name}")
            if args.remove_uploaded:
                path.unlink()

    convert(source, args.out, args.expert_bits, args.group_size, args.workers,
            activation_quantization=not args.no_activation_quantization, on_shard=on_shard)


if __name__ == "__main__":
    main()
