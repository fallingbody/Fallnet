"""
Activation Tensor Codec

Wire format: [mode: uint8][scale: float32][ndims: uint8][dims: uint32 * ndims][payload]

Modes:
  FP16 (0) - 2 bytes/elem. Near-lossless for activations; matched HF greedy output
             token-for-token on SmolLM-135M in our probe. DEFAULT for the pipeline.
  FP8  (1) - 1 byte/elem, float8_e4m3fn with a per-tensor absmax scale. Real FP8.
             LOSSY: halves bandwidth but can change generated tokens. Measure first.
  INT8 (2) - 1 byte/elem, symmetric per-tensor absmax. LOSSY.

Note: before this rewrite, `quantize_fp16_to_fp8` silently produced FP16 (scale=1.0).
"""
import struct
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

FP16, FP8, INT8 = 0, 1, 2
_FP8_MAX = 448.0  # max finite value of float8_e4m3fn


class TensorCodec:
    @staticmethod
    def encode_payload(tensor: torch.Tensor, mode: int = FP16) -> bytes:
        shape = tuple(tensor.shape)
        t = tensor.detach().to("cpu", torch.float32)
        scale = 1.0
        if mode == FP16:
            data = t.to(torch.float16).numpy().tobytes()
        elif mode == FP8:
            absmax = t.abs().max().item()
            scale = absmax / _FP8_MAX if absmax > 0 else 1.0
            data = (t / scale).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
        elif mode == INT8:
            absmax = t.abs().max().item()
            scale = absmax / 127.0 if absmax > 0 else 1.0
            data = torch.round(t / scale).clamp(-127, 127).to(torch.int8).numpy().tobytes()
        else:
            raise ValueError(f"Unknown codec mode {mode}")
        header = struct.pack(f"!BfB{len(shape)}I", mode, scale, len(shape), *shape)
        return header + data

    @staticmethod
    def decode_payload(packed: bytes, device: str = "cpu") -> torch.Tensor:
        """Returns a float16 tensor (callers cast to their compute dtype)."""
        mode, scale, ndims = struct.unpack_from("!BfB", packed, 0)
        offset = 6
        shape = struct.unpack_from(f"!{ndims}I", packed, offset)
        offset += ndims * 4
        buf = packed[offset:]
        numel = int(np.prod(shape)) if shape else 1
        expected = numel * (2 if mode == FP16 else 1)
        if len(buf) != expected:
            raise ValueError(f"Corrupt payload: expected {expected} data bytes, got {len(buf)}")

        if mode == FP16:
            t = torch.from_numpy(np.frombuffer(buf, dtype=np.float16).copy())
        elif mode == FP8:
            raw = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).copy())
            t = raw.view(torch.float8_e4m3fn).to(torch.float32) * scale
        elif mode == INT8:
            t = torch.from_numpy(np.frombuffer(buf, dtype=np.int8).copy()).to(torch.float32) * scale
        else:
            raise ValueError(f"Unknown codec mode {mode}")
        return t.reshape(shape).to(dtype=torch.float16, device=device)

    # --- Backwards-compatible names (now honest) ---------------------------------
    @staticmethod
    def quantize_fp16_to_fp8(tensor: torch.Tensor) -> Tuple[bytes, float, tuple]:
        """Real FP8 (lossy). Use encode_payload(t, FP16) when exactness matters."""
        packed = TensorCodec.encode_payload(tensor, FP8)
        scale = struct.unpack_from("!f", packed, 1)[0]
        return packed, scale, tuple(tensor.shape)

    @staticmethod
    def dequantize_fp8_to_fp16(packed: bytes, device: str = "cpu") -> torch.Tensor:
        """Decodes any mode (the header says which)."""
        return TensorCodec.decode_payload(packed, device=device)

    @staticmethod
    def compute_cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> float:
        f1 = t1.to(torch.float32).flatten().unsqueeze(0)
        f2 = t2.to(torch.float32).flatten().unsqueeze(0)
        return F.cosine_similarity(f1, f2).item()
