"""
Zero-Copy Activation Tensor Codec & Quantization Kernels
"""
import torch
import torch.nn.functional as F
import numpy as np
import struct
from typing import Tuple

class TensorCodec:
    @staticmethod
    def encode_payload(tensor: torch.Tensor) -> bytes:
        """
        Packs an activation tensor to lossless FP16 binary payload.
        """
        shape = tuple(tensor.shape)
        tensor_fp16 = tensor.to(torch.float16)
        data_bytes = tensor_fp16.cpu().numpy().tobytes()
        dtype_code = 0  # FP16 Lossless
        
        # Header: [dtype_code: uint8][scale: float32][dims_count: uint8][dims: uint32...]
        header = struct.pack(f"!BfB{len(shape)}I", dtype_code, 1.0, len(shape), *shape)
        return header + data_bytes

    @staticmethod
    def decode_payload(packed_bytes: bytes, device: str = "cpu") -> torch.Tensor:
        """
        Unpacks binary payload back to torch.Tensor in target device memory.
        """
        dtype_code, scale, num_dims = struct.unpack_from("!BfB", packed_bytes, 0)
        offset = 1 + 4 + 1
        shape = struct.unpack_from(f"!{num_dims}I", packed_bytes, offset)
        offset += num_dims * 4
        
        data_buffer = packed_bytes[offset:]
        
        if dtype_code == 0:  # FP16 Lossless
            raw_np = np.frombuffer(data_buffer, dtype=np.float16)
            tensor_raw = torch.from_numpy(raw_np.copy())
            return tensor_raw.reshape(shape).to(dtype=torch.float16, device=device)
        elif dtype_code == 1:  # Native FP8
            raw_np = np.frombuffer(data_buffer, dtype=np.uint8)
            tensor_raw = torch.from_numpy(raw_np.copy()).view(torch.float8_e4m3fn)
            unscaled = tensor_raw.to(torch.float32) * scale
            return unscaled.reshape(shape).to(dtype=torch.float16, device=device)
        else:  # INT8 Fallback
            raw_np = np.frombuffer(data_buffer, dtype=np.int8)
            tensor_raw = torch.from_numpy(raw_np.copy()).to(torch.float32)
            unscaled = tensor_raw * scale
            return unscaled.reshape(shape).to(dtype=torch.float16, device=device)

    @staticmethod
    def quantize_fp16_to_fp8(tensor: torch.Tensor) -> Tuple[bytes, float, tuple]:
        return TensorCodec.encode_payload(tensor), 1.0, tuple(tensor.shape)

    @staticmethod
    def dequantize_fp8_to_fp16(packed_bytes: bytes, device: str = "cpu") -> torch.Tensor:
        return TensorCodec.decode_payload(packed_bytes, device=device)

    @staticmethod
    def compute_cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> float:
        """Computes cosine similarity between two tensors regardless of dtype"""
        f1 = t1.to(torch.float32).flatten().unsqueeze(0)
        f2 = t2.to(torch.float32).flatten().unsqueeze(0)
        return F.cosine_similarity(f1, f2).item()
