import torch
import numpy as np
import pytest
from swarmnet.protocol.tensor_codec import TensorCodec, FP16, FP8

def test_codec_cosine_similarity():
    # Test with typical transformer hidden dimension
    t_orig = torch.randn(1, 12, 576, dtype=torch.float32)
    
    # 1. FP16 Lossless Mode
    fp16_bytes = TensorCodec.encode_payload(t_orig, mode=FP16)
    t_fp16 = TensorCodec.decode_payload(fp16_bytes)
    sim_fp16 = TensorCodec.compute_cosine_similarity(t_orig, t_fp16)
    assert sim_fp16 >= 0.9999, f"FP16 similarity too low: {sim_fp16}"

    # 2. FP8 Mode
    fp8_bytes, scale, shape = TensorCodec.quantize_fp16_to_fp8(t_orig)
    t_fp8 = TensorCodec.dequantize_fp8_to_fp16(fp8_bytes)
    sim_fp8 = TensorCodec.compute_cosine_similarity(t_orig, t_fp8)
    assert sim_fp8 >= 0.99, f"FP8 similarity too low: {sim_fp8}"
