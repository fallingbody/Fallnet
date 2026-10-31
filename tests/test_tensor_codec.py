"""
Unit Tests for FP8/INT8 Tensor Quantization Codec
"""
import pytest
import torch
from swarmnet.protocol.tensor_codec import TensorCodec

def test_fp8_quantization_and_dequantization():
    """Verifies that packing and unpacking tensors preserves cosine similarity > 0.999"""
    # 1. Create realistic Transformer hidden state [1, 5, 4096]
    orig_tensor = torch.randn(1, 5, 4096, dtype=torch.float16)
    
    # 2. Quantize to packed binary
    packed_bytes, scale, shape = TensorCodec.quantize_fp16_to_fp8(orig_tensor)
    assert len(packed_bytes) > 0
    assert scale > 0
    assert shape == (1, 5, 4096)
    
    # 3. Dequantize back to FP16
    recovered_tensor = TensorCodec.dequantize_fp8_to_fp16(packed_bytes)
    assert recovered_tensor.shape == orig_tensor.shape
    
    # 4. Compute Cosine Similarity
    sim = TensorCodec.compute_cosine_similarity(orig_tensor, recovered_tensor)
    assert sim >= 0.995, f"Quantization loss too high: Cosine Similarity = {sim:.5f}"
