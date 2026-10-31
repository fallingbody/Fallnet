# 🧪 Test Suite 01: Unit & Quantization Verification

This test suite verifies the mathematical precision and accuracy of **4-bit AWQ, GGUF, BitNet 1.58b, and FP8 activation quantization** against full FP16/FP32 baseline models.

---

## 🎯 1. Test Objectives & Coverage

1. **Perplexity Degradation Limit:** Quantized weights must maintain $<1.5\%$ perplexity deviation on WikiText-2 / C4 benchmarks.
2. **Layer Forward Math Invariance:** Forward pass activations through individual Transformer layers must match canonical PyTorch output within $\text{Cosine Similarity} > 0.9995$.
3. **FP8 Dynamic Scaling Stability:** Dynamic scaling exponent must not overflow or underflow across $10,000$ consecutive forward passes.

---

## 💻 2. Test Execution Script

```python
import pytest
import torch
import torch.nn.functional as F

def test_layer_quantization_accuracy():
    """Verify quantized layer output closely matches FP16 reference"""
    # 1. Generate random dummy hidden states
    batch_size, seq_len, d_model = 1, 16, 4096
    h_fp16 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float16)
    
    # 2. Reference Layer (FP16) vs Quantized Shard Layer (4-bit AWQ)
    ref_output = run_reference_layer(h_fp16)
    quant_output = run_quantized_shard_layer(h_fp16)
    
    # 3. Compute Cosine Similarity
    sim = F.cosine_similarity(ref_output.flatten().unsqueeze(0), quant_output.flatten().unsqueeze(0)).item()
    
    assert sim >= 0.9995, f"Quantization drift too high: Cosine Similarity = {sim:.5f} < 0.9995"
```
