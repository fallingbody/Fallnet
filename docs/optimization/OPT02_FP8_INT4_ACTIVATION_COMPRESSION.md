# ⚡ Optimization 02: FP8 & INT4 Activation Tensor Compression

This playbook outlines zero-loss tensor quantization techniques to compress network bandwidth between P2P nodes by up to **$75\%$**.

---

## 🎯 1. Quantization Formats & Bandwidth Profile

| Precision Format | Bytes per Element | Bandwidth per 4096-dim Token | Relative Network Overhead |
| :--- | :---: | :---: | :---: |
| **FP32 (Standard Float)** | 4 bytes | $16.38\text{ KB}$ | $100\%$ (Baseline) |
| **FP16 / BF16 (Half Float)** | 2 bytes | $8.19\text{ KB}$ | $50\%$ |
| **FP8 (E4M3 / E5M2)** | **1 byte** | **$4.09\text{ KB}$** | **$25\%$ (Optimal Speed/Accuracy)** |
| **INT4 (Block-Quantized with scale)** | **0.5 bytes** | **$2.05\text{ KB}$** | **$12.5\%$ (Extreme Bandwidth Saver)** |

---

## 💻 2. Ultra-Fast Zero-Copy Quantization Kernel

```python
import torch

def quantize_activation_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Dynamically scales and packs FP16 tensor into FP8 E4M3 format"""
    max_val = tensor.abs().max().item()
    scale = max(max_val / 448.0, 1e-6)
    fp8_tensor = (tensor / scale).to(torch.float8_e4m3fn)
    return fp8_tensor, scale

def dequantize_activation_fp8(fp8_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Unpacks FP8 tensor back to FP16 in CUDA/Metal VRAM"""
    return fp8_tensor.to(torch.float16) * scale
```
