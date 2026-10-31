"""
Layer Miner Execution Engine
Executes forward-pass Transformer blocks across assigned model layer slices.
"""
import logging
import gc
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)
from transformers import AutoModelForCausalLM
from swarmnet.miner.paged_kv import PagedKVCachePool
from swarmnet.protocol.tensor_codec import TensorCodec

class ShardedLayerBlock(nn.Module):
    def __init__(self, layer_idx: int, d_model: int = 576, num_heads: int = 9):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Pre-LN Feedforward
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x

class LayerMinerRunner:
    def __init__(self, layer_start: int, layer_end: int, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", d_model: int = 1536, device: str = "cpu"):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.d_model = d_model
        self.device = device
        self.model_name = model_name
        
        print(f"🧠 Loading {model_name} layers {layer_start} to {layer_end}...")
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        except Exception:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            
        self.layers = hf_model.model.layers[layer_start : layer_end + 1].to(device)
        self.rotary_emb = hf_model.model.rotary_emb.to(device)
        
        # Free unused layers and LM head from memory
        del hf_model
        import gc
        gc.collect()
        
        self.kv_pool = PagedKVCachePool(device=device)

    def execute_forward_slice(self, inbound_packed_bytes: bytes, session_id: str) -> bytes:
        """
        1. Dequantize inbound FP8 tensor
        2. Execute forward pass across assigned layers
        3. Quantize output back to FP8
        """
        # Dequantize
        h_in = TensorCodec.dequantize_fp8_to_fp16(inbound_packed_bytes, device=self.device)
        
        # Forward pass through local layer block
        h_out = h_in.to(self.layers[0].input_layernorm.weight.dtype)
        batch_size, seq_len, _ = h_out.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(h_out, position_ids=position_ids)
        
        with torch.no_grad():
            for layer in self.layers:
                layer_out = layer(h_out, position_embeddings=position_embeddings, position_ids=position_ids)
                h_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                
        # Re-quantize to FP8 for network transmission
        outbound_bytes, _, _ = TensorCodec.quantize_fp16_to_fp8(h_out)
        return outbound_bytes
