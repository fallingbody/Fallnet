import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

config = LlamaConfig(
    hidden_size=576,
    num_attention_heads=9,
    num_key_value_heads=3,
    head_dim=64,
)
attn = LlamaAttention(config, layer_idx=0)
h = torch.randn(1, 16, 576)
cos = torch.randn(1, 16, 64)
sin = torch.randn(1, 16, 64)
position_embeddings = (cos, sin)

try:
    attn(hidden_states=h, position_embeddings=position_embeddings)
    print("Success!")
except Exception as e:
    print("Error:", e)
