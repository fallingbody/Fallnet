import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

config = LlamaConfig(
    hidden_size=576,
    num_attention_heads=9,
    num_key_value_heads=3,
    head_dim=64,
)
layer = LlamaDecoderLayer(config, layer_idx=0)
h = torch.randn(1, 16, 576)
cos = torch.randn(1, 16, 64)
sin = torch.randn(1, 16, 64)
position_embeddings = (cos, sin)

out1 = layer(hidden_states=h, position_embeddings=position_embeddings)
print("Layer 1 output shape:", out1[0].shape)
out2 = layer(hidden_states=out1[0], position_embeddings=position_embeddings)
print("Layer 2 output shape:", out2[0].shape)
