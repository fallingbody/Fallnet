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
print("Tuple len:", len(out1))
for i, t in enumerate(out1):
    if isinstance(t, torch.Tensor):
        print(f"out1[{i}] shape:", t.shape)
    else:
        print(f"out1[{i}] type:", type(t))
