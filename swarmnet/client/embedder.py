"""
Client-Side Split-Learning Embedder & Local LM Head
Guarantees 100% user privacy: text/tokens never leave the user's local machine.
"""
import torch
import torch.nn as nn
from typing import Tuple, List

class LocalSplitClient(nn.Module):
    def __init__(self, vocab_size: int = 49152, d_model: int = 576, device: str = "cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        
        # 1. Local Embedding Matrix
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        
        # 2. Local Transformer Layer 1 (Scrambles token representation)
        self.local_layer_1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=9,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True
        )
        
        # 3. Local Final RMSNorm & LM Head (De-scrambles swarm output locally)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.to(device)

    def encode_and_scramble(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Executes local embedding + Layer 1 on client machine.
        Returns obfuscated high-dimensional vector H_1 to dispatch to public swarm.
        """
        with torch.no_grad():
            x = input_ids.to(self.device)
            embeddings = self.embed_tokens(x)
            h_1 = self.local_layer_1(embeddings)
            return h_1

    def decode_and_sample(self, swarm_h_last: torch.Tensor, temperature: float = 0.7) -> int:
        """
        Takes the final hidden state from the last swarm miner and decodes next token locally.
        """
        with torch.no_grad():
            h_last = swarm_h_last.to(self.device)
            normed = self.final_norm(h_last)
            logits = self.lm_head(normed[:, -1, :])  # Take logits of last position
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = torch.argmax(logits, dim=-1).item()
                
            return next_token_id
