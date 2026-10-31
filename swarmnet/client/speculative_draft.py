"""
Client Speculative Draft Engine & Dynamic K Adaptation
"""
import torch
import torch.nn as nn
from typing import List, Tuple

class SpeculativeDrafter:
    def __init__(self, vocab_size: int = 32000, d_model: int = 1024, initial_k: int = 5, device: str = "cpu"):
        self.k = initial_k
        self.min_k = 1
        self.max_k = 8
        self.vocab_size = vocab_size
        self.device = device
        self.acceptance_history: List[float] = []
        
        # Tiny 0.5B-equivalent lightweight draft model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.draft_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.to(device)

    def to(self, device):
        self.embed.to(device)
        self.draft_layer.to(device)
        self.head.to(device)
        return self

    def generate_draft_tokens(self, current_input_ids: torch.Tensor) -> List[int]:
        """
        Quickly generates K candidate tokens locally on client device.
        """
        drafts = []
        curr_seq = current_input_ids.clone().to(self.device)
        
        with torch.no_grad():
            for _ in range(self.k):
                x = self.embed(curr_seq)
                h = self.draft_layer(x)
                logits = self.head(h[:, -1, :])
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                drafts.append(next_tok.item())
                curr_seq = torch.cat([curr_seq, next_tok], dim=1)
                
        return drafts

    def record_verification_result(self, accepted_count: int):
        """
        Dynamically adapts K based on rolling acceptance rate.
        """
        rate = accepted_count / max(self.k, 1)
        self.acceptance_history.append(rate)
        if len(self.acceptance_history) > 10:
            self.acceptance_history.pop(0)
            
        avg_rate = sum(self.acceptance_history) / len(self.acceptance_history)
        if avg_rate > 0.75 and self.k < self.max_k:
            self.k += 1
        elif avg_rate < 0.40 and self.k > self.min_k:
            self.k -= 1
