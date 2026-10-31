"""
Sparse Dynamic Mixture-of-Experts (MoE) Router
Selects Top-K active experts across the sequence and aggregates outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class MoEExpert(nn.Module):
    def __init__(self, expert_id: int, d_model: int = 576):
        super().__init__()
        self.expert_id = expert_id
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SparseMoERouter(nn.Module):
    def __init__(self, num_experts: int = 8, top_k: int = 2, d_model: int = 576, prune_threshold: float = 0.04):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.prune_threshold = prune_threshold
        
        # Router Gating Matrix
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Local simulated expert instances for testing/execution
        self.experts = nn.ModuleList([MoEExpert(i, d_model=d_model) for i in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Gating and dispatch across sequence length.
        Input x: [batch, seq_len, d_model]
        """
        router_logits = self.gate(x)
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1), self.top_k, dim=-1)
        
        # Tail-pruning: filter out weights below threshold
        valid_mask = weights >= self.prune_threshold
        pruned_weights = weights * valid_mask.float()
        
        # Re-normalize
        weight_sum = pruned_weights.sum(dim=-1, keepdim=True) + 1e-6
        normalized_weights = pruned_weights / weight_sum
        
        final_output = torch.zeros_like(x)
        dispatched_experts = {}
        
        # Apply routing for all sequence positions
        for k_idx in range(self.top_k):
            # Compute expert output across batch
            for exp_id in range(self.num_experts):
                exp_mask = (indices[..., k_idx] == exp_id) & (normalized_weights[..., k_idx] > 0)
                if exp_mask.any():
                    expert_out = self.experts[exp_id](x)
                    w = normalized_weights[..., k_idx].unsqueeze(-1)
                    final_output += w * expert_out * exp_mask.unsqueeze(-1).float()
                    dispatched_experts[exp_id] = float(normalized_weights[..., k_idx].mean().item())
                
        return final_output, dispatched_experts
