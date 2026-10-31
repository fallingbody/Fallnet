"""
Sparse Dynamic Mixture-of-Experts (MoE) Router
Selects Top-K active experts and prunes tail-latency stragglers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class MoEExpert(nn.Module):
    def __init__(self, expert_id: int, d_model: int = 4096):
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
    def __init__(self, num_experts: int = 32, top_k: int = 4, d_model: int = 4096, prune_threshold: float = 0.04):
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
        1. Evaluates gating logits
        2. Selects Top-K experts
        3. Prunes slow/low-weight experts
        4. Re-normalizes and computes weighted sum
        """
        # Gating logits shape: [batch, seq_len, num_experts]
        router_logits = self.gate(x)
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1), self.top_k, dim=-1)
        
        # Tail-pruning: filter out weights below threshold
        valid_mask = weights >= self.prune_threshold
        pruned_weights = weights * valid_mask.float()
        
        # Re-normalize
        weight_sum = pruned_weights.sum(dim=-1, keepdim=True) + 1e-6
        normalized_weights = pruned_weights / weight_sum
        
        # Aggregate expert outputs
        final_output = torch.zeros_like(x)
        dispatched_experts = {}
        
        for k_idx in range(self.top_k):
            exp_id = indices[0, -1, k_idx].item()
            exp_weight = normalized_weights[0, -1, k_idx].item()
            
            if exp_weight > 0:
                expert_out = self.experts[exp_id](x)
                final_output += exp_weight * expert_out
                dispatched_experts[exp_id] = exp_weight
                
        return final_output, dispatched_experts
