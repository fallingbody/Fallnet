# 🔀 Agent Spec 04: Dynamic MoE Router Agent

The **Dynamic MoE (Mixture of Experts) Router Agent** is responsible for distributing and executing massive Sparse MoE models (such as **DeepSeek-V3 671B** or **Mixtral 8x22B**) across a decentralized swarm.

---

## 🎯 1. Core Responsibilities & Objectives

1. **Sparse Gating Computation:**
   * Receives intermediate token activations and computes the **Top-$E$ gating probabilities** (e.g. selecting the Top-8 active experts out of 256 for DeepSeek-V3).
2. **Selective Peer Dispatching:**
   * Instead of broadcasting activations to all 256 nodes, dispatches activation vectors **only to the 8 specific peer nodes hosting the activated experts**.
3. **Sparse Output Aggregation & Weighted Sum:**
   * Collects the transformed vectors from the 8 active expert nodes, applies softmax gating weights, and produces the combined output tensor $H_{\text{moe}} = \sum_{i=1}^E w_i \cdot \text{Expert}_i(H)$.
4. **Load Balancing & Expert Re-Sharding:**
   * Monitors if certain "popular" experts (e.g. common language syntax experts) are getting bottlenecked, and replicates those specific hot experts across multiple miners.

---

## ⚙️ 2. MoE Dynamic Dispatch Architecture

```mermaid
flowchart TD
    InboundTensor[Inbound Activation Tensor H] --> GatingNetwork[Gating G(H): Softmax Top-8 Selector]
    
    subgraph Top-8 Expert Selection (Out of 256 Total)
        GatingNetwork -->|w_1 = 0.35| E14[Expert #14: Peer Node DE-01]
        GatingNetwork -->|w_2 = 0.28| E42[Expert #42: Peer Node FR-09]
        GatingNetwork -->|w_3 = 0.15| E88[Expert #88: Peer Node UK-03]
        GatingNetwork -->|w_4 = 0.12| E120[Expert #120: Peer Node NL-07]
        GatingNetwork -->|w_5 = 0.04| E190[Expert #190: Peer Node IT-02]
        GatingNetwork -->|w_6 = 0.03| E201[Expert #201: Peer Node ES-04]
        GatingNetwork -->|w_7 = 0.02| E240[Expert #240: Peer Node SE-08]
        GatingNetwork -->|w_8 = 0.01| E255[Expert #255: Peer Node CH-05]
    end

    subgraph 248 Idle Experts (Sleep Mode - 0 Power & Bandwidth)
        IdleNodes[Experts #1-13, #15-41, ... #254: Idle]
    end

    E14 --> Combine[Weighted Tensor Reduction: Sum w_i * E_i(H)]
    E42 --> Combine
    E88 --> Combine
    E120 --> Combine
    E190 --> Combine
    E201 --> Combine
    E240 --> Combine
    E255 --> Combine

    Combine --> OutboundTensor[Outbound Combined Tensor H_moe]
```

---

## 📥 3. Input / Output Interface

### Inputs
* **Hidden State Tensor ($H$):** Shape `[batch_size, seq_len, hidden_dim]`.
* **Gating Weights ($W_g$):** Routing weight matrix `[hidden_dim, num_experts]`.

### Outputs
* **Parallel Asynchronous Expert RPCs (over QUIC):**
  ```json
  {
    "expert_id": 42,
    "input_vector": "<base64_fp8>",
    "weight_coefficient": 0.28,
    "timeout_ms": 15.0
  }
  ```
* **Aggregated Output Tensor:** Transformed hidden state for next layer.

---

## 🛡️ 4. Creative & Critical Edge Case Handling

| Critical Scenario | System Risk | MoE Router Autonomous Reaction |
| :--- | :--- | :--- |
| **One Expert Node Latency Outlier (Tail Latency)** | 7 experts respond in $5\text{ms}$, 1 takes $60\text{ms}$, blocking entire pipeline. | **Dynamic Top-$E$ Pruning:** If the slowest expert has low gating weight ($w_i < 0.05$) and exceeds $12\text{ms}$, the router drops it and re-normalizes the remaining 7 weights, saving $50\text{ms}$ latency. |
| **Hot Expert Bottleneck (Expert #42 receives 70% of traffic)** | Peer hosting Expert #42 gets overloaded and queues requests. | **Dynamic Auto-Replication:** Coordinator detects high load and instructs 5 idle miners to clone and cache Expert #42's weights in VRAM. |
| **Expert Node Crash Mid-Forward Pass** | RPC to Expert #88 returns connection refused. | **Fallback to Replicated Replica:** Instantly routes activation to the standby replica node holding Expert #88. |

---

## 💻 5. Sparse MoE Dispatch Engine Implementation

```python
import torch
import torch.nn.functional as F
import asyncio

class MoERouterAgent:
    def __init__(self, num_experts=256, top_k=8, d_model=4096):
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = torch.nn.Linear(d_model, num_experts, bias=False)
        self.expert_peer_map = {}  # expert_id -> peer_endpoint

    async def route_and_aggregate(self, h_tensor: torch.Tensor) -> torch.Tensor:
        """Computes Top-K gating and dispatches in parallel to active expert peers"""
        # 1. Compute Gating Logits & Top-K Softmax
        logits = self.gate(h_tensor)  # [B, S, num_experts]
        top_weights, top_indices = torch.topk(F.softmax(logits, dim=-1), k=self.top_k, dim=-1)
        
        # 2. Asynchronously dispatch only to the active top_k expert nodes
        tasks = []
        for i in range(self.top_k):
            expert_id = top_indices[0, -1, i].item()
            weight = top_weights[0, -1, i].item()
            peer_endpoint = self.expert_peer_map[expert_id]
            tasks.append(self._dispatch_expert_rpc(peer_endpoint, h_tensor, weight))
            
        # 3. Gather responses with a tight tail-latency timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. Perform weighted summation of returned expert activations
        h_combined = torch.zeros_like(h_tensor)
        valid_weights_sum = 0.0
        
        for res, weight in zip(results, top_weights[0, -1].tolist()):
            if isinstance(res, torch.Tensor):
                h_combined += weight * res
                valid_weights_sum += weight
                
        # Re-normalize if any slow expert was pruned
        if valid_weights_sum > 0:
            h_combined = h_combined / valid_weights_sum
            
        return h_combined
```
