# 🔀 Workflow 03: MoE Dynamic Expert Routing

This workflow outlines how giant Sparse Mixture-of-Experts models (like **DeepSeek-V3 671B**) are executed across hundreds of consumer devices without overloading network bandwidth.

---

## 🎯 1. Workflow Architecture & Objectives

1. **Gate Evaluation:** Computes gating router logits $G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))$.
2. **Targeted Dispatch:** Routes activations asynchronously **only to the Top-8 active expert nodes**.
3. **Tail-Latency Pruning:** Drops slow outlier experts with low weight ($<0.04$) to guarantee consistent streaming latency.
4. **Weighted Aggregation:** Recombines expert activations into the final layer output tensor.

---

## ⚙️ 2. Dynamic MoE Sparse Routing Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Router as MoE Router Node
    participant E14 as Peer Node A (Expert 14)
    participant E42 as Peer Node B (Expert 42)
    participant E88 as Peer Node C (Expert 88)
    participant E201 as Peer Node D (Expert 201)

    Router->>Router: Forward pass through Dense Layer $\rightarrow$ Compute Top-8 Experts
    
    par Parallel Expert Forward Passes
        Router->>E14: Dispatch FP8 Activation (Weight = 0.38)
        Router->>E42: Dispatch FP8 Activation (Weight = 0.29)
        Router->>E88: Dispatch FP8 Activation (Weight = 0.18)
        Router->>E201: Dispatch FP8 Activation (Weight = 0.03 - Low Weight)
    end

    E14-->>Router: Return E14 Output (Elapsed: 4.2ms)
    E42-->>Router: Return E42 Output (Elapsed: 4.8ms)
    E88-->>Router: Return E88 Output (Elapsed: 5.1ms)
    Note over E201,Router: E201 takes >12ms (Tail Latency) -> Prune and Re-normalize
    
    Router->>Router: Compute Weighted Sum: H_out = (0.38*E14 + 0.29*E42 + 0.18*E88) / 0.85
    Router->>Router: Stream H_out to subsequent model block
```

---

## 🛡️ 3. Hot-Expert Auto-Replication Mechanism

If a specific expert (e.g. Expert #42 for Python coding) receives $>15\%$ of all global tokens:
1. The MoE Router notifies the Swarm Coordinator.
2. The Coordinator instructs 5 nearby idle nodes to clone and load Expert #42's 2.5GB weights.
3. Subsequent requests to Expert #42 are load-balanced across the 5 replicas, eliminating bottlenecks.
