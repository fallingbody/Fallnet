# 🚀 Workflow 02: Speculative Swarm Pipeline Inference

This workflow describes the core **$25\text{+ tokens/second}$ inference pipeline**, showing how speculative decoding collapses network latency across distributed nodes.

---

## 🎯 1. Workflow Architecture & Objectives

1. **Client Draft Phase:** The client's local 0.5B model generates $K=5$ draft tokens locally ($10\text{ms}$).
2. **Batch Forward Traversal:** The 5-token hidden state passes through Node 1 $\rightarrow$ Node 2 $\rightarrow$ Node 3 in a single network pass.
3. **Target Model Verification:** The swarm evaluates all 5 tokens simultaneously using speculative tree verification.
4. **Token Commitment:** Accepted tokens are committed to the client stream, and rejected tokens are corrected immediately.

---

## ⚙️ 2. Speculative Execution Trace

```mermaid
flowchart LR
    subgraph Client Device
        DraftGen[0.5B Drafter: Predicts Tokens [t1, t2, t3, t4, t5]] --> PackFP8[Pack Activation to FP8 Tensor]
    end

    PackFP8 -->|Single Network Hop| NodeA[Miner A: Layers 2-25]
    NodeA -->|Single Network Hop| NodeB[Miner B: Layers 26-50]
    NodeB -->|Single Network Hop| NodeC[Miner C: Layers 51-79]

    subgraph Client Verification & Output
        NodeC -->|Returns 5-Token Target Logits| Verifier[Speculative Accept/Reject Verifier]
        Verifier -->|Accept 4 Tokens, Correct 5th| EmitTokens[Emit [t1, t2, t3, t4, c5] to UI Stream]
    end
```

---

## 📊 3. Latency Comparison: Naive vs Speculative Swarm

| Metric | Traditional P2P (Naive) | AirCompute Speculative Swarm |
| :--- | :--- | :--- |
| **Network Hops per Token** | 4 hops ($100\text{ms}$) | $4\text{ hops} \div 4.2\text{ accepted tokens} = \mathbf{0.95\text{ hops}}$ |
| **Inter-Token Latency (ITL)** | $100\text{ms}$ | **$24\text{ms}$** |
| **Throughput Speed** | $10\text{ tokens/sec}$ | **$41.6\text{ tokens/sec}$** |
| **Bandwidth Overhead** | $4 \times 1\text{ token activations}$ | $1 \times 5\text{ token activation batch}$ (65% lower overhead) |
