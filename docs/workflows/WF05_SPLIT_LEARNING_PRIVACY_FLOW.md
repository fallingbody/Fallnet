# 🔒 Workflow 05: Split-Learning Privacy Flow

This workflow defines the **Zero-Knowledge Privacy Isolation Protocol** of AirCompute, ensuring that no miner in the public network can ever read the user's prompt or the AI's generated response.

---

## 🎯 1. Privacy Guarantee & Threat Model

* **The Threat:** Malicious miners on the P2P network attempting to log user inputs (e.g. medical queries, proprietary code, passwords).
* **The AirCompute Guarantee:**
  * Raw human text is **never** transmitted over the network.
  * Token IDs are **never** transmitted over the network.
  * Only abstract, continuous activation vectors ($H \in \mathbb{R}^{B \times S \times 4096}$) pass between miners.

---

## ⚙️ 2. Split-Learning Execution Diagram

```mermaid
flowchart TD
    subgraph Client Secure Boundary (User Device)
        Prompt["User Prompt: 'My patient has symptom X...'"] --> Tokenizer[Local Tokenizer]
        Tokenizer --> Embed[Local Token Embedding Matrix]
        Embed --> Layer1[Local Transformer Layer 1]
        Layer1 --> ObfuscatedTensor["Obfuscated High-Dimensional Activation H_1"]
    end

    ObfuscatedTensor -->|Encrypted QUIC Socket| SwarmPod[Untrusted Public Swarm Pod: Layers 2 through 79]

    subgraph Client Output Decoding
        SwarmPod -->|Returns Final Hidden Vector H_79| FinalNorm[Local RMSNorm]
        FinalNorm --> LMHead[Local LM Head / Logits Matrix]
        LMHead --> DecodedText["Final Decoded Words Rendered to User Screen"]
    end
```

---

## 🛡️ 3. Mathematical Inversion Defense

Even if a malicious miner intercepts $H_1$, reconstructing the original English tokens requires solving an under-determined, non-linear inverse problem across multiple Attention and MLP projections:
* **Activation Noise Injection:** The client optionally adds a tiny $\epsilon \sim \mathcal{N}(0, 10^{-5})$ differential privacy noise to $H_1$ that has zero impact on LLM output accuracy but mathematically destroys prompt inversion attacks.
