# 🌐 AirCompute (SwarmNet): Decentralized AI Compute as a Utility

> **Vision**: Transform frontier Artificial Intelligence into a self-sustaining, ubiquitous, low-cost utility—just like mobile data packs, Wi-Fi, or electricity—by pooling the world's idle consumer and enterprise hardware into a decentralized, peer-to-peer compute mesh.

---

## 🏛️ 1. High-Level System Architecture

In traditional cloud computing, massive AI models (like DeepSeek-V3 671B or Llama-3.3 70B) are locked inside centralized data centers requiring multi-million dollar GPU clusters. 

**AirCompute breaks this monopoly** by slicing massive open-source models across a decentralized P2P network of everyday devices (laptops, MacBooks, gaming PCs, edge servers, and phones).

```mermaid
flowchart TD
    subgraph Client Layer
        UserApp([User Application / Web Client]) --> LocalPre[Local Split-Learning Client\n(Embedding + Layer 1 + 0.5B Speculative Drafter)]
    end

    LocalPre -->|Compressed FP8 Hidden Activations + 5 Draft Tokens| SwarmRouter[Geo-Clustered Swarm Router / DHT]

    subgraph Decentralized P2P Compute Mesh (Pipeline & MoE Sharding)
        SwarmRouter --> PodAlpha[Geo-Pod: Central Europe < 10ms Ping]
        
        subgraph PodAlpha [Active Inference Pod]
            Node1[Peer Node 1: Layers 2-10\nPrimary + Hot-Standby] -->|FP8 Tensor Stream| Node2[Peer Node 2: Layers 11-20\nPrimary + Hot-Standby]
            Node2 -->|FP8 Tensor Stream| Node3[Peer Node 3: MoE Experts 1-64\nDynamic Sparse Activation]
            Node3 -->|FP8 Tensor Stream| Node4[Peer Node 4: Layers 50-80\nPrimary + Hot-Standby]
        end
    end

    Node4 -->|Verified Multi-Token Vector Batch| LocalPost[Local LM Head\n(Client-Side Text Dequantization & Rendering)]
    LocalPost --> UserApp

    subgraph Trust, Verification & Economic Settlement
        TrapVerifier[Deterministic Trap-Prompt Verifier]
        StateChannel[(State Channel Micro-Ledger)]
        
        Node1 -.->|Heartbeat + Proof of Inference| TrapVerifier
        Node2 -.->|Heartbeat + Proof of Inference| TrapVerifier
        Node3 -.->|Heartbeat + Proof of Inference| TrapVerifier
        Node4 -.->|Heartbeat + Proof of Inference| TrapVerifier
        
        TrapVerifier -->|Award Token Credits| StateChannel
        UserApp -->|Settle Micro-Debits| StateChannel
    end
```

---

## 🧩 2. Core Architectural Pillars

### 1. The Prosumer "BitTorrent for AI" Model
Every participant on the network is both a **Producer** and a **Consumer** (Prosumer):
* **When Idle (e.g. Charging / Overnight):** Your device runs a background worker (15MB RAM footprint) hosting 1–2 model shards or MoE experts, processing inference batches for others and accumulating **"AI Fuel Credits"**.
* **When Active (Asking Questions):** Your app burns your accumulated fuel credits to query frontier models (DeepSeek-R1, Llama-70B) for free.
* **Non-Contributing Users:** Can purchase prepaid "AI Data Packs" (e.g. ₹20 / $0.50 for 5 Million tokens), with payments distributed directly to active seeders.

---

### 2. Mixture-of-Experts (MoE) Dynamic Sharding
Running a 671B parameter model (e.g. DeepSeek-V3) is impossible on a single consumer PC. However, MoE models possess a unique structural advantage:
* Total Parameters: **671 Billion**
* Active Parameters per Token: **~37 Billion** (8 out of 256 experts activated).
* **AirCompute Sharding Strategy:**
  * The 256 expert networks are distributed across 256 distinct peer nodes (holding only ~2.5GB each).
  * The routing gate directs activations **only to the 8 specific peers** holding the active experts for that token.
  * The remaining 248 peers remain in low-power idle, dramatically minimizing global energy and bandwidth usage.

---

### 3. Speculative Swarm Decoding (Overcoming the Latency Trap)
Sequential token generation across 8 network hops over consumer internet would normally limit speed to ~5 tokens/sec ($8 \times 25\text{ms} = 200\text{ms}$ per token).

**AirCompute's Speculative Swarm Algorithm:**
1. The client's local device runs a tiny 0.5B draft model to generate $K=5$ speculative tokens locally in parallel ($10\text{ms}$).
2. The entire 8-node swarm evaluates and verifies all 5 tokens **in a single network traversal pass**.
3. **Result:** 5 tokens verified in 1 round-trip ($200\text{ms}$ total $\rightarrow$ **$25\text{ tokens/sec}$** over standard internet).

---

### 4. Split-Learning for Total Privacy
* The raw text prompt **never leaves the user's device**.
* The user's device executes Tokenization, Embedding, and Layer 1 locally, converting human words into a high-dimensional mathematical vector ($d_{\text{model}} = 4096$).
* The swarm only sees and computes mathematical transformations on abstract vectors without knowing the underlying text.
* The final output vector is sent back to the user's device, which executes the final Normalization and LM Head locally to decode the words.

---

### 5. Deterministic Trap-Prompts & Optimistic Slashing
To prevent malicious or lazy miners from submitting random noise to collect rewards:
* The coordinator silently injects synthetic **"Trap Prompts"** (pre-calculated mathematical vectors with known output hashes) into 3% of traffic.
* If a node returns a mismatched hash, it is instantly flagged as Byzantine/Dishonest.
* The dishonest node's unbonded reward balance is slashed and redistributed to honest peers as a bounty.

---

## 🛠️ 3. Production Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Edge Compute Core** | `llama.cpp` (C++) / `Candle` (Rust) | Zero-dependency 4-bit/2-bit quantized execution on CPU, Apple Metal, Nvidia CUDA, AMD ROCm |
| **High-Throughput Swarm** | `vLLM` / `SGLang` | PagedAttention, Continuous Batching, and Chunked Prefill for dedicated GPU peers |
| **P2P Networking** | `libp2p` (Rust/Go) + `WebRTC` | Kademlia DHT discovery, STUN/TURN NAT hole-punching, Noise-encrypted multiplexed streams |
| **Binary Protocol** | `QUIC` (HTTP/3) + FlatBuffers / Protobuf | Zero-copy binary serialization of compressed FP8 activation tensors |
| **Client Background App**| `Tauri` (Rust + React) | 15MB ultra-lightweight desktop tray client with battery/thermal awareness |
| **Browser Seeder** | `WebGPU` + Transformers.js | 1-click browser-based compute sharing without installation |
| **Micro-Billing** | Ed25519 State Channels + Redis | Sub-millisecond cryptographic micropayments with zero blockchain gas fees |
| **Developer Gateway** | FastAPI / Axum | 100% OpenAI-compatible endpoint interface (`/v1/chat/completions`) |

---

## 📚 Complete Multi-Agent Blueprint Index

Explore the dedicated documentation modules:

* [**Agents Directory (`docs/agents/`)**](file:///docs/agents/)
* [**Workflows Directory (`docs/workflows/`)**](file:///docs/workflows/)
* [**Debugging Directory (`docs/debugging/`)**](file:///docs/debugging/)
* [**Testing Directory (`docs/testing/`)**](file:///docs/testing/)
* [**Optimization Directory (`docs/optimization/`)**](file:///docs/optimization/)
