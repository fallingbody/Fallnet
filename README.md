---
title: FALL • Decentralized P2P AI Mesh
emoji: 🍂
colorFrom: amber
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🍂 FALL (SwarmNet)
### *Self-Sustaining Decentralized P2P AI Compute & Inference Utility*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Architecture: P2P Swarm](https://img.shields.io/badge/Architecture-P2P_Swarm-green.svg)]()
[![Model Support: MoE & Dense](https://img.shields.io/badge/Models-DeepSeek--V3_|_Llama--3.3_|_SmolLM--135M-purple.svg)]()

FALL is a decentralized, peer-to-peer AI compute mesh that turns frontier AI models into a low-cost, self-sustaining utility by pooling idle consumer hardware across the globe.

---

## 🏛️ Autonomous Documentation Governance & Maintenance Council

All repository documentation is autonomously maintained, audited, and updated through a 3-tier hierarchical engineering council:

* 👑 [**`docs/governance/TEAM_LEAD_EXECUTIVE_DIRECTIVE.md`**](file:///docs/governance/TEAM_LEAD_EXECUTIVE_DIRECTIVE.md) — **Team Lead / Principal Architect (Supreme Authority)**: Exclusive right to edit its own file; direct unilateral write and override authority across all senior and junior `.md` files in the repository.
* 🧠 [**`docs/governance/SENIOR_ENGINEER_1_SYSTEMS_AND_ARCH.md`**](file:///docs/governance/SENIOR_ENGINEER_1_SYSTEMS_AND_ARCH.md) — **Senior CS Engineer 1 (Systems & Architecture)**: Audits tensor math, distributed P2P protocols, GPU/Metal kernels, and submits Systems RFCs directly to the Team Lead.
* 🛡️ [**`docs/governance/SENIOR_ENGINEER_2_RELIABILITY_AND_SECURITY.md`**](file:///docs/governance/SENIOR_ENGINEER_2_RELIABILITY_AND_SECURITY.md) — **Senior CS Engineer 2 (Reliability & Security)**: Audits Byzantine defenses, chaos testing, failover recovery paths, and submits Security/QA RFCs directly to the Team Lead.

---

## 📑 Complete Documentation Directory

### 🏛️ Core Architecture
* [**`docs/ARCHITECTURE_OVERVIEW.md`**](file:///docs/ARCHITECTURE_OVERVIEW.md) — High-level architecture, MoE sharding, speculative decoding, and prosumer tokenomics.

---

### 🤖 1. Multi-Agent Specifications (`docs/agents/`)
Detailed role definitions, internal state, and I/O specifications for every agent in the mesh:
* [**`01_CLIENT_DRAFT_AGENT.md`**](file:///docs/agents/01_CLIENT_DRAFT_AGENT.md) — Local Split-Learning, Speculative Drafter, Local LM Head.
* [**`02_SWARM_COORDINATOR_AGENT.md`**](file:///docs/agents/02_SWARM_COORDINATOR_AGENT.md) — DHT Geo-Clustering, Pod Topology Manager, Load Balancer.
* [**`03_LAYER_MINER_AGENT.md`**](file:///docs/agents/03_LAYER_MINER_AGENT.md) — Primary & Hot-Standby Layer Worker, Memory & KV Manager.
* [**`04_MOE_ROUTER_AGENT.md`**](file:///docs/agents/04_MOE_ROUTER_AGENT.md) — Sparse Dynamic MoE Expert Dispatcher.
* [**`05_TRAP_VERIFIER_AGENT.md`**](file:///docs/agents/05_TRAP_VERIFIER_AGENT.md) — Byzantine Fault Detector, Trap-Prompt Validator & Slasher.
* [**`06_LEDGER_BILLING_AGENT.md`**](file:///docs/agents/06_LEDGER_BILLING_AGENT.md) — State Channel Micropayment & Prosumer Fuel Credit Manager.

---

### 🔄 2. Operational Workflows (`docs/workflows/`)
Step-by-step state machines, communication protocols, and edge-case handling:
* [**`WF01_PEER_DISCOVERY_AND_POD_SETUP.md`**](file:///docs/workflows/WF01_PEER_DISCOVERY_AND_POD_SETUP.md) — Kademlia DHT discovery and <10ms Geo-Pod clustering.
* [**`WF02_PIPELINE_INFERENCE_AND_SPECULATION.md`**](file:///docs/workflows/WF02_PIPELINE_INFERENCE_AND_SPECULATION.md) — Multi-token speculative pipeline execution.
* [**`WF03_MOE_DYNAMIC_EXPERT_ROUTING.md`**](file:///docs/workflows/WF03_MOE_DYNAMIC_EXPERT_ROUTING.md) — Sparse 8-of-256 MoE routing across consumer nodes.
* [**`WF04_HOT_STANDBY_AND_SELF_HEALING.md`**](file:///docs/workflows/WF04_HOT_STANDBY_AND_SELF_HEALING.md) — Instant node dropout failover without context loss.
* [**`WF05_SPLIT_LEARNING_PRIVACY_FLOW.md`**](file:///docs/workflows/WF05_SPLIT_LEARNING_PRIVACY_FLOW.md) — Zero-knowledge client-side embedding and head isolation.
* [**`WF06_PROSUMER_CREDIT_SETTLEMENT.md`**](file:///docs/workflows/WF06_PROSUMER_CREDIT_SETTLEMENT.md) — Instant state channel micro-billing and fuel credit balance.

---

### 🔍 3. Debugging Playbooks (`docs/debugging/`)
Root cause analysis, diagnostic decision trees, and automated self-healing loops:
* [**`DB01_NETWORK_NAT_AND_LATENCY_ERRORS.md`**](file:///docs/debugging/DB01_NETWORK_NAT_AND_LATENCY_ERRORS.md) — STUN/TURN timeouts, NAT hole punch failures, high jitter.
* [**`DB02_TENSOR_MISMATCH_AND_KV_DESYNC.md`**](file:///docs/debugging/DB02_TENSOR_MISMATCH_AND_KV_DESYNC.md) — Dimension mismatch, KV cache drift, FP8 dequantization errors.
* [**`DB03_EDGE_OOM_AND_THERMAL_THROTTLING.md`**](file:///docs/debugging/DB03_EDGE_OOM_AND_THERMAL_THROTTLING.md) — Consumer RAM/VRAM exhaustion, thermal downclocking, battery protections.
* [**`DB04_BYZANTINE_CHEAT_AND_POISONING.md`**](file:///docs/debugging/DB04_BYZANTINE_CHEAT_AND_POISONING.md) — Detecting fake activations, Sybil attacks, and malicious weights.
* [**`DB05_AUTONOMOUS_RCA_AND_HEALING_LOOP.md`**](file:///docs/debugging/DB05_AUTONOMOUS_RCA_AND_HEALING_LOOP.md) — 5-step automated closed-loop bug resolution engine.

---

### 🧪 4. Testing Matrix (`docs/testing/`)
Comprehensive test plans from unit level to global chaos engineering:
* [**`TS01_UNIT_AND_QUANTIZATION_TESTS.md`**](file:///docs/testing/TS01_UNIT_AND_QUANTIZATION_TESTS.md) — Layer math verification, BitNet/AWQ quantization precision.
* [**`TS02_MULTI_NODE_SWARM_INTEGRATION.md`**](file:///docs/testing/TS02_MULTI_NODE_SWARM_INTEGRATION.md) — 10-node local virtual swarm integration test suite.
* [**`TS03_CHAOS_AND_FAULT_INJECTION.md`**](file:///docs/testing/TS03_CHAOS_AND_FAULT_INJECTION.md) — Simulating random peer termination during active streaming.
* [**`TS04_BYZANTINE_SECURITY_TESTBED.md`**](file:///docs/testing/TS04_BYZANTINE_SECURITY_TESTBED.md) — Malicious node injection, slashing validation, trap-prompt hit rates.
* [**`TS05_BENCHMARK_AND_TTFT_LATENCY.md`**](file:///docs/testing/TS05_BENCHMARK_AND_TTFT_LATENCY.md) — TTFT, Inter-Token Latency (ITL), and Bandwidth benchmarks.

---

### ⚡ 5. Optimization Cycles (`docs/optimization/`)
Performance playbooks and closed-loop engineering optimization cycles:
* [**`OPT01_SPECULATIVE_DECODING_TUNING.md`**](file:///docs/optimization/OPT01_SPECULATIVE_DECODING_TUNING.md) — Tuning $K$-draft tokens, tree speculative decoding, acceptance rates.
* [**`OPT02_FP8_INT4_ACTIVATION_COMPRESSION.md`**](file:///docs/optimization/OPT02_FP8_INT4_ACTIVATION_COMPRESSION.md) — Zero-loss activation quantization and streaming compression.
* [**`OPT03_PAGED_KV_AND_FLASHINFER.md`**](file:///docs/optimization/OPT03_PAGED_KV_AND_FLASHINFER.md) — Paged KV cache memory pooling and FlashInfer kernel acceleration.
* [**`OPT04_GEO_POD_LATENCY_OPTIMIZATION.md`**](file:///docs/optimization/OPT04_GEO_POD_LATENCY_OPTIMIZATION.md) — DHT proximity routing and ping minimization.
* [**`OPT05_CLOSED_LOOP_LIFECYCLE_ENGINE.md`**](file:///docs/optimization/OPT05_CLOSED_LOOP_LIFECYCLE_ENGINE.md) — Continuous self-optimizing feedback loop across the full stack.
