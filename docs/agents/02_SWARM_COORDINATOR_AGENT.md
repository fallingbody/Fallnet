# 🌐 Agent Spec 02: Swarm Coordinator & Pod Topology Agent

The **Swarm Coordinator Agent** acts as the decentralized traffic control center of the AirCompute mesh. It manages the **Kademlia Distributed Hash Table (DHT)**, groups peer nodes into ultra-low-latency **Geo-Pods**, and balances inference workloads across the network.

---

## 🎯 1. Core Responsibilities & Objectives

1. **Peer Discovery & DHT Topology Management:**
   * Tracks active peer nodes, their available VRAM/RAM, compute capability (FLOPS), and geographic coordinates.
   * Maps model architectures (e.g. Llama-3.3-70B: 80 layers, or DeepSeek-V3: 256 MoE experts) into optimal shard slices.
2. **Geo-Pod Formation (<10ms Latency Clusters):**
   * Clusters miners located in close geographic proximity (e.g., Frankfurt/London, Tokyo/Seoul, Ohio/Virginia) into execution Pods to eliminate inter-hop latency.
3. **Primary & Hot-Standby Pairing:**
   * Assigns every pipeline layer range a **Primary Miner** and a **Hot-Standby Miner** for instant failover.
4. **Health Check & Heartbeat Monitor:**
   * Polls peer nodes every $100\text{ms}$ via lightweight UDP heartbeats. Instantly triggers self-healing rerouting upon peer dropout.

---

## ⚙️ 2. Pod Topology & Clustering Engine

```mermaid
graph TD
    Coordinator([Swarm Coordinator Agent]) --> DHT[(Kademlia DHT Peer Registry)]
    
    subgraph Geo-Clustering Algorithm
        DHT --> GeoFilter{Ping & Geo Filter}
        GeoFilter -->|Latency < 10ms| PodEU[Geo-Pod EU-Central]
        GeoFilter -->|Latency < 12ms| PodUS[Geo-Pod US-East]
        GeoFilter -->|Latency < 15ms| PodAPAC[Geo-Pod APAC-East]
    end

    subgraph PodEU Topology Layout
        PodEU --> L1[Layers 2-20: Node A (Primary) + Node A2 (Standby)]
        L1 --> L2[Layers 21-40: Node B (Primary) + Node B2 (Standby)]
        L2 --> L3[Layers 41-60: Node C (Primary) + Node C2 (Standby)]
        L3 --> L4[Layers 61-79: Node D (Primary) + Node D2 (Standby)]
    end
```

---

## 📥 3. Input / Output Interface

### Inputs
* **Miner Registration Heartbeat (`JSON` over gRPC/QUIC):**
  ```json
  {
    "peer_id": "12D3KooWStq...xyz",
    "ip_address": "85.214.132.11",
    "nat_type": "FullCone / Symmetric",
    "available_vram_mb": 8192,
    "quant_engine": "llama.cpp_metal",
    "ping_rtt_ms": {"peer_b": 4.2, "peer_c": 6.1},
    "current_load_pct": 12.5
  }
  ```
* **Client Pod Request (`JSON`):** Model requested (e.g. `DeepSeek-V3`), client geo-coordinates / IP.

### Outputs
* **Pod Assignment Descriptor (to Client & Miners):**
  ```json
  {
    "pod_id": "pod_eu_central_409",
    "model": "deepseek-ai/DeepSeek-V3",
    "pipeline_chain": [
      {"layer_range": [2, 20], "primary": "peer_a", "standby": "peer_a2", "endpoint": "quic://85.214.132.11:9001"},
      {"layer_range": [21, 40], "primary": "peer_b", "standby": "peer_b2", "endpoint": "quic://194.25.0.42:9001"},
      {"moe_experts": [1, 64], "node": "peer_c", "endpoint": "quic://142.132.10.8:9001"}
    ],
    "expires_at_epoch": 1786291200
  }
  ```

---

## 🛡️ 4. Creative & Critical Edge Case Handling

| Critical Scenario | System Risk | Coordinator Autonomous Reaction |
| :--- | :--- | :--- |
| **Mass Network Churn (25% of nodes drop simultaneously)** | Pipeline breaks, active inference sessions hang. | **Dynamic Topology Compression:** Automatically shrinks pipeline depth from 8 nodes to 4 higher-capacity nodes; loads 4-bit INT4 shards on remaining nodes within $400\text{ms}$. |
| **Symmetric NAT / Strict Firewall Blocking Direct P2P** | Direct QUIC/UDP stream cannot penetrate router. | **Auto-Relay Routing:** Dispatches a high-bandwidth UPnP / relay peer to bridge connections using WebRTC TURN relays. |
| **Sudden Traffic Spike (Surge Demand)** | All Geo-Pods at 100% capacity. | **Surge Incentive Broadcast:** Broadcasts higher token reward multipliers to idle nodes on the DHT, incentivizing sleeping devices to wake up and seed compute. |

---

## 💻 5. Topology Optimizer Algorithm

```python
from typing import List, Dict

class SwarmCoordinatorAgent:
    def __init__(self, target_max_rtt_ms=10.0):
        self.dht_peers = {}
        self.active_pods = {}
        self.target_max_rtt_ms = target_max_rtt_ms

    def form_geo_pod(self, model_name: str, region: str) -> Dict:
        """Finds closest peer nodes with adequate memory and forms an execution Pod"""
        candidates = [p for p in self.dht_peers.values() if p['region'] == region and p['is_alive']]
        
        # Sort candidates by lowest mutual latency and highest available VRAM
        candidates.sort(key=lambda x: (x['load_pct'], x['avg_rtt_ms']))
        
        if len(candidates) < 4:
            # Fallback: expand search to adjacent geographic regions
            candidates = self._expand_geo_search(region)
            
        pod_layout = self._assign_model_shards(model_name, candidates)
        return pod_layout

    def handle_peer_heartbeat_failure(self, dropped_peer_id: str):
        """Instant failover: Promote Standby Node to Primary in active Pod"""
        for pod_id, pod in self.active_pods.items():
            for shard in pod['pipeline_chain']:
                if shard['primary'] == dropped_peer_id:
                    print(f"🚨 Failover Alert: Promoting standby {shard['standby']} to primary in {pod_id}")
                    shard['primary'] = shard['standby']
                    shard['standby'] = self._find_replacement_standby(shard['layer_range'])
                    self._broadcast_pod_update(pod_id, pod)
```
