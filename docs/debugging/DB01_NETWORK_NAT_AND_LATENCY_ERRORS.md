# 🔍 Debug Playbook 01: Network, NAT Traversal & Latency Spikes

This playbook details how to diagnose and resolve P2P connection failures, STUN/TURN timeouts, and tail-latency bottlenecks across the mesh.

---

## 🚨 1. Error Symptom Matrix

| Error Code | Error Signature | Root Cause | Automated Resolution |
| :--- | :--- | :--- | :--- |
| `ERR_NAT_HOLEPUNCH_TIMEOUT` | `libp2p: dial back failed to 192.168.1.x` | Symmetric NAT router blocking direct UDP hole punch. | **Fallback to WebRTC Relay:** Route stream through nearest UPnP relay node; enable TURN fallback. |
| `ERR_POD_RTT_SPIKE` | `Pod ping exceeded threshold: 142ms > 15ms` | Wi-Fi congestion or ISP routing change on a miner node. | **Pod Re-Clustering:** Evict lagging node from Pod; recruit lower-ping peer from local DHT subnet. |
| `ERR_QUIC_PACKET_DROP` | `CRC32 checksum error on incoming FP8 tensor` | Network buffer packet drop over UDP. | **Fast Selective ARQ:** Retransmit only the missing tensor chunks with dynamic Congestion Window downscaling. |

---

## 🛠️ 2. Autonomous Diagnostic Decision Tree

```mermaid
flowchart TD
    Detect[Symptom: Inbound Tensor Stream Timed Out > 200ms] --> PingCheck{Ping Target Peer}
    
    PingCheck -->|Ping > 100ms| ReRoute[Trigger Geo-Pod Eviction & Standby Promotion]
    PingCheck -->|Ping < 10ms (Peer is reachable)| PortCheck{Is QUIC Port Open?}
    
    PortCheck -->|Port Closed / Blocked| RelayFallback[Switch Transport to STUN / TURN WebRTC Relay]
    PortCheck -->|Port Open| BufferCheck{Is Miner Process Blocked?}
    
    BufferCheck -->|Process High Load| ShedLoad[Shed 50% Stream Load to Hot-Standby]
    BufferCheck -->|Process Dead| KillAndRestart[Trigger Local Miner Worker Auto-Restart]
```

---

## 💻 3. Diagnostic Command & Verification

```bash
# Verify direct P2P NAT punchability on local peer
python3 -m swarmnet.diagnostics.p2p_ping --peer-id "12D3KooW..." --protocol quic --timeout 3.0
```
