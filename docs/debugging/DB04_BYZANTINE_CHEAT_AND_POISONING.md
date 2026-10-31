# 🔍 Debug Playbook 04: Byzantine Cheat Detection & Malicious Poisoning

This playbook outlines the diagnostic and investigative workflow when the Trap Verifier flags a dishonest miner, a weight-poisoning attack, or a Sybil cluster.

---

## 🚨 1. Attack Signature Matrix

| Attack Type | Detected Signature | Investigation Method | Automated Countermeasure |
| :--- | :--- | :--- | :--- |
| **Zero-Compute Free Riding** | Miner returns constant zeros or identity tensor in $<1\text{ms}$. | Inject deterministic Trap Prompt vector. | **Instant Slasher Execution:** Burn 100% of node stake; blacklist IP and Peer ID across DHT. |
| **Subtle Weight Poisoning** | Model output deviates on specific trigger words (Backdoor attack). | Compare layer activation against canonical reference weight hash. | **Weight Verification Challenge:** Demand node submit cryptographic Merkle proof of model shard weights. |
| **Sybil Swarm Flooding** | 500 new miners join from same `/24` subnet. | Analyze IP subnet and hardware telemetry fingerprints. | **Subnet Quota Throttling:** Limit maximum active miners per `/24` subnet to 2 nodes. |

---

## 🛠️ 2. Byzantine Triage Flow

```mermaid
flowchart TD
    Alert[Trap Verifier Flags Node: Cosine Similarity < 0.9995] --> DoubleCheck{Run Redundant Verification on 2 Independent Watchdogs}
    
    DoubleCheck -->|Watchdogs Confirm Honest (Hardware Precision Edge Case)| AdjustThreshold[Update Hardware Cosine Calibration Profile]
    DoubleCheck -->|Watchdogs Confirm Cheating / Malicious Output| ExecuteSlash[Execute On-Chain / Ledger Slashing]
    
    ExecuteSlash --> BurnDeposit[Burn Miner Staked Credits]
    ExecuteSlash --> DistributeBounty[Pay 50% Bounty to Reporting Watchdogs]
    ExecuteSlash --> GlobalBroadcast[Broadcast Peer Blacklist to all Kademlia DHT Nodes]
```
