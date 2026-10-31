# 🛡️ Agent Spec 05: Trap Verifier & Byzantine Slasher Agent

The **Trap Verifier Agent** is the decentralized security guard of the AirCompute mesh. It enforces **Proof-of-Inference**, detects cheating/lazy miners, prevents Byzantine poisoning, and executes slashing protocols.

---

## 🎯 1. Core Responsibilities & Objectives

1. **Deterministic Trap-Prompt Injection:**
   * Generates mathematical test vectors with pre-computed activation checksums and secretly injects them into ~3% of live traffic streams.
2. **Cryptographic Activation Verification:**
   * Compares miner output activation hashes against ground-truth mathematical tolerances ($\epsilon < 10^{-4}$).
3. **Optimistic Slashing & Penalty Execution:**
   * If a miner returns garbage, zeros, or skips forward passes, slashes their pending reward stake and burns their reputation score.
4. **Bounty Distribution to Honest Watchdogs:**
   * Rewards peers who report and prove Byzantine behavior with a share of the slashed node's deposit.

---

## ⚙️ 2. Trap Injection & Verification Flow

```mermaid
flowchart TD
    UserStream[Live Token Inference Stream] --> TrafficSplitter{Traffic Splitter}
    
    TrafficSplitter -->|97% Real Traffic| NormalPipeline[Standard Pipeline Execution]
    TrafficSplitter -->|3% Synthetic Trap Prompts| TrapInjector[Secret Trap Prompt Injector]
    
    TrapInjector --> MinerNode[Suspect Miner Node]
    MinerNode -->|Returns Activation Output| HashComparator[Hash Comparator & Tolerance Check]
    
    PrecomputedHash[(Precomputed Mathematical Truth DB)] --> HashComparator
    
    HashComparator -->|Hash Matches (Honest Miner)| CreditReward[Award Token Credits & Increase Reputation]
    HashComparator -->|Hash Mismatch (Cheater Detected)| SlashEngine[Trigger Slasher Engine]
    
    SlashEngine --> SlashStake[Slash Unbonded Token Stake]
    SlashEngine --> BlacklistNode[Blacklist Node IP & Peer ID in DHT]
    SlashEngine --> PayBounty[Pay 50% Bounty to Detection Watchdogs]
```

---

## 📥 3. Input / Output Interface

### Inputs
* **Miner Forward Pass Response:** The computed activation tensor $H_{\text{out}}$ and signed hash receipt.
* **Precomputed Truth Table:** Pre-calculated activation values for known model layers.

### Outputs
* **Slashing Transaction / Alert:**
  ```json
  {
    "verdict": "BYZANTINE_CHEAT_DETECTED",
    "cheating_peer_id": "12D3KooWCheater99",
    "layer_tested": [20, 39],
    "expected_hash": "sha256:7f3b89...",
    "received_hash": "sha256:000000...",
    "slashed_amount_tokens": 50000,
    "bounty_recipient": "peer_watchdog_04",
    "timestamp_epoch": 1786291205
  }
  ```

---

## 🛡️ 4. Creative & Critical Edge Case Handling

| Critical Scenario | System Risk | Verifier Autonomous Reaction |
| :--- | :--- | :--- |
| **Float Precision Divergence (CUDA vs Metal vs CPU)** | Different chip architectures produce minor FP16 rounding differences ($\Delta < 0.0001$), triggering false-positive slashing. | **Adaptive Cosine Similarity Threshold:** Instead of strict bit-by-bit exact hashing, uses **Cosine Similarity Metric** ($\text{Sim}(A, B) > 0.9995$) and L2 norm bounds, avoiding hardware rounding false alarms. |
| **Miner Discovers Trap-Prompt Patterns** | Malicious miner builds a cache of known trap prompts to only compute traps and cheat on user prompts. | **Dynamically Seeded Random Projections:** Trap prompts are continuously generated using randomized pseudo-inputs derived from cryptographic block hashes. |
| **Sybil Attack (1 bad actor spawns 1,000 virtual fake miners)** | Swarm gets flooded with unverified nodes. | **Proof-of-Stake / Compute Warm-up:** New nodes must complete 100 successful trap benchmarks before receiving paid user streams. |

---

## 💻 5. Cosine-Tolerant Verification Kernel

```python
import torch
import torch.nn.functional as F

class TrapVerifierAgent:
    def __init__(self, similarity_threshold=0.9995):
        self.similarity_threshold = similarity_threshold
        self.trap_db = {}  # trap_id -> (input_tensor, expected_output_tensor)

    def verify_miner_output(self, trap_id: str, received_tensor: torch.Tensor, peer_id: str) -> bool:
        """Verifies if the miner output matches ground-truth mathematical tolerances"""
        _, expected_tensor = self.trap_db[trap_id]
        
        # 1. Check shape consistency
        if received_tensor.shape != expected_tensor.shape:
            self._execute_slashing(peer_id, "SHAPE_MISMATCH")
            return False
            
        # 2. Compute Cosine Similarity between flattened vectors
        cos_sim = F.cosine_similarity(
            received_tensor.flatten().unsqueeze(0),
            expected_tensor.flatten().unsqueeze(0)
        ).item()
        
        # 3. Verdict
        if cos_sim >= self.similarity_threshold:
            self._record_honest_pass(peer_id)
            return True
        else:
            self._execute_slashing(peer_id, f"ACTIVATION_DRIFT (Sim={cos_sim:.5f} < {self.similarity_threshold})")
            return False

    def _execute_slashing(self, peer_id: str, reason: str):
        print(f"🚨 BYZANTINE CHEAT DETECTED on {peer_id}! Reason: {reason}. Slashing node stake.")
```
