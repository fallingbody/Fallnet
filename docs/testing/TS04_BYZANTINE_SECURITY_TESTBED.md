# 🧪 Test Suite 04: Byzantine Security & Anti-Cheating Testbed

This test suite simulates adversarial, dishonest, and malicious miners to verify that the **Trap Verifier Agent** reliably detects cheating, triggers slashing, and preserves output integrity.

---

## 🎯 1. Adversarial Test Matrix

1. **Lazy Miner (All Zeros / Fast Return):** Node returns constant $0.0$ tensor in $0.5\text{ms}$.
   * *Expected Result:* Trap prompt triggered within 3 requests; node slashed and blacklisted immediately.
2. **Noise Injector (Random Gaussian Noise):** Node returns random noise $\sim \mathcal{N}(0, 1)$.
   * *Expected Result:* Cosine similarity fails ($<0.1$); node slashed.
3. **Replay Attacker:** Node replays previous activation vectors to avoid forward pass compute.
   * *Expected Result:* Nonce mismatch detected; payment voucher rejected.

---

## 💻 2. Byzantine Simulation Test Script

```python
import pytest
import torch

def test_trap_verifier_slashes_lazy_miner():
    """Verify Trap Verifier catches a miner returning zeroed tensors"""
    verifier = TrapVerifierAgent()
    fake_trap_id = "trap_challenge_001"
    
    # 1. Setup ground truth trap
    expected_output = torch.randn(1, 1, 4096)
    verifier.trap_db[fake_trap_id] = (None, expected_output)
    
    # 2. Simulate Lazy Cheating Miner
    cheater_output = torch.zeros(1, 1, 4096)
    
    # 3. Test verification
    is_honest = verifier.verify_miner_output(fake_trap_id, cheater_output, peer_id="bad_miner_01")
    
    assert is_honest is False, "Trap Verifier failed to detect lazy cheating miner!"
```
