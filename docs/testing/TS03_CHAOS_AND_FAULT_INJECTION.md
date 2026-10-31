# 🧪 Test Suite 03: Chaos Engineering & Fault Injection

This test suite acts as **"Chaos Monkey for AI Swarms"**: it randomly kills nodes, throttles bandwidth, and injects network packet loss during active live inference to prove self-healing resilience.

---

## 🎯 1. Chaos Scenarios Tested

1. **Random Node Termination (SIGKILL):** Abruptly terminates a primary miner at token index #35 of a 100-token stream.
   * *Success Criteria:* Standby twin takes over within $200\text{ms}$; zero missing tokens in user output.
2. **Network Latency Jitter (Packet Delay Injection):** Injects random $150\text{ms}$ artificial latency on 2 of the 8 hops.
   * *Success Criteria:* MoE Router / Coordinator detects latency degradation and re-routes to backup Pod within 2 tokens.
3. **50% Swarm Churn (Mass Dropout):** Terminates 5 out of 10 miners simultaneously.
   * *Success Criteria:* Coordinator successfully compacts remaining layers onto available nodes without throwing uncaught exceptions.

---

## 💻 2. Chaos Injection Harness

```python
import time
import random
import os
import signal

def run_chaos_dropout_test(miner_process_pids: list, active_stream_handle):
    """Randomly kills a miner process mid-stream and verifies failover recovery"""
    time.sleep(1.5)  # Let generation start streaming
    
    victim_pid = random.choice(miner_process_pids)
    print(f"🐒 CHAOS MONKEY: Injecting SIGKILL into Miner PID {victim_pid} mid-generation...")
    os.kill(victim_pid, signal.SIGKILL)
    
    # Verify stream finishes successfully
    final_output = active_stream_handle.await_completion(timeout_sec=10.0)
    assert final_output.is_complete and not final_output.has_errors, "Stream failed during node dropout!"
```
