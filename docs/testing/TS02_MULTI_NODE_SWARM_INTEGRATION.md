# 🧪 Test Suite 02: Multi-Node Virtual Swarm Integration

This test suite spins up a **10-node local virtual swarm** on a single development machine (using multi-processing and loopback QUIC sockets) to validate end-to-end multi-hop streaming inference.

---

## 🎯 1. Test Topology Layout

```mermaid
flowchart LR
    ClientTest[Client Process: Draft Agent] --> Node1[Node 1: Layers 1-8]
    Node1 --> Node2[Node 2: Layers 9-16]
    Node2 --> Node3[Node 3: Layers 17-24]
    Node3 --> Node4[Node 4: Layers 25-32]
    Node4 --> ClientTest
```

---

## 💻 2. Automated Integration Test Runner

```python
import pytest
import subprocess
import time
import requests

def test_10_node_virtual_swarm_pipeline():
    """Spawns 10 background worker processes and tests streaming prompt completion"""
    # 1. Start Coordinator & 10 Layer Miners
    processes = []
    try:
        coord_proc = subprocess.Popen(["python3", "-m", "swarmnet.coordinator.server", "--port", "9000"])
        processes.append(coord_proc)
        time.sleep(1.0)
        
        for i in range(10):
            p = subprocess.Popen([
                "python3", "-m", "swarmnet.miner.worker",
                "--miner-id", f"miner_{i}",
                "--layer-start", str(i * 8),
                "--layer-end", str((i + 1) * 8 - 1),
                "--coordinator-url", "http://localhost:9000"
            ])
            processes.append(p)
            
        time.sleep(2.0)  # Wait for DHT pod formation
        
        # 2. Dispatch Test Prompt from Client
        res = run_client_query("Explain quantum computing in 3 sentences.")
        
        assert len(res) > 20, "Output text generation failed or returned empty."
        assert "quantum" in res.lower(), "Generated text does not match prompt context."
        
    finally:
        for p in processes:
            p.terminate()
```
