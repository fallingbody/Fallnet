#!/usr/bin/env python3
"""
FALL (SwarmNet) - Hugging Face Space Production Entrypoint
Runs on port 7860 (Hugging Face default).
Launches:
1. DHT Genesis Node & Client UI on 0.0.0.0:7860
2. In-process/sub-process Local Swarm Miners (L0-L14 & L15-L29) so visitors get an instant,
   fully functional decentralized pipeline demo out-of-the-box.
"""
import sys
import os
import subprocess
import time
import uvicorn

# Ensure repository root is in python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

from swarmnet.config import CONFIG

def main():
    port = int(os.environ.get("PORT", 7860))
    print("=" * 70)
    print("🍂 FALL: DECENTRALIZED P2P AI INFERENCE MESH")
    print(f"🚀 Initializing on Port {port} for Hugging Face Spaces...")
    print("=" * 70)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = ROOT_DIR

    # 1. Start Miner 1 (Layers 0 to 14) in background
    print("⚡ Starting Swarm Miner 01 (Layers 0-14)...")
    m1_proc = subprocess.Popen(
        [
            sys.executable, "scripts/run_miner.py",
            "--miner-id", "hf_miner_alpha_01",
            "--port", "8001",
            "--dht-port", "8470",
            "--layers", "0-14",
            "--bootstrap-ip", "127.0.0.1",
            "--bootstrap-port", "8468"
        ],
        env=env
    )

    # 2. Start Miner 2 (Layers 15 to 29) in background
    print("⚡ Starting Swarm Miner 02 (Layers 15-29)...")
    m2_proc = subprocess.Popen(
        [
            sys.executable, "scripts/run_miner.py",
            "--miner-id", "hf_miner_beta_02",
            "--port", "8002",
            "--dht-port", "8471",
            "--layers", "15-29",
            "--bootstrap-ip", "127.0.0.1",
            "--bootstrap-port", "8468"
        ],
        env=env
    )

    # Allow background miners to spin up before UI starts
    time.sleep(2)

    # 3. Launch the Client UI & DHT Genesis server on port 7860
    from scripts.run_client_ui import app
    print(f"\n🌐 FALL UI & Swarm Genesis listening on http://0.0.0.0:{port}\n")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    finally:
        print("\n🛑 Terminating background miner processes...")
        m1_proc.terminate()
        m2_proc.terminate()

if __name__ == "__main__":
    main()
