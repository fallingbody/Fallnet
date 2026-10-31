#!/usr/bin/env python3
"""
AirCompute Full Autonomous Production Swarm Launcher
Spins up Genesis Bootstrap UI Node and Auto-Negotiating Miner Nodes seamlessly.
"""
import asyncio
import subprocess
import sys
import os
import time
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    print("=" * 70)
    print("🚀 AIRCOMPUTE FULL AUTONOMOUS DECENTRALIZED PRODUCT LAUNCHER")
    print("=" * 70)
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "."

    # 1. Start Genesis Node & Client UI
    print("1️⃣  Starting Genesis Bootstrap Node & Client UI on http://0.0.0.0:8000 ...")
    ui_proc = subprocess.Popen(
        [sys.executable, "scripts/run_client_ui.py"],
        env=env
    )
    
    # Wait for Client UI & DHT Genesis to initialize
    time.sleep(4)
    
    # 2. Start Auto-Miner 1 (Laptop / High Capacity)
    print("2️⃣  Starting Auto-Miner 1 (Auto-Negotiates Layers based on RAM/FLOPS)...")
    miner1_proc = subprocess.Popen(
        [
            sys.executable, "scripts/run_miner.py",
            "--miner-id", "production_miner_01",
            "--port", "8001",
            "--dht-port", "8470",
            "--bootstrap-ip", "127.0.0.1",
            "--bootstrap-port", "8468"
        ],
        env=env
    )
    
    time.sleep(3)

    # 3. Start Auto-Miner 2 (Secondary Worker / Auto-Negotiates remaining layers)
    print("3️⃣  Starting Auto-Miner 2 (Auto-Negotiates remaining layers)...")
    miner2_proc = subprocess.Popen(
        [
            sys.executable, "scripts/run_miner.py",
            "--miner-id", "production_miner_02",
            "--port", "8002",
            "--dht-port", "8471",
            "--bootstrap-ip", "127.0.0.1",
            "--bootstrap-port", "8468"
        ],
        env=env
    )

    print("\n✅ Full Decentralized Swarm Mesh is running!")
    print("🌐 Open http://localhost:8000 in your browser to access the Chat UI and Swarm Inspector.")
    print("Press Ctrl+C to stop all nodes.\n")

    try:
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Swarm Mesh nodes...")
        ui_proc.terminate()
        miner1_proc.terminate()
        miner2_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
