#!/usr/bin/env python3
"""
FALL (SwarmNet) - Hugging Face Space Production Entrypoint (Gradio SDK runtime)
Mounts our custom FastAPI & WebSocket UI directly so no Docker billing is needed!
"""
import sys
import os
import subprocess
import time
import gradio as gr

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

from swarmnet.config import CONFIG

# 1. Start Miner 1 (Layers 0 to 14) in background
print("⚡ Starting Swarm Miner 01 (Layers 0-14)...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = ROOT_DIR

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

time.sleep(2)

# 3. Mount our custom FastAPI app inside Gradio
from scripts.run_client_ui import app as fastapi_app

# Gradio mounts our full custom FastAPI suite with WebSocket & Chat Studio
demo = gr.mount_gradio_app(fastapi_app, gr.Blocks(), path="/gradio")

# Export for HF Spaces Gradio runner
app = fastapi_app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🌐 FALL UI & Swarm Genesis listening on http://0.0.0.0:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
