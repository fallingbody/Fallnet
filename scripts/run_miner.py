#!/usr/bin/env python3
"""
Physical Miner Node Runner
Starts a real MinerWorkerServer that listens on a port for inbound FP8 tensors.
"""
import sys
import os
import argparse
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarmnet.miner.worker_server import MinerWorkerServer

def main():
    parser = argparse.ArgumentParser(description="AirCompute Layer Miner Node")
    parser.add_argument("--miner-id", type=str, required=True, help="Unique miner ID")
    parser.add_argument("--port", type=int, required=True, help="HTTP Port to listen on")
    parser.add_argument("--dht-port", type=int, required=True, help="UDP Port for Kademlia DHT")
    parser.add_argument("--layers", type=str, default=None, help="Layer range (e.g. 0-14). If omitted, dynamically auto-negotiates based on hardware profile.")
    parser.add_argument("--bootstrap-ip", type=str, default=None, help="Bootstrap node IP for DHT")
    parser.add_argument("--bootstrap-port", type=int, default=None, help="Bootstrap node UDP Port")
    
    from swarmnet.config import CONFIG
    parser.add_argument("--model-name", type=str, default=CONFIG.MODEL_NAME, help="Hugging Face model ID")
    parser.add_argument("--total-layers", type=int, default=30, help="Total layers in model")
    
    args = parser.parse_args()
    
    start_layer, end_layer = None, None
    if args.layers:
        start_layer, end_layer = map(int, args.layers.split("-"))
    
    server = MinerWorkerServer(
        miner_id=args.miner_id,
        port=args.port,
        dht_port=args.dht_port,
        bootstrap_host=args.bootstrap_ip,
        bootstrap_port=args.bootstrap_port,
        layer_start=start_layer,
        layer_end=end_layer,
        model_name=args.model_name,
        total_layers=args.total_layers
    )
    
    if args.layers:
        print(f"🚀 Starting Miner {args.miner_id} on HTTP port {args.port} | DHT UDP {args.dht_port} | Static Layers: {start_layer}-{end_layer}")
    else:
        print(f"🚀 Starting Auto-Miner {args.miner_id} on HTTP port {args.port} | DHT UDP {args.dht_port} | Auto-Negotiation ENABLED")
        
    uvicorn.run(server.app, host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
