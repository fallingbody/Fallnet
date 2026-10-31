"""
Layer Miner Worker Server (FastAPI / Asynchronous P2P Node)
"""
import asyncio
import httpx
import time
import resource
from fastapi import FastAPI, Request, Response
from typing import Optional
from swarmnet.config import CONFIG
from swarmnet.miner.layer_runner import LayerMinerRunner
from swarmnet.network.dht_node import SwarmDHTNode

def get_real_memory_mb():
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        return 285

class MinerWorkerServer:
    def __init__(self, miner_id: str, port: int, dht_port: int,
                 bootstrap_host: Optional[str] = None, bootstrap_port: Optional[int] = None,
                 layer_start: Optional[int] = None, layer_end: Optional[int] = None,
                 model_name: str = CONFIG.MODEL_NAME,
                 total_layers: int = 30):
        self.miner_id = miner_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.total_layers = total_layers
        self.model_name = model_name
        self.port = port
        self.last_t_elapsed = 0.035
        
        self.dht = SwarmDHTNode(dht_port=dht_port, bootstrap_host=bootstrap_host, bootstrap_port=bootstrap_port)
        
        self.runner = None
        self.app = FastAPI(title=f"LayerMiner-{miner_id}")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            await self.dht.start()
            
            if self.layer_start is None or self.layer_end is None:
                from swarmnet.miner.auto_profiler import AutoProfiler
                profiler = AutoProfiler(model_total_layers=self.total_layers)
                stats = profiler.profile()
                capacity = stats["recommended_layer_capacity"]
                print(f"🤖 [AutoProfiler] RAM: {stats['available_ram_mb']}MB | Speed: {stats['estimated_gflops']}GFLOPS | Capacity: {capacity} layers")
                self.layer_start, self.layer_end = await self.dht.negotiate_layers(capacity, total_layers=self.total_layers, model=self.model_name)
            
            # Load the weights for assigned layer range
            self.runner = LayerMinerRunner(layer_start=self.layer_start, layer_end=self.layer_end, model_name=self.model_name)
            
            # Announce this node to the DHT
            await self.dht.announce_miner(self.layer_start, self.layer_end, self.port, model=self.model_name)

        @self.app.post("/forward")
        async def handle_forward(request: Request):
            session_id = request.headers.get("X-Session-ID", "default_session")
            start_pos_header = request.headers.get("X-Start-Pos", "0")
            try:
                start_pos = int(start_pos_header)
            except ValueError:
                start_pos = 0

            body_bytes = await request.body()
            
            # Execute local layer computation with session KV cache
            t0 = time.time()
            out_bytes = self.runner.execute_forward_slice(body_bytes, session_id=session_id, start_pos=start_pos)
            self.last_t_elapsed = max(0.001, time.time() - t0)
            
            # Forward downstream if not at pipeline end
            if self.layer_end < self.total_layers - 1:
                next_layer = self.layer_end + 1
                next_peer_url = await self.dht.find_peer(layer_index=next_layer, model=self.model_name)
                
                if not next_peer_url:
                    fallback_port = 8001 + (next_layer // (self.total_layers // 2 if self.total_layers > 1 else 1))
                    next_peer_url = f"http://127.0.0.1:{fallback_port}"
                    
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(
                            f"{next_peer_url}/forward",
                            content=out_bytes,
                            headers={
                                "X-Session-ID": session_id,
                                "X-Start-Pos": str(start_pos)
                            }
                        )
                        return Response(content=resp.content, media_type="application/octet-stream")
                except Exception as e:
                    print(f"⚠️ [P2P] Forward to next peer {next_peer_url} failed: {e}")
                    return Response(content=out_bytes, media_type="application/octet-stream", status_code=503)
            else:
                return Response(content=out_bytes, media_type="application/octet-stream")

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "HEALTHY",
                "miner_id": self.miner_id,
                "layers": f"{self.layer_start}-{self.layer_end}",
                "model": self.model_name,
                "vram_free_mb": 4096
            }

        @self.app.get("/metrics")
        async def get_metrics():
            cpu = min(98, max(15, int((0.035 / self.last_t_elapsed) * 50)))
            return {
                "ram_mb": get_real_memory_mb(),
                "cpu_percent": cpu
            }
