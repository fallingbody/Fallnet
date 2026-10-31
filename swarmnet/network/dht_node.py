import asyncio
from kademlia.network import Server
import socket
from typing import Optional, Tuple

def get_local_ip() -> str:
    """Gets the local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class SwarmDHTNode:
    """
    Wraps Kademlia DHT for Swarm Node Discovery.
    Provides methods to announce the current miner's HTTP port and query for peers.
    """
    def __init__(self, dht_port: int, bootstrap_host: Optional[str] = None, bootstrap_port: Optional[int] = None):
        self.dht_port = dht_port
        self.bootstrap_host = bootstrap_host
        self.bootstrap_port = bootstrap_port
        self.node = Server()
        self.local_ip = get_local_ip()

    async def start(self):
        """Starts the DHT UDP server and bootstraps if a host is provided."""
        await self.node.listen(self.dht_port)
        if self.bootstrap_host and self.bootstrap_port:
            try:
                print(f"📡 [DHT] Bootstrapping from {self.bootstrap_host}:{self.bootstrap_port}...")
                await self.node.bootstrap([(self.bootstrap_host, self.bootstrap_port)])
                print("✅ [DHT] Bootstrapped successfully.")
            except Exception as e:
                print(f"⚠️ [DHT] Bootstrap failed: {e}")
        else:
            print("🌟 [DHT] Running as Genesis Bootstrap Node.")

    async def announce_miner(self, layer_start: int, layer_end: int, http_port: int, model: str = "SmolLM-135M"):
        """
        Publishes this node's HTTP endpoint for a specific layer shard.
        Key: <model>:layers:<start>-<end>
        Value: http://<local_ip>:<http_port>
        """
        import json
        key = f"{model}:layers:{layer_start}-{layer_end}"
        value = f"http://{self.local_ip}:{http_port}"
        print(f"📣 [DHT] Announcing {key} -> {value}")
        await self.node.set(key, value)

        # Update mesh_topology list
        topology_key = f"{model}:mesh_topology"
        current_state = await self.node.get(topology_key)
        claimed_ranges = []
        if current_state:
            try:
                claimed_ranges = json.loads(current_state)
            except Exception:
                claimed_ranges = []
        
        # Avoid duplicate entries
        if [layer_start, layer_end] not in claimed_ranges:
            claimed_ranges.append([layer_start, layer_end])
            await self.node.set(topology_key, json.dumps(claimed_ranges))

    async def find_peer(self, layer_index: int, model: str = "SmolLM-135M") -> Optional[str]:
        """
        Queries the DHT for the HTTP endpoint of a peer hosting the specific layer_index.
        Dynamically discovers the exact layer range using the mesh_topology.
        """
        import json
        topology_key = f"{model}:mesh_topology"
        
        # 1. Fetch current topology to find the exact range holding layer_index
        current_state = await self.node.get(topology_key)
        target_range = None
        if current_state:
            try:
                claimed_ranges = json.loads(current_state)
                for r in claimed_ranges:
                    if isinstance(r, list) and len(r) == 2:
                        if r[0] <= layer_index <= r[1]:
                            target_range = r
                            break
            except Exception:
                pass
                
        if not target_range:
            if layer_index < 15:
                target_range = [0, 14]
            else:
                target_range = [15, 29]
            
        # 2. Query the exact key for that range
        key = f"{model}:layers:{target_range[0]}-{target_range[1]}"
        print(f"🔍 [DHT] Searching for exact key: {key}...")
        result = await self.node.get(key)
        return result

    async def negotiate_layers(self, capacity: int, total_layers: int = 30, model: str = "SmolLM-135M") -> Tuple[int, int]:
        """
        Dynamically finds an unassigned block of layers up to 'capacity' size.
        Uses a shared DHT key 'mesh_topology' to track claims.
        """
        import json
        topology_key = f"{model}:mesh_topology"
        
        # 1. Fetch current topology
        current_state = await self.node.get(topology_key)
        if current_state:
            try:
                claimed_ranges = json.loads(current_state)
            except Exception:
                claimed_ranges = []
        else:
            claimed_ranges = []
            
        # 2. Find the first gap or append to the end
        # A simple algorithm: just take the highest claimed layer + 1
        highest_layer = -1
        for r in claimed_ranges:
            if isinstance(r, list) and len(r) == 2:
                highest_layer = max(highest_layer, r[1])
                
        start_layer = highest_layer + 1
        
        # If we exceeded total layers, just help out on the first block (overlap)
        if start_layer >= total_layers:
            print("⚠️ [DHT] Mesh is fully covered. Joining as redundant replica for layers 0 to capacity.")
            start_layer = 0
            
        end_layer = min(start_layer + capacity - 1, total_layers - 1)
        
        # 3. Claim the range
        claimed_ranges.append([start_layer, end_layer])
        await self.node.set(topology_key, json.dumps(claimed_ranges))
        
        print(f"🤝 [DHT] Auto-negotiated layers: {start_layer} to {end_layer} (Capacity: {capacity})")
        return start_layer, end_layer

