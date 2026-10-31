import pytest
import asyncio
from swarmnet.network.dht_node import SwarmDHTNode

def test_dynamic_auto_negotiation():
    async def _run():
        # 1. Start a Genesis Node (mocking the Client UI)
        genesis = SwarmDHTNode(dht_port=8468)
        await genesis.start()
        
        # 2. Start Miner A (e.g. powerful laptop with high capacity)
        miner_a = SwarmDHTNode(dht_port=8470, bootstrap_host="127.0.0.1", bootstrap_port=8468)
        await miner_a.start()
        
        # Miner A negotiates 15 layers
        start_a, end_a = await miner_a.negotiate_layers(capacity=15, total_layers=30)
        assert start_a == 0
        assert end_a == 14
        
        # 3. Start Miner B (e.g. medium desktop)
        miner_b = SwarmDHTNode(dht_port=8471, bootstrap_host="127.0.0.1", bootstrap_port=8468)
        await miner_b.start()
        
        # Miner B negotiates 10 layers
        start_b, end_b = await miner_b.negotiate_layers(capacity=10, total_layers=30)
        assert start_b == 15
        assert end_b == 24
        
        # 4. Start Miner C (e.g. phone with low capacity)
        miner_c = SwarmDHTNode(dht_port=8472, bootstrap_host="127.0.0.1", bootstrap_port=8468)
        await miner_c.start()
        
        # Miner C negotiates 10 layers, but only 5 are left
        start_c, end_c = await miner_c.negotiate_layers(capacity=10, total_layers=30)
        assert start_c == 25
        assert end_c == 29
        
        # 5. Start Miner D (redundant node joining after full mesh)
        miner_d = SwarmDHTNode(dht_port=8473, bootstrap_host="127.0.0.1", bootstrap_port=8468)
        await miner_d.start()
        
        # Mesh is covered, so it should start over at 0 to act as a redundant replica
        start_d, end_d = await miner_d.negotiate_layers(capacity=10, total_layers=30)
        assert start_d == 0
        assert end_d == 9
        
        # 6. Verify find_peer dynamically resolves target ranges
        await miner_a.announce_miner(start_a, end_a, 8001)
        await miner_b.announce_miner(start_b, end_b, 8002)
        await miner_c.announce_miner(start_c, end_c, 8003)
        
        # Find peer for layer 16 (should be miner_b)
        url = await genesis.find_peer(layer_index=16)
        assert url == f"http://{miner_b.local_ip}:8002"
        
        # Find peer for layer 28 (should be miner_c)
        url2 = await genesis.find_peer(layer_index=28)
        assert url2 == f"http://{miner_c.local_ip}:8003"
        
        genesis.node.stop()
        miner_a.node.stop()
        miner_b.node.stop()
        miner_c.node.stop()
        miner_d.node.stop()

    asyncio.run(_run())
