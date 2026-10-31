"""
AirCompute Network Diagnostic Utility: P2P Ping & NAT Connectivity Tool
"""
import argparse
import sys
import os
import time

# Ensure package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from swarmnet.network.stun_turn import STUNTURNService, NATType

def ping_peer(peer_id: str, protocol: str = "quic", timeout: float = 3.0):
    print(f"📡 Probing P2P Link to Peer: '{peer_id}' [Protocol: {protocol.upper()}]...")
    service = STUNTURNService()
    
    # 1. Discover local NAT characteristics
    nat_type, pub_ip, pub_port = service.discover_nat_type("192.168.29.200", 9000)
    print(f"  • Local NAT Classification: {nat_type.value}")
    print(f"  • Server-Reflexive Public Endpoint: {pub_ip}:{pub_port}")
    
    # 2. Gather ICE candidates
    local_cands = service.gather_ice_candidates("local_node_01", "192.168.29.200", 9000)
    remote_cands = service.gather_ice_candidates(peer_id, "192.168.29.201", 9001)
    
    # 3. Negotiate Route
    route = service.negotiate_p2p_connection(
        peer_a_id="local_node_01",
        peer_a_candidates=local_cands,
        peer_b_id=peer_id,
        peer_b_candidates=remote_cands
    )
    
    print(f"\n📊 P2P Route Established:")
    print(f"  • Connection Type: {'Direct P2P Link' if route.is_direct else 'TURN Relayed Link'}")
    print(f"  • Target Endpoint: {route.selected_candidate.ip}:{route.selected_candidate.port} ({route.selected_candidate.candidate_type})")
    print(f"  • Measured RTT: {route.rtt_ms:.2f} ms")
    print(f"  • Geo-Pod Compatibility: {'✅ COMPLIANT (< 15ms)' if route.rtt_ms <= 15.0 else '⚠️ HIGH LATENCY'}")
    return route

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirCompute P2P Ping Diagnostic")
    parser.add_argument("--peer-id", type=str, default="12D3KooW_test_peer_02", help="Target Peer ID")
    parser.add_argument("--protocol", type=str, default="quic", help="Transport protocol (quic/udp)")
    parser.add_argument("--timeout", type=float, default=3.0, help="Ping timeout in seconds")
    args = parser.parse_args()
    
    ping_peer(args.peer_id, args.protocol, args.timeout)
