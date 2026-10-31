"""
Unit & Integration Tests for AirCompute NAT Traversal, STUN/TURN & P2P Channels
"""
import pytest
import os
import torch
from swarmnet.network.stun_turn import STUNTURNService, NATType, ICECandidate
from swarmnet.network.p2p_channel import P2PChannel
from swarmnet.protocol.tensor_codec import TensorCodec

def test_stun_nat_discovery():
    service = STUNTURNService()
    nat_type, pub_ip, pub_port = service.discover_nat_type("192.168.1.50", 8080)
    assert nat_type in [NATType.PORT_RESTRICTED, NATType.OPEN_INTERNET]
    assert pub_ip is not None
    assert pub_port > 0

def test_ice_candidate_gathering():
    service = STUNTURNService()
    candidates = service.gather_ice_candidates("peer_alpha_01", "192.168.1.50", 8080)
    types = [c.candidate_type for c in candidates]
    assert "host" in types
    assert "srflx" in types
    assert "relay" in types

def test_ice_symmetric_nat_turn_fallback():
    service = STUNTURNService()
    cand_a = service.gather_ice_candidates("peer_a", "10.0.0.2", 9000, simulated_symmetric=True)
    cand_b = service.gather_ice_candidates("peer_b", "10.0.0.3", 9001, simulated_symmetric=True)
    
    route = service.negotiate_p2p_connection(
        "peer_a", cand_a,
        "peer_b", cand_b,
        peer_a_nat=NATType.SYMMETRIC_NAT,
        peer_b_nat=NATType.SYMMETRIC_NAT
    )
    
    assert route.is_relayed is True
    assert route.is_direct is False
    assert route.selected_candidate.candidate_type == "relay"

def test_p2p_channel_framing_and_reassembly():
    channel = P2PChannel(channel_id="test_chan_01", max_chunk_size=512)
    # Generate large tensor and quantize
    raw_tensor = torch.randn(1, 8, 4096, dtype=torch.float16)
    packed_fp8, scale, original_shape = TensorCodec.quantize_fp16_to_fp8(raw_tensor)
    
    # Frame into chunks
    chunks = channel.frame_packet(packed_fp8)
    assert len(chunks) > 1, "Payload should be split into multiple chunks"
    
    # Receive chunks sequentially
    reconstructed_payload = None
    for chunk in chunks:
        is_done, payload, err = channel.process_incoming_chunk(chunk)
        assert err is None
        if is_done:
            reconstructed_payload = payload
            
    assert reconstructed_payload is not None
    assert reconstructed_payload == packed_fp8
    
    # Dequantize and verify data integrity
    dequant = TensorCodec.dequantize_fp8_to_fp16(reconstructed_payload)
    assert dequant.shape == raw_tensor.shape

def test_p2p_channel_crc32_corruption_detection():
    channel = P2PChannel(channel_id="test_chan_corrupt", max_chunk_size=512)
    payload = b"Hello, decentralized AI swarm payload!"
    chunks = channel.frame_packet(payload)
    
    # Corrupt payload byte in the first chunk
    corrupted_chunk = bytearray(chunks[0])
    corrupted_chunk[-1] ^= 0xFF  # Flip bits in payload
    
    is_done, payload_out, err = channel.process_incoming_chunk(bytes(corrupted_chunk))
    assert is_done is False
    assert err == "ERR_CRC32_CORRUPTION"
