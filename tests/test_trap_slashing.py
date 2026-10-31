"""
Unit Tests for Trap-Prompt Byzantine Slasher Engine
"""
import pytest
import torch
from swarmnet.coordinator.trap_verifier import TrapVerifierAgent
from swarmnet.protocol.tensor_codec import TensorCodec

def test_trap_verifier_honesty():
    verifier = TrapVerifierAgent()
    trap_id = "test_trap_001"
    
    # Create valid challenge
    packed_input, ground_truth_ref = verifier.create_trap_challenge(trap_id)
    
    # Honest miner returns exact ground truth
    honest_packed, _, _ = TensorCodec.quantize_fp16_to_fp8(ground_truth_ref)
    
    is_valid = verifier.verify_miner_output(trap_id, honest_packed, peer_id="honest_miner")
    assert is_valid is True
    assert verifier.is_peer_slashed("honest_miner") is False

def test_trap_verifier_slashes_lazy_cheater():
    verifier = TrapVerifierAgent()
    trap_id = "test_trap_002"
    
    # Create challenge
    verifier.create_trap_challenge(trap_id)
    
    # Cheating miner returns all zeros
    fake_zero_tensor = torch.zeros(1, 1, 4096, dtype=torch.float16)
    cheater_packed, _, _ = TensorCodec.quantize_fp16_to_fp8(fake_zero_tensor)
    
    is_valid = verifier.verify_miner_output(trap_id, cheater_packed, peer_id="lazy_cheater")
    assert is_valid is False
    assert verifier.is_peer_slashed("lazy_cheater") is True
