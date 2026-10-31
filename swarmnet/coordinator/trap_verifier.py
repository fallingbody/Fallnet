"""
Deterministic Trap Prompt Injection & Byzantine Slasher Engine
"""
import torch
import random
from typing import Dict, Tuple, Optional
from swarmnet.config import CONFIG
from swarmnet.protocol.tensor_codec import TensorCodec

class TrapVerifierAgent:
    def __init__(self, tolerance: float = CONFIG.COSINE_SIMILARITY_TOLERANCE):
        self.tolerance = tolerance
        # In-memory Trap DB: trap_id -> (input_tensor_packed, reference_ground_truth_output)
        self.trap_db: Dict[str, Tuple[bytes, torch.Tensor]] = {}
        self.slashed_peers: Dict[str, str] = {}  # peer_id -> reason

    def create_trap_challenge(self, trap_id: str, d_model: int = 4096) -> Tuple[bytes, torch.Tensor]:
        """Generates a synthetic deterministic challenge with ground-truth forward math"""
        torch.manual_seed(hash(trap_id) % (2**31 - 1))
        h_input = torch.randn(1, 1, d_model, dtype=torch.float16)
        
        # Ground truth simulated mathematical reference
        ref_output = h_input * 1.05 + 0.02
        packed_input, _, _ = TensorCodec.quantize_fp16_to_fp8(h_input)
        
        self.trap_db[trap_id] = (packed_input, ref_output)
        return packed_input, ref_output

    def verify_miner_output(self, trap_id: str, miner_output_packed: bytes, peer_id: str) -> bool:
        """
        Dequantizes miner output and compares cosine similarity against ground truth reference.
        Slashes peer if cheating is detected.
        """
        if trap_id not in self.trap_db:
            return True  # Normal non-trap request
            
        _, expected_ref = self.trap_db[trap_id]
        miner_output = TensorCodec.dequantize_fp8_to_fp16(miner_output_packed)
        
        sim = TensorCodec.compute_cosine_similarity(miner_output, expected_ref)
        
        if sim < self.tolerance:
            # CHEATING DETECTED: Trigger Slasher
            self.slash_miner(peer_id, reason=f"Byzantine failure: Cosine Sim {sim:.5f} < {self.tolerance}")
            return False
            
        return True

    def slash_miner(self, peer_id: str, reason: str):
        self.slashed_peers[peer_id] = reason

    def is_peer_slashed(self, peer_id: str) -> bool:
        return peer_id in self.slashed_peers
