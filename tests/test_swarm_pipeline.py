"""
End-to-End Test for Split-Learning Pipeline & MoE Routing
"""
import pytest
import torch
from swarmnet.client.embedder import LocalSplitClient
from swarmnet.miner.layer_runner import LayerMinerRunner
from swarmnet.moe.moe_router import SparseMoERouter
from swarmnet.protocol.tensor_codec import TensorCodec

def test_full_pipeline_forward_pass():
    d_model = 576  # SmolLM-135M test dimension (divisible by 9 heads)
    vocab_size = 1000
    
    # 1. Client creates local hidden state (Embedding + Layer 1)
    client = LocalSplitClient(vocab_size=vocab_size, d_model=d_model)
    input_ids = torch.tensor([[12, 45, 99, 230]])
    h_1 = client.encode_and_scramble(input_ids)
    
    # 2. Pack to FP8
    packed_h1, _, _ = TensorCodec.quantize_fp16_to_fp8(h_1)
    
    # 3. Miner 1 (Layers 2-4)
    miner_1 = LayerMinerRunner(layer_start=2, layer_end=4, d_model=d_model)
    packed_h4 = miner_1.execute_forward_slice(packed_h1, session_id="test_sess")
    
    # 4. MoE Router (Sparse Top-2 Dispatch)
    h4_unpacked = TensorCodec.dequantize_fp8_to_fp16(packed_h4)
    moe_router = SparseMoERouter(num_experts=8, top_k=2, d_model=d_model)
    moe_out, dispatched = moe_router(h4_unpacked.to(torch.float32))
    assert len(dispatched) > 0, "No experts were dispatched!"
    
    packed_moe, _, _ = TensorCodec.quantize_fp16_to_fp8(moe_out)
    
    # 5. Miner 2 (Layers 5-8)
    miner_2 = LayerMinerRunner(layer_start=5, layer_end=8, d_model=d_model)
    packed_h8 = miner_2.execute_forward_slice(packed_moe, session_id="test_sess")
    
    # 6. Client decodes final token locally
    final_h = TensorCodec.dequantize_fp8_to_fp16(packed_h8).to(torch.float32)
    next_token = client.decode_and_sample(final_h, temperature=0.0)
    
    assert isinstance(next_token, int)
    assert 0 <= next_token < vocab_size
