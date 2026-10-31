"""
End-to-End Test for Real Split-Learning Pipeline with Real Weights
"""
import pytest
import torch
from swarmnet.client.embedder import LocalSplitClient
from swarmnet.miner.layer_runner import LayerMinerRunner
from swarmnet.protocol.tensor_codec import TensorCodec, FP16

def test_full_pipeline_forward_pass():
    # 1. Client loads real embedding & LM head
    client = LocalSplitClient()
    prompt = "Hello"
    input_ids = client.tokenizer(prompt, return_tensors="pt").input_ids
    
    # 2. Client embeds
    h_0 = client.embed(input_ids)
    assert h_0.shape[-1] == client.config.hidden_size
    packed_h0 = TensorCodec.encode_payload(h_0, mode=FP16)
    
    # 3. Miner 1 (e.g., layers 0 to 14)
    miner_1 = LayerMinerRunner(layer_start=0, layer_end=14)
    packed_h1 = miner_1.execute_forward_slice(packed_h0, session_id="test_sess", start_pos=0, out_mode=FP16)
    
    # 4. Miner 2 (e.g., layers 15 to 29)
    miner_2 = LayerMinerRunner(layer_start=15, layer_end=29)
    packed_h2 = miner_2.execute_forward_slice(packed_h1, session_id="test_sess", start_pos=0, out_mode=FP16)
    
    # 5. Client decodes final token locally
    final_h = TensorCodec.decode_payload(packed_h2)
    next_token = client.sample(final_h, temperature=0.0)
    
    assert isinstance(next_token, int)
    assert 0 <= next_token < client.config.vocab_size
    word = client.tokenizer.decode([next_token])
    assert len(word) > 0
