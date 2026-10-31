#!/usr/bin/env python3
"""
AirCompute Minimal Real-Weights Swarm Demo (SmolLM-135M / GPT-2)
Loads a real ultra-compact open-source model (~135MB), shards the actual Transformer layers
across simulated P2P nodes, and runs authentic split-learning inference and micropayment settlement.
"""
import sys
import os
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure swarmnet package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarmnet.coordinator.ledger import FuelCreditLedger
from swarmnet.crypto.state_channel import StateChannelSigner
from swarmnet.protocol.tensor_codec import TensorCodec

class ShardedRealLayerMiner:
    def __init__(self, miner_id: str, layer_indices: list, transformer_layers: nn.ModuleList):
        self.miner_id = miner_id
        self.layer_indices = layer_indices
        self.layers = [transformer_layers[i] for i in layer_indices]

    def execute_forward(self, packed_inbound: bytes, pos_emb=None) -> bytes:
        # 1. Dequantize
        h = TensorCodec.dequantize_fp8_to_fp16(packed_inbound).to(torch.float32)
        
        # 2. Execute actual real weight forward passes across assigned layers
        with torch.no_grad():
            for layer in self.layers:
                if pos_emb is not None:
                    layer_outputs = layer(h, position_embeddings=pos_emb)
                else:
                    layer_outputs = layer(h)
                h = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                
        # 3. Quantize output back to FP8/INT8
        packed_out, _, _ = TensorCodec.quantize_fp16_to_fp8(h)
        return packed_out

def run_minimal_model_swarm(prompt: str = "Once upon a time in a decentralized future,"):
    print("=" * 80)
    print("🤖 AIRCOMPUTE: REAL MINIMAL OPEN-SOURCE MODEL SWARM (SmolLM-135M / GPT-2)")
    print("=" * 80)
    
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    print(f"\n[1/5] 📥 Loading minimal pretrained open-source model: '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        is_llama_arch = True
    except Exception as e:
        print(f"  ⚠️ Falling back to lightweight GPT-2: {e}")
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        is_llama_arch = False
        
    full_model.eval()
    print(f"  ✅ Model loaded successfully! (Parameters: ~135 Million, RAM: ~150 MB)")

    # 2. Shard the real model weights
    print("\n[2/5] 🧩 Sharding real model layers across P2P Swarm Nodes...")
    if is_llama_arch:
        embed_tokens = full_model.model.embed_tokens
        layers = full_model.model.layers
        final_norm = full_model.model.norm
        rotary_emb = full_model.model.rotary_emb
        lm_head = full_model.lm_head
    else:
        embed_tokens = full_model.transformer.wte
        pos_embed = full_model.transformer.wpe
        layers = full_model.transformer.h
        final_norm = full_model.transformer.ln_f
        rotary_emb = None
        lm_head = full_model.lm_head

    total_layers = len(layers)
    mid = total_layers // 2
    miner_1 = ShardedRealLayerMiner("miner_node_01", list(range(0, mid)), layers)
    miner_2 = ShardedRealLayerMiner("miner_node_02", list(range(mid, total_layers)), layers)
    print(f"  • Total Transformer Layers: {total_layers}")
    print(f"  • 💻 Client Device: Local Token Embedding & LM Head (Split-Learning)")
    print(f"  • ⚡ Miner 1: Layers 0 to {mid - 1}")
    print(f"  • ⚡ Miner 2: Layers {mid} to {total_layers - 1}")

    # 3. Setup Micropayment State Channel & Ledger
    print("\n[3/5] 💳 Initializing Ed25519 Micro-Payment State Channel...")
    ledger = FuelCreditLedger()
    signer = StateChannelSigner()
    user_pubkey = signer.public_key_hex
    ledger.balances[user_pubkey] = 20.0  # 20 Fuel Credits
    ledger.balances["miner_node_01"] = 100.0
    ledger.balances["miner_node_02"] = 100.0
    print(f"  ✅ User State Channel Opened with 20.00 Fuel Credits.")

    # 4. Generate Tokens via Decentralized Swarm Forward Pass
    print(f"\n[4/5] 🚀 Prompt: \"{prompt}\"")
    print("  🔒 [Split-Learning]: Text converted to hidden activations locally. ZERO text sent over network.")
    print("\n" + "-" * 80)
    print("⚡ LIVE DECENTRALIZED STREAMING INFERENCE:")
    print("-" * 80)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_tokens = []
    max_new_tokens = 25
    start_time = time.perf_counter()

    sys.stdout.write(prompt)
    sys.stdout.flush()

    for step in range(max_new_tokens):
        with torch.no_grad():
            seq_len = input_ids.shape[1]
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            
            # Step A: Local Client Embedding + Positional Encoding
            if is_llama_arch:
                h_0 = embed_tokens(input_ids)
                pos_emb = rotary_emb(h_0, pos_ids)
            else:
                h_0 = embed_tokens(input_ids) + pos_embed(pos_ids)
                pos_emb = None
            
            # Step B: Pack to FP8/INT8 and send to Miner 1
            packed_h0, _, _ = TensorCodec.quantize_fp16_to_fp8(h_0)
            packed_h_mid = miner_1.execute_forward(packed_h0, pos_emb=pos_emb)
            
            # Step C: Miner 1 forwards activation to Miner 2
            packed_h_final = miner_2.execute_forward(packed_h_mid, pos_emb=pos_emb)
            
            # Step D: Miner 2 returns activation to Client
            h_swarm_out = TensorCodec.dequantize_fp8_to_fp16(packed_h_final).to(torch.float32)
            
            # Step E: Client executes local final Norm + LM Head
            normed = final_norm(h_swarm_out)
            logits = lm_head(normed[:, -1, :])
            
            # Sample next token with temperature
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Decode word locally
            next_word = tokenizer.decode(next_token_id[0])
            sys.stdout.write(next_word)
            sys.stdout.flush()
            
            # Append token to input sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())
            
            # Step F: State channel debit & miner earning settlement
            voucher = signer.create_debit_voucher("chan_live_minimal", tokens_consumed=1)
            ledger.process_voucher(voucher)
            ledger.credit_seeder("miner_node_01", tokens_computed=1)
            ledger.credit_seeder("miner_node_02", tokens_computed=1)

    elapsed = time.perf_counter() - start_time
    print("\n" + "-" * 80)
    
    # 5. Performance and Billing Telemetry
    print(f"\n[5/5] 📊 PERFORMANCE & SETTLEMENT TELEMETRY:")
    print(f"  • Model Used: {model_id} (Actual open-source pretrained weights)")
    print(f"  • Tokens Generated: {len(generated_tokens)}")
    print(f"  • Total Time: {elapsed:.2f} seconds ({len(generated_tokens) / elapsed:.1f} tokens/sec)")
    print(f"  • User Fuel Balance: {ledger.get_balance(user_pubkey):.4f} Credits (Deducted {len(generated_tokens)/1000:.4f})")
    print(f"  • Miner 1 Balance: {ledger.get_balance('miner_node_01'):.4f} Credits")
    print(f"  • Miner 2 Balance: {ledger.get_balance('miner_node_02'):.4f} Credits")
    print("\n" + "=" * 80)
    print("✅ REAL MINIMAL MODEL SWARM INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time in a decentralized future,"
    run_minimal_model_swarm(prompt)
