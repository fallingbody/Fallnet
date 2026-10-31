import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from swarmnet.miner.layer_runner import LayerMinerRunner
from swarmnet.protocol.tensor_codec import TensorCodec

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
model.eval()

prompt = "What is 2 + 2?"
chat_messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(chat_messages, return_tensors="pt", add_generation_prompt=True)
input_ids = inputs["input_ids"] if hasattr(inputs, "input_ids") or isinstance(inputs, dict) else inputs

runner_m1 = LayerMinerRunner(layer_start=0, layer_end=14)
runner_m2 = LayerMinerRunner(layer_start=15, layer_end=29)

# Method 1: Local PyTorch (debug_pipeline.py)
with torch.no_grad():
    curr_ids = input_ids.clone()
    words_local = []
    for step in range(10):
        embeds = model.model.embed_tokens(curr_ids).to(model.dtype)
        h_out = embeds
        seq_len = h_out.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
        pos_emb = model.model.rotary_emb(h_out, position_ids=position_ids)
        
        for l in model.model.layers[0:15]:
            out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)
            h_out = out[0] if isinstance(out, tuple) else out
        for l in model.model.layers[15:30]:
            out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)
            h_out = out[0] if isinstance(out, tuple) else out
            
        logits = model.lm_head(model.model.norm(h_out))[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        words_local.append(tokenizer.decode(next_id[0]))
        curr_ids = torch.cat([curr_ids, next_id], dim=1)
    print("Method 1 Local:", "".join(words_local))

# Method 2: LayerRunner Engine (in-memory invocation)
with torch.no_grad():
    curr_ids = input_ids.clone()
    words_runner = []
    for step in range(10):
        embeds = model.model.embed_tokens(curr_ids).to(model.dtype)
        
        # Client FP16
        bytes0 = embeds.to(torch.float16).cpu().numpy().tobytes()
        shape = embeds.shape
        
        # M1 execution without FP8
        h_in1 = torch.from_numpy(np.frombuffer(bytes0, dtype=np.float16).copy()).reshape(shape).to(model.dtype)
        seq_len = h_in1.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
        pos_emb = runner_m1.rotary_emb(h_in1, position_ids=position_ids)
        
        h_out = h_in1
        for l in runner_m1.layers:
            out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)
            h_out = out[0] if isinstance(out, tuple) else out
            
        bytes1 = h_out.to(torch.float16).cpu().numpy().tobytes()
        
        # M2 execution without FP8
        h_in2 = torch.from_numpy(np.frombuffer(bytes1, dtype=np.float16).copy()).reshape(shape).to(model.dtype)
        pos_emb2 = runner_m2.rotary_emb(h_in2, position_ids=position_ids)
        
        h_out2 = h_in2
        for l in runner_m2.layers:
            out = l(h_out2, position_embeddings=pos_emb2, position_ids=position_ids)
            h_out2 = out[0] if isinstance(out, tuple) else out
            
        # Head
        h_normed = model.model.norm(h_out2)
        logits = model.lm_head(h_normed)[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        words_runner.append(tokenizer.decode(next_id[0]))
        curr_ids = torch.cat([curr_ids, next_id], dim=1)
    print("Method 2 Runner:", "".join(words_runner))
