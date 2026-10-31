import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
model.eval()

prompt = "What is 2 + 2?"
chat = [{"role": "user", "content": prompt}]
batch = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True)
input_ids = batch["input_ids"] if hasattr(batch, "input_ids") or isinstance(batch, dict) else batch

# 1. Full un-sharded generation
with torch.no_grad():
    gen_ids = model.generate(input_ids, max_new_tokens=25, do_sample=False)
print(f"Full Model output: '{tokenizer.decode(gen_ids[0][input_ids.shape[1]:])}'\n")

# 2. Exact sharded pipeline match test
layers_0_14 = model.model.layers[0:15]
layers_15_29 = model.model.layers[15:30]
rotary_emb = model.model.rotary_emb
norm = model.model.norm
lm_head = model.lm_head

with torch.no_grad():
    curr_ids = input_ids.clone()
    print("--- SHARDED MODEL VERIFICATION ---")
    words = []
    for step in range(25):
        embeds = model.model.embed_tokens(curr_ids).to(model.dtype)
        
        # Miner 1: 0-14
        h_out = embeds
        batch_size, seq_len, _ = h_out.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
        pos_emb = rotary_emb(h_out, position_ids=position_ids)
        
        for l in layers_0_14:
            out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)
            h_out = out[0] if isinstance(out, tuple) else out
            
        # Miner 2: 15-29
        for l in layers_15_29:
            out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)
            h_out = out[0] if isinstance(out, tuple) else out
            
        # Head
        h_normed = norm(h_out)
        next_token_logits = lm_head(h_normed)[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        word = tokenizer.decode(next_token_id[0])
        words.append(word)
        curr_ids = torch.cat([curr_ids, next_token_id], dim=1)
        
    print(f"Sharded Output: '{''.join(words)}'")
