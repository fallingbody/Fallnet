import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from swarmnet.protocol.tensor_codec import TensorCodec

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
model.eval()

prompt = "What is 2 + 2?"
chat = [{"role": "user", "content": prompt}]
batch = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True)
input_ids = batch["input_ids"] if hasattr(batch, "input_ids") or isinstance(batch, dict) else batch

layers_0_14 = model.model.layers[0:15]
layers_15_29 = model.model.layers[15:30]
rotary_emb = model.model.rotary_emb
norm = model.model.norm
lm_head = model.lm_head

# 1. Uncompressed FP16 Pipeline
with torch.no_grad():
    curr_ids = input_ids.clone()
    words_fp16 = []
    for step in range(15):
        embeds = model.model.embed_tokens(curr_ids).to(model.dtype)
        h_out = embeds
        seq_len = h_out.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
        pos_emb = rotary_emb(h_out, position_ids=position_ids)
        
        for l in layers_0_14:
            h_out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)[0]
        for l in layers_15_29:
            h_out = l(h_out, position_embeddings=pos_emb, position_ids=position_ids)[0]
            
        logits = lm_head(norm(h_out))[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        words_fp16.append(tokenizer.decode(next_id[0]))
        curr_ids = torch.cat([curr_ids, next_id], dim=1)
    print("FP16 Pipeline:", "".join(words_fp16))

# 2. FP8 Quantized Pipeline (Client->M1 FP8, M1->M2 FP8, M2->Client FP8)
with torch.no_grad():
    curr_ids = input_ids.clone()
    words_fp8 = []
    for step in range(15):
        embeds = model.model.embed_tokens(curr_ids).to(model.dtype)
        # Client -> M1 FP8
        bytes_c_m1, _, _ = TensorCodec.quantize_fp16_to_fp8(embeds.to(torch.float16))
        h_out = TensorCodec.dequantize_fp8_to_fp16(bytes_c_m1).to(model.dtype)
        
        seq_len = h_out.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
        pos_emb = rotary_emb(h_out, position_ids=position_ids)
        
        # M1 (0-14)
        for l in layers_0_14:
            h_out = l(h_out, position_ids=position_ids, position_embeddings=pos_emb)[0]
            
        # M1 -> M2 FP8
        bytes_m1_m2, _, _ = TensorCodec.quantize_fp16_to_fp8(h_out.to(torch.float16))
        h_out = TensorCodec.dequantize_fp8_to_fp16(bytes_m1_m2).to(model.dtype)
        
        # M2 (15-29)
        for l in layers_15_29:
            h_out = l(h_out, position_ids=position_ids, position_embeddings=pos_emb)[0]
            
        # M2 -> Client FP8
        bytes_m2_c, _, _ = TensorCodec.quantize_fp16_to_fp8(h_out.to(torch.float16))
        h_last = TensorCodec.dequantize_fp8_to_fp16(bytes_m2_c).to(model.dtype)
        
        logits = lm_head(norm(h_last))[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        words_fp8.append(tokenizer.decode(next_id[0]))
        curr_ids = torch.cat([curr_ids, next_id], dim=1)
    print("FP8 Pipeline: ", "".join(words_fp8))
