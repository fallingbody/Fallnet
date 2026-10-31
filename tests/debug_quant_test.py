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

# Test FP16 Raw vs FP8 Codec
with torch.no_grad():
    embeds = model.model.embed_tokens(input_ids).to(model.dtype)
    h_out1 = embeds
    seq_len = h_out1.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
    cos, sin = rotary_emb(h_out1, position_ids=position_ids)
    pos_emb = (cos.unsqueeze(1), sin.unsqueeze(1))
    
    # Layer 0-14
    for l in layers_0_14:
        h_out1 = l(h_out1, position_ids=position_ids, position_embeddings=pos_emb)[0]
        
    # Option A: FP16 Exact
    h_out2_fp16 = h_out1.clone()
    for l in layers_15_29:
        h_out2_fp16 = l(h_out2_fp16, position_ids=position_ids, position_embeddings=pos_emb)[0]
    logits_fp16 = lm_head(norm(h_out2_fp16))[:, -1, :]
    token_fp16 = torch.argmax(logits_fp16, dim=-1).item()
    print(f"FP16 Token: '{tokenizer.decode([token_fp16])}' (id {token_fp16})")
    
    # Option B: FP8 Quantized between L14 and L15
    bytes_m1, _, _ = TensorCodec.quantize_fp16_to_fp8(h_out1.to(torch.float16))
    h_out2_fp8 = TensorCodec.dequantize_fp8_to_fp16(bytes_m1, device="cpu").to(model.dtype)
    for l in layers_15_29:
        h_out2_fp8 = l(h_out2_fp8, position_ids=position_ids, position_embeddings=pos_emb)[0]
    logits_fp8 = lm_head(norm(h_out2_fp8))[:, -1, :]
    token_fp8 = torch.argmax(logits_fp8, dim=-1).item()
    print(f"FP8 Token:  '{tokenizer.decode([token_fp8])}' (id {token_fp8})")
