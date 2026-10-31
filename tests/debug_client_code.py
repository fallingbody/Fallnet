import torch
import httpx
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from swarmnet.protocol.tensor_codec import TensorCodec

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
model.eval()

prompt = "What is 2 + 2?"
chat_messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(chat_messages, return_tensors="pt", add_generation_prompt=True)
input_ids = inputs["input_ids"] if hasattr(inputs, "input_ids") or isinstance(inputs, dict) else inputs

async def run():
    global input_ids
    words = []
    with torch.no_grad():
        for step in range(15):
            embeds = model.model.embed_tokens(input_ids.to("cpu"))
            fp8_bytes, _, _ = TensorCodec.quantize_fp16_to_fp8(embeds.to(torch.float16))
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post("http://127.0.0.1:8001/forward", content=fp8_bytes)
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}")
                    break
            
            h_last = TensorCodec.dequantize_fp8_to_fp16(resp.content, device="cpu").to(model.dtype)
            h_normed = model.model.norm(h_last)
            next_token_logits = model.lm_head(h_normed)[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            word = tokenizer.decode(next_token_id[0])
            words.append(word)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
    print("Exact Client Loop Output: ", "".join(words))

if __name__ == "__main__":
    asyncio.run(run())
