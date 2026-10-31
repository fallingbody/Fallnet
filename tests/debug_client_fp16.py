import torch
import httpx
import asyncio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
model.eval()

prompt = "What is 2 + 2?"
chat_messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(chat_messages, return_tensors="pt", add_generation_prompt=True)
input_ids = inputs["input_ids"] if hasattr(inputs, "input_ids") or isinstance(inputs, dict) else inputs

def pack_fp16(t: torch.Tensor) -> bytes:
    shape = tuple(t.shape)
    header = f"{len(shape)}|" + ",".join(map(str, shape)) + "|"
    return header.encode() + t.cpu().numpy().astype(np.float16).tobytes()

def unpack_fp16(b: bytes) -> torch.Tensor:
    parts = b.split(b"|", 2)
    shape = tuple(map(int, parts[1].decode().split(",")))
    arr = np.frombuffer(parts[2], dtype=np.float16).copy()
    return torch.from_numpy(arr).reshape(shape)

async def run():
    global input_ids
    words = []
    with torch.no_grad():
        for step in range(15):
            embeds = model.model.embed_tokens(input_ids.to("cpu"))
            fp16_bytes = pack_fp16(embeds.to(torch.float16))
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post("http://127.0.0.1:8001/forward_fp16", content=fp16_bytes)
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}")
                    break
            
            h_last = unpack_fp16(resp.content).to(model.dtype)
            h_normed = model.model.norm(h_last)
            next_token_logits = model.lm_head(h_normed)[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            word = tokenizer.decode(next_token_id[0])
            words.append(word)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
    print("FP16 Network Swarm Loop Output: ", "".join(words))

if __name__ == "__main__":
    asyncio.run(run())
