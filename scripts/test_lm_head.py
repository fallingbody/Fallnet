import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct", local_files_only=True)
h_last = torch.randn(1, 5, 576, dtype=torch.float32)
try:
    logits = model.lm_head(h_last)[:, -1, :]
    print("Logits shape:", logits.shape)
except Exception as e:
    print("ERROR:", e)
