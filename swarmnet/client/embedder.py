"""
Client-side split: token embedding (in) and final norm + LM head (out) run locally.

PRIVACY - READ THIS:
  This split does NOT hide the prompt from the first miner. Embedding rows are public
  (they ship with the open model), so a nearest-neighbour lookup recovers every token
  exactly (see tests/test_privacy_reality.py). Running a few layers locally does not fix
  this either; activation-inversion attacks are well documented.
  The only property that holds: the sampled OUTPUT token is chosen locally.
  Real prompt privacy needs trusted hardware, MPC/HE, or running the whole model locally.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarmnet.config import CONFIG


class LocalSplitClient:
    def __init__(self, model_name: str = CONFIG.MODEL_NAME, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.model_name, self.device, self.dtype = model_name, device, dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO(roadmap P1): load only embed/norm/lm_head tensors instead of the full model.
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.config = hf.config
        self.embed_tokens = hf.model.embed_tokens.to(device).eval()
        self.final_norm = hf.model.norm.to(device).eval()
        self.lm_head = hf.lm_head.to(device).eval()  # may be tied to embed_tokens
        del hf

    @torch.no_grad()
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids.to(self.device))

    @torch.no_grad()
    def logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        h = last_hidden.to(self.device, self.dtype)
        return self.lm_head(self.final_norm(h[:, -1, :]))

    @torch.no_grad()
    def sample(self, last_hidden: torch.Tensor, temperature: float = 0.0,
               generator: Optional[torch.Generator] = None) -> int:
        logits = self.logits(last_hidden)
        if temperature <= 0:
            return int(logits.argmax(-1).item())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())
