"""
Layer Miner Execution Engine
Runs a contiguous slice of a HuggingFace decoder model's layers.

Correctness contract (enforced by tests/test_pipeline_equivalence.py):
  client embed -> shard_1 -> ... -> shard_n -> client norm+lm_head
  must reproduce HF `generate()` greedy output token-for-token (FP16 wire).

Protocol per session:
  - First call: start_pos=0 with the full prompt's hidden states (prefill).
  - Next calls: start_pos=<tokens already processed>, hidden states of NEW tokens only.
  - start_pos=0 always resets the session's KV cache.
  - Any other start_pos that disagrees with the cached length raises KVDesyncError.
"""
import gc
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from swarmnet.config import CONFIG
from swarmnet.protocol.tensor_codec import FP16, TensorCodec


class KVDesyncError(RuntimeError):
    """Client and miner disagree on how many tokens this session has processed."""


def build_causal_mask(q_len: int, start_pos: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Additive 4D mask [1, 1, q_len, start_pos + q_len]. Query i may attend keys 0..start_pos+i.

    Passed explicitly so correctness does not depend on SDPA's implicit `is_causal`
    fallback (without it, eager attention silently produces garbage).
    """
    kv_len = start_pos + q_len
    q_pos = torch.arange(start_pos, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(k_pos > q_pos, torch.finfo(dtype).min)
    return mask[None, None]


class LayerMinerRunner:
    def __init__(self, layer_start: int, layer_end: int, model_name: str = CONFIG.MODEL_NAME,
                 device: str = "cpu", dtype: torch.dtype = torch.float32, max_sessions: int = 32,
                 attn_implementation: Optional[str] = None):
        config = AutoConfig.from_pretrained(model_name)
        n = config.num_hidden_layers
        if not (0 <= layer_start <= layer_end < n):
            raise ValueError(f"Layer range {layer_start}-{layer_end} invalid for {model_name} ({n} layers: 0-{n - 1})")

        self.layer_start, self.layer_end = layer_start, layer_end
        self.model_name, self.device, self.dtype = model_name, device, dtype
        self.max_sessions = max_sessions

        # TODO(roadmap P1): load only this shard's tensors from safetensors.
        # Today the FULL model is loaded then trimmed -> peak RAM = whole model.
        kwargs = {"dtype": dtype}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.config = hf_model.config
        self.layers = nn.ModuleList(hf_model.model.layers[layer_start:layer_end + 1]).to(device).eval()
        self.rotary_emb = hf_model.model.rotary_emb.to(device)
        del hf_model
        gc.collect()

        self._sessions: "OrderedDict[str, tuple[DynamicCache, int]]" = OrderedDict()

    # ------------------------------------------------------------------ sessions
    def session_length(self, session_id: str) -> int:
        return self._sessions[session_id][1] if session_id in self._sessions else 0

    def release_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _get_cache(self, session_id: str, start_pos: int) -> DynamicCache:
        if start_pos == 0:
            self._sessions.pop(session_id, None)
            cache = DynamicCache(config=self.config)
        else:
            if session_id not in self._sessions:
                raise KVDesyncError(f"Unknown session '{session_id}' at start_pos={start_pos} (evicted or never prefilled)")
            cache, length = self._sessions[session_id]
            if length != start_pos:
                raise KVDesyncError(f"Session '{session_id}': miner has {length} tokens, client sent start_pos={start_pos}")
            self._sessions.move_to_end(session_id)
        return cache

    # ------------------------------------------------------------------ compute
    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, session_id: str, start_pos: int = 0) -> torch.Tensor:
        hidden = hidden.to(self.device, self.dtype)
        _, q_len, _ = hidden.shape
        cache = self._get_cache(session_id, start_pos)

        position_ids = torch.arange(start_pos, start_pos + q_len, device=self.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden, position_ids=position_ids)
        mask = build_causal_mask(q_len, start_pos, self.dtype, self.device)

        for layer in self.layers:
            out = layer(hidden, attention_mask=mask, position_ids=position_ids,
                        past_key_values=cache, use_cache=True, position_embeddings=position_embeddings)
            hidden = out[0] if isinstance(out, tuple) else out

        self._sessions[session_id] = (cache, start_pos + q_len)
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return hidden

    def execute_forward_slice(self, inbound: bytes, session_id: str, start_pos: int = 0,
                              out_mode: int = FP16) -> bytes:
        h_in = TensorCodec.decode_payload(inbound, device=self.device)
        h_out = self.forward(h_in, session_id=session_id, start_pos=start_pos)
        return TensorCodec.encode_payload(h_out, out_mode)
