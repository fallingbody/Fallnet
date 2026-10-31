"""
P2P Protocol Messages & Schemas for AirCompute
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PeerRegistration(BaseModel):
    peer_id: str
    host: str
    port: int
    vram_mb: int
    tflops: float
    region: str
    assigned_layer_start: Optional[int] = None
    assigned_layer_end: Optional[int] = None
    is_standby: bool = False
    paired_twin_peer_id: Optional[str] = None

class GeoPodLayout(BaseModel):
    pod_id: str
    region: str
    model_name: str
    pipeline: List[Dict[str, Any]]
    active_expert_nodes: Dict[int, str] = Field(default_factory=dict)

class InferenceRequest(BaseModel):
    request_id: str
    session_id: str
    pod_id: str
    draft_tokens: List[int] = Field(default_factory=list)
    state_channel_voucher: Optional[Dict[str, Any]] = None
    is_trap_prompt: bool = False
    trap_id: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    session_id: str
    accepted_token_count: int
    verified_tokens: List[int]
    is_complete: bool = False
    miner_signatures: List[str] = Field(default_factory=list)

class StateChannelVoucher(BaseModel):
    channel_id: str
    nonce: int
    user_public_key: str
    tokens_consumed: int
    user_signature: str
