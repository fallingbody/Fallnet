"""
FALL Global Network & Model Constants
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class NetworkConfig:
    # Single source of truth for the served model. Miners, client and DHT keys must agree.
    MODEL_NAME: str = "HuggingFaceTB/SmolLM-135M-Instruct"

    # Model dimensions (legacy toy defaults; real code reads dims from the HF config)
    DEFAULT_D_MODEL: int = 4096
    DEFAULT_NUM_HEADS: int = 32
    DEFAULT_VOCAB_SIZE: int = 32000
    
    # Speculative Decoding Constants
    DEFAULT_K_DRAFT: int = 5
    MIN_K_DRAFT: int = 1
    MAX_K_DRAFT: int = 8
    
    # Tolerances & Thresholds
    COSINE_SIMILARITY_TOLERANCE: float = 0.9995
    HEARTBEAT_INTERVAL_MS: int = 100
    HEARTBEAT_TIMEOUT_MS: int = 250
    POD_MAX_RTT_MS: float = 12.0
    
    # Trap Prompts
    TRAP_PROMPT_PERCENTAGE: float = 0.03  # 3% of requests
    
    # Economics (Token Costs & Fuel Credits)
    TOKENS_PER_FUEL_CREDIT: int = 1000
    SLASH_PENALTY_CREDITS: int = 50000
    BOUNTY_REWARD_PCT: float = 0.50

CONFIG = NetworkConfig()
