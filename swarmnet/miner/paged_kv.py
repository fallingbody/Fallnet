"""
Paged KV-Cache Memory Pool for Consumer Swarm Nodes
Eliminates memory fragmentation and enables high-concurrency stream serving.
"""
import torch
from typing import Dict, List, Optional, Tuple

class PagedKVCachePool:
    def __init__(self, num_blocks: int = 256, block_size: int = 16, num_heads: int = 32, head_dim: int = 128, device: str = "cpu"):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Pre-allocated physical memory buffers [num_blocks, block_size, num_heads, head_dim]
        self.k_pool = torch.zeros((num_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
        self.v_pool = torch.zeros((num_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
        
        self.free_blocks = list(range(num_blocks))
        self.session_block_tables: Dict[str, List[int]] = {}  # session_id -> list of block_ids
        self.session_token_counts: Dict[str, int] = {}

    def allocate_blocks(self, session_id: str, num_tokens_needed: int) -> List[int]:
        current_blocks = self.session_block_tables.get(session_id, [])
        current_capacity = len(current_blocks) * self.block_size
        
        tokens_short = num_tokens_needed - current_capacity
        if tokens_short > 0:
            blocks_to_add = (tokens_short + self.block_size - 1) // self.block_size
            for _ in range(blocks_to_add):
                if not self.free_blocks:
                    # LRU Eviction: evict oldest session if pool is full
                    self._evict_oldest_session()
                block_id = self.free_blocks.pop(0)
                current_blocks.append(block_id)
                
            self.session_block_tables[session_id] = current_blocks
            
        return current_blocks

    def update_kv(self, session_id: str, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int):
        seq_len = k_new.shape[1]
        self.allocate_blocks(session_id, start_pos + seq_len)
        block_table = self.session_block_tables[session_id]
        
        for i in range(seq_len):
            token_idx = start_pos + i
            block_idx = token_idx // self.block_size
            slot_idx = token_idx % self.block_size
            physical_block = block_table[block_idx]
            
            self.k_pool[physical_block, slot_idx] = k_new[0, i]
            self.v_pool[physical_block, slot_idx] = v_new[0, i]
            
        self.session_token_counts[session_id] = start_pos + seq_len

    def release_session(self, session_id: str):
        if session_id in self.session_block_tables:
            for block_id in self.session_block_tables[session_id]:
                self.free_blocks.append(block_id)
            del self.session_block_tables[session_id]
            del self.session_token_counts[session_id]

    def _evict_oldest_session(self):
        if self.session_block_tables:
            oldest_session = next(iter(self.session_block_tables))
            self.release_session(oldest_session)
