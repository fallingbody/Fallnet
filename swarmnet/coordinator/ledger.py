"""
In-Memory Fuel Credit Ledger & State Channel Settlement Engine
"""
from typing import Dict
from swarmnet.crypto.state_channel import StateChannelVerifier

class FuelCreditLedger:
    def __init__(self):
        self.balances: Dict[str, float] = {}  # user_or_peer_id -> fuel credits
        self.verifier = StateChannelVerifier()

    def get_balance(self, entity_id: str) -> float:
        return self.balances.get(entity_id, 0.0)

    def credit_seeder(self, peer_id: str, tokens_computed: int):
        """Credits an idle seeder for computing forward passes"""
        earned = tokens_computed / 1000.0  # 1.0 credit per 1000 tokens
        self.balances[peer_id] = self.balances.get(peer_id, 0.0) + earned

    def process_voucher(self, voucher: dict) -> bool:
        """Verifies and settles a cryptographic payment voucher"""
        if not self.verifier.verify_voucher(voucher):
            return False
            
        user_pubkey = voucher["user_public_key"]
        tokens = voucher["tokens_consumed"]
        cost = tokens / 1000.0
        
        current_bal = self.balances.get(user_pubkey, 0.0)
        self.balances[user_pubkey] = current_bal - cost
        return True

    def slash_deposit(self, peer_id: str, penalty: float = 50000.0) -> float:
        """Burns staked credits of a malicious peer"""
        current_bal = self.balances.get(peer_id, 0.0)
        burned = min(current_bal, penalty)
        self.balances[peer_id] = current_bal - burned
        return burned
