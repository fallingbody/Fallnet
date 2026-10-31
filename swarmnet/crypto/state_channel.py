"""
Ed25519 Cryptographic State Channel Micro-Billing Engine
"""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from typing import Tuple, Dict
import json

class StateChannelSigner:
    """Client-side Ed25519 signer for micro-billing debit vouchers"""
    def __init__(self, private_key: Ed25519PrivateKey = None):
        self.private_key = private_key or Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.public_key_hex = self.public_key.public_bytes_raw().hex()
        self.current_nonce = 0

    def create_debit_voucher(self, channel_id: str, tokens_consumed: int) -> Dict[str, any]:
        self.current_nonce += 1
        payload = f"{channel_id}:{self.current_nonce}:{tokens_consumed}:{self.public_key_hex}"
        signature = self.private_key.sign(payload.encode('utf-8'))
        
        return {
            "channel_id": channel_id,
            "nonce": self.current_nonce,
            "tokens_consumed": tokens_consumed,
            "user_public_key": self.public_key_hex,
            "user_signature": signature.hex()
        }

class StateChannelVerifier:
    """Gateway/Coordinator side verifier with monotonic nonce tracking"""
    def __init__(self):
        self.channel_nonces: Dict[str, int] = {}  # channel_id -> last_seen_nonce

    def verify_voucher(self, voucher: Dict[str, any]) -> bool:
        channel_id = voucher["channel_id"]
        nonce = voucher["nonce"]
        tokens_consumed = voucher["tokens_consumed"]
        user_pubkey_hex = voucher["user_public_key"]
        sig_hex = voucher["user_signature"]
        
        # 1. Monotonic Nonce Check (Prevents Replay & Double-Spending)
        last_nonce = self.channel_nonces.get(channel_id, 0)
        if nonce <= last_nonce:
            return False  # Nonce already spent or out of order
            
        # 2. Cryptographic Signature Verification
        try:
            pubkey_bytes = bytes.fromhex(user_pubkey_hex)
            sig_bytes = bytes.fromhex(sig_hex)
            public_key = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
            
            payload = f"{channel_id}:{nonce}:{tokens_consumed}:{user_pubkey_hex}"
            public_key.verify(sig_bytes, payload.encode('utf-8'))
            
            # 3. Update nonce on success
            self.channel_nonces[channel_id] = nonce
            return True
        except (InvalidSignature, ValueError, Exception):
            return False
