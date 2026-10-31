"""
Unit Tests for Ed25519 State Channel Micropayments & Nonce Verification
"""
import pytest
from swarmnet.crypto.state_channel import StateChannelSigner, StateChannelVerifier
from swarmnet.coordinator.ledger import FuelCreditLedger

def test_state_channel_happy_path():
    signer = StateChannelSigner()
    verifier = StateChannelVerifier()
    
    # Nonce 1: Consume 100 tokens
    voucher_1 = signer.create_debit_voucher("chan_123", tokens_consumed=100)
    assert verifier.verify_voucher(voucher_1) is True
    
    # Nonce 2: Consume 250 tokens
    voucher_2 = signer.create_debit_voucher("chan_123", tokens_consumed=250)
    assert verifier.verify_voucher(voucher_2) is True

def test_state_channel_double_spend_prevention():
    signer = StateChannelSigner()
    verifier = StateChannelVerifier()
    
    voucher_1 = signer.create_debit_voucher("chan_456", tokens_consumed=50)
    assert verifier.verify_voucher(voucher_1) is True
    
    # Replaying voucher_1 MUST fail
    assert verifier.verify_voucher(voucher_1) is False
