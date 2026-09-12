import pytest
from atomic.config import AtomicConfig

def test_config_defaults_and_presets():
    # Test nano
    nano = AtomicConfig.nano()
    assert nano.d_model == 128
    assert nano.n_layers == 4
    assert nano.n_heads == 4
    assert nano.d_head == 32
    assert nano.vocab_size == 4096

    # Test micro
    micro = AtomicConfig.micro()
    assert micro.d_model == 256
    assert micro.n_layers == 6
    assert micro.n_heads == 8

    # Test reasoning prod
    reasoning = AtomicConfig.reasoning()
    assert reasoning.d_model == 384
    assert reasoning.n_layers == 8
    assert reasoning.n_heads == 8

    # Test base
    base = AtomicConfig.base()
    assert base.d_model == 768
    assert base.n_layers == 12

def test_config_validation():
    # Incompatible d_model and n_heads should raise ValueError
    with pytest.raises(ValueError):
        AtomicConfig(d_model=127, n_heads=4)

def test_config_serialization():
    cfg = AtomicConfig.nano()
    d = cfg.to_dict()
    restored = AtomicConfig.from_dict(d)
    assert restored.d_model == cfg.d_model
    assert restored.n_layers == cfg.n_layers
