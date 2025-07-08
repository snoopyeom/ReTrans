import pytest

# Skip the test if PyTorch is unavailable
torch = pytest.importorskip("torch")

from model.transformer_ae import AnomalyTransformerAE


def test_replay_horizon_pruning():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
        replay_size=100,
        replay_horizon=2,
    )
    dummy = torch.zeros(1, 4, 1)
    for _ in range(5):
        model(dummy)
    # ensure purge is triggered by sampling
    model.generate_replay_samples(1)
    threshold = model.current_step - model.replay_horizon
    assert all(item["step"] > threshold for item in model.z_bank)
