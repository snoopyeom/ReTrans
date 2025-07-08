import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from model.transformer_ae import AnomalyTransformerAE, train_model_with_replay


def test_deterministic_replay():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    dummy = torch.zeros(1, 4, 1)
    model(dummy)
    out1 = model.generate_replay_samples(1, deterministic=True)
    out2 = model.generate_replay_samples(1, deterministic=True)
    assert torch.allclose(out1, out2)


def test_freeze_encoder():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
        freeze_after=1,
    )
    dummy = torch.zeros(1, 4, 1)
    model(dummy)
    model(dummy)
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_decoder_types():
    dummy = torch.zeros(1, 4, 1)
    for dec in ["mlp", "rnn", "attention"]:
        model = AnomalyTransformerAE(
            win_size=4,
            enc_in=1,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=4,
            latent_dim=2,
            decoder_type=dec,
        )
        out, _, _, _ = model(dummy)
        assert out.shape == (1, 4, 1)


def test_generate_replay_sequence():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    dummy = torch.zeros(1, 4, 1)
    for _ in range(3):
        model(dummy)
    seq = model.generate_replay_sequence(deterministic=True)
    assert seq.shape[0] >= 3


def test_z_bank_stores_x():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    dummy = torch.ones(1, 4, 1)
    model(dummy)
    assert torch.equal(model.z_bank[0]["x"], dummy[0])


def test_replay_consistency_loss():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    dummy = torch.ones(1, 4, 1)
    model(dummy)  # populate z_bank
    before = model.decoder(model.z_bank[0]["z"].unsqueeze(0)).detach()
    train_model_with_replay(
        model,
        opt,
        dummy,
        replay_consistency_weight=1.0,
        cpd_penalty=0,
    )
    after = model.decoder(model.z_bank[0]["z"].unsqueeze(0))
    target = model.z_bank[0]["x"].unsqueeze(0)
    assert F.mse_loss(after, target) < F.mse_loss(before, target)
