import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import copy

from .AnomalyTransformer import EncoderLayer, Encoder
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding

from utils.utils import my_kl_loss, filter_short_segments


try:
    import ruptures as rpt
except ImportError:  # ruptures might not be installed
    rpt = None


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class MLPDecoder(nn.Module):
    """Two-layer MLP decoder operating on latent sequences."""

    def __init__(self, latent_dim: int, d_model: int, enc_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_in),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, latent_dim]
        B, L, _ = z.size()
        out = self.net(z.view(B * L, -1))
        return out.view(B, L, -1)


class RNNDecoder(nn.Module):
    """GRU-based decoder for sequential reconstruction."""

    def __init__(self, latent_dim: int, d_model: int, enc_in: int):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, enc_in)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, latent_dim]
        inputs = self.in_proj(z)
        out, _ = self.rnn(inputs)
        return self.out(out)


class AttentionDecoder(nn.Module):
    """Transformer decoder variant operating on latent sequences."""

    def __init__(self, latent_dim: int, d_model: int, enc_in: int,
                 n_heads: int, d_ff: int):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.fc = nn.Linear(latent_dim, d_model)
        self.out = nn.Linear(d_model, enc_in)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, latent_dim]
        tgt = self.fc(z)
        memory = torch.zeros_like(tgt)
        out = self.decoder(tgt, memory)
        return self.out(out)


class ConditionalTransformerDecoder(nn.Module):
    """Transformer decoder with optional conditioning on recent input."""

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        win_size: int,
        enc_in: int,
        n_heads: int,
        d_ff: int,
        cond_len: int = 5,
    ) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.lat_proj = nn.Linear(latent_dim, d_model)
        self.cond_embed = DataEmbedding(enc_in, d_model, dropout=0.0)
        self.out = nn.Linear(d_model, enc_in)
        self.pos_embed = PositionalEmbedding(d_model)
        self.cond_len = cond_len
        self.requires_condition = True

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # z: [B, L, latent_dim]
        tgt = self.lat_proj(z) + self.pos_embed(z)
        if cond is not None:
            memory = self.cond_embed(cond)
            memory = memory + self.pos_embed(memory)
        else:
            memory = torch.zeros(z.size(0), 1, tgt.size(-1), device=z.device)
        out = self.decoder(tgt, memory)
        return self.out(out)



class AnomalyTransformerAE(nn.Module):
    """Anomaly Transformer using a deterministic autoencoder branch."""

    def __init__(self, win_size, enc_in, d_model=512, n_heads=8, e_layers=3,
                 d_ff=512, dropout=0.0, activation='gelu', latent_dim=16,
                 replay_size: int = 1000,
                 replay_horizon: int | None = None,
                 freeze_after: int | None = None,
                 ema_decay: float | None = None,
                 decoder_type: str = 'mlp',
                 cond_len: int = 5,
                 latent_noise_std: float = 0.0):

        super().__init__()
        self.win_size = win_size
        self.enc_in = enc_in

        self.replay_size = replay_size
        self.replay_horizon = replay_horizon
        self.current_step = 0
        self.freeze_after = freeze_after
        self.ema_decay = ema_decay
        self.encoder_frozen = False
        self.latent_noise_std = latent_noise_std


        # Transformer components
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False,
                                         attention_dropout=dropout,
                                         output_attention=True),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # VAE components
        self.fc_latent = nn.Linear(d_model, latent_dim)

        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(latent_dim, d_model, enc_in)
        elif decoder_type == 'rnn':
            self.decoder = RNNDecoder(latent_dim, d_model, enc_in)
        elif decoder_type == 'attention':
            self.decoder = AttentionDecoder(latent_dim, d_model, enc_in, n_heads, d_ff)
        elif decoder_type == 'conditional':
            self.decoder = ConditionalTransformerDecoder(
                latent_dim,
                d_model,
                win_size,
                enc_in,
                n_heads,
                d_ff,
                cond_len=cond_len,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # store tuples of (input, latent_vector, step) for experience replay
        self.z_bank = []

        if self.ema_decay is not None:
            self.encoder_ema = copy.deepcopy(self.encoder)
            for p in self.encoder_ema.parameters():
                p.requires_grad_(False)
        else:
            self.encoder_ema = None

    def _purge_z_bank(self) -> None:
        """Remove stale latent vectors based on ``replay_horizon`` and size."""
        if self.replay_horizon is not None:
            threshold = self.current_step - self.replay_horizon
            self.z_bank = [item for item in self.z_bank if item["step"] > threshold]
        if len(self.z_bank) > self.replay_size:
            self.z_bank = self.z_bank[-self.replay_size:]

    def update_ema(self) -> None:
        """Update EMA weights for the encoder."""
        if self.ema_decay is None or self.encoder_ema is None:
            return
        with torch.no_grad():
            for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
                p_ema.mul_(self.ema_decay)
                p_ema.add_(p * (1.0 - self.ema_decay))

    def maybe_freeze_encoder(self) -> None:
        """Freeze encoder parameters after ``freeze_after`` steps."""
        if (self.freeze_after is not None and
                self.current_step >= self.freeze_after and
                not self.encoder_frozen):
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder_frozen = True

    def compute_attention_discrepancy(self, series, prior):
        total = 0.0
        for u in range(len(prior)):
            p = series[u]
            q = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True)
            total += torch.mean(my_kl_loss(p, q.detach()))
            total += torch.mean(my_kl_loss(q.detach(), p))
        return total / len(prior)

    def forward(self, x, indices: torch.Tensor | None = None):
        """Forward pass returning reconstruction and attention info.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, L, C)``.
        indices : torch.Tensor or None, optional
            Starting index of each window in the original series. When provided
            this value is stored in ``z_bank`` as ``"idx"`` for precise
            alignment.
        """
        enc = self.embedding(x)
        enc, series, prior, _ = self.encoder(enc)
        z = self.fc_latent(enc)
        if self.latent_noise_std > 0 and self.training:
            noise = torch.randn_like(z) * self.latent_noise_std
            z = z + noise
        if getattr(self.decoder, "requires_condition", False):
            cond = x[:, -self.decoder.cond_len :]
            recon = self.decoder(z, cond)
        else:
            recon = self.decoder(z)

        # advance time step and store (input, latent) pairs for later replay
        self.current_step += 1
        for i, (x_i, vec) in enumerate(zip(x, z)):
            entry = {
                "x": x_i.detach().cpu(),
                "z": vec.detach().cpu(),
                "step": self.current_step,
                "usage": 0,
            }
            if indices is not None:
                idx_val = int(indices[i])
                if idx_val >= 0:
                    entry["idx"] = idx_val
            self.z_bank.append(entry)
        self._purge_z_bank()

        self.maybe_freeze_encoder()

        return recon, series, prior, z

    def loss_function(self, recon_x, x, weights: torch.Tensor | None = None):
        loss = F.mse_loss(recon_x, x, reduction="none")
        loss = loss.mean(dim=(1, 2))
        if weights is not None:
            weights = weights / weights.mean()
            loss = loss * weights
        return loss.mean()

    def generate_replay_samples(
        self,
        n: int,
        deterministic: bool = False,
        current_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
        """Return replay reconstructions and optional weights."""
        self._purge_z_bank()
        if len(self.z_bank) == 0:
            return None
        device = next(self.parameters()).device
        bank_z = torch.stack([item["z"] for item in self.z_bank]).to(device)
        steps = torch.tensor([item["step"] for item in self.z_bank], device=device, dtype=torch.float32)
        usages = torch.tensor([item.get("usage", 0) for item in self.z_bank], device=device, dtype=torch.float32)
        if current_z is None:
            idx = np.random.choice(len(self.z_bank), size=min(n, len(self.z_bank)), replace=False)
            z = bank_z[idx]
            with torch.no_grad():
                if getattr(self.decoder, "requires_condition", False):
                    cond = torch.stack([self.z_bank[i]["x"] for i in idx]).to(device)[:, -self.decoder.cond_len :]
                    recon = self.decoder(z, cond)
                else:
                    recon = self.decoder(z)
            for i in idx:
                self.z_bank[int(i)]["usage"] += 1
            return recon

        # ``current_z`` has shape (batch, seq_len, latent_dim). We average
        # across both batch and temporal dimensions to obtain a single latent
        # vector representing the current window of data. This ensures the
        # cosine similarity operates on vectors of shape ``(latent_dim,)``.
        ref = current_z.mean(dim=(0, 1))
        sims = F.cosine_similarity(bank_z.mean(dim=1), ref.unsqueeze(0), dim=1)
        decay = 1.0 / (1.0 + (self.current_step - steps))
        penalty = 1.0 / (1.0 + usages)
        sims = sims * decay * penalty
        order = torch.argsort(sims, descending=True)
        top_k = order[: min(len(order), n * 5)]
        cand_z = bank_z[top_k]
        cand_x = torch.stack([self.z_bank[i]["x"] for i in top_k]).to(device)
        with torch.no_grad():
            if getattr(self.decoder, "requires_condition", False):
                cond = cand_x[:, -self.decoder.cond_len :]
                recon = self.decoder(cand_z, cond)
            else:
                recon = self.decoder(cand_z)
            losses = F.mse_loss(recon, cand_x, reduction="none").mean(dim=(1, 2))
        baseline = losses.mean().detach()

        selected: list[int] = []
        selected_weights = []
        for idx_tensor in top_k:
            if len(selected) >= n:
                break
            idx = int(idx_tensor)
            candidate = bank_z[idx].mean(dim=0)
            if selected:
                sims_sel = F.cosine_similarity(
                    candidate.unsqueeze(0), bank_z[torch.tensor(selected, device=device)].mean(dim=1), dim=1
                )
                if sims_sel.max() > 0.95:
                    continue
            selected.append(idx)
            selected_weights.append(losses[(top_k == idx_tensor).nonzero(as_tuple=True)[0]].item())
        if len(selected) < n:
            remaining = [int(i) for i in order if int(i) not in selected]
            for idx in remaining:
                if len(selected) == n:
                    break
                selected.append(idx)
                idx_tensor = torch.tensor(idx, device=device)
                selected_weights.append(losses[(top_k == idx_tensor).nonzero(as_tuple=True)[0]].item() if idx_tensor in top_k else baseline.item())

        z = bank_z[torch.tensor(selected, device=device)]
        stored_x = torch.stack([self.z_bank[i]["x"] for i in selected]).to(device)
        with torch.no_grad():
            if getattr(self.decoder, "requires_condition", False):
                cond = stored_x[:, -self.decoder.cond_len :]
                recon = self.decoder(z, cond)
            else:
                recon = self.decoder(z)
        weights = torch.tensor(selected_weights, device=device)
        weights = weights / (baseline + 1e-6)
        for idx in selected:
            self.z_bank[int(idx)]["usage"] += 1
        return recon, weights

    def generate_replay_sequence(self, deterministic: bool = False):
        """Decode all stored latents in chronological order."""
        self._purge_z_bank()
        if len(self.z_bank) == 0:
            return None
        ordered = sorted(self.z_bank, key=lambda t: t["step"])
        device = next(self.parameters()).device
        z = torch.stack([t["z"] for t in ordered]).to(device)
        with torch.no_grad():
            recon = self.decoder(z)
        return recon


def detect_drift_with_ruptures(window: np.ndarray, pen: int = 20, min_gap: int = 30) -> bool:
    if rpt is None:
        raise ImportError("ruptures is required for drift detection")
    # accept batches in (batch, seq_len, features) form
    if window.ndim == 3:
        window = window.reshape(window.shape[0], -1)
    algo = rpt.Pelt(model="l2").fit(window)
    result = algo.predict(pen=pen)
    result = filter_short_segments(result, min_gap)
    return len(result) > 1


def train_model_with_replay(
    model: AnomalyTransformerAE,
    optimizer: torch.optim.Optimizer,
    current_data: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
    cpd_penalty: int = 20,
    min_gap: int = 30,
    replay_consistency_weight: float = 0.0,
    max_replay_samples: int = 32,
) -> tuple[float, bool]:
    """Train model with replay based on detected concept drift."""
    model.train()
    data = current_data
    idx_all = indices
    weights = torch.ones(len(current_data), device=current_data.device)
    drift_detected = False
    if rpt is not None:
        try:
            drift = detect_drift_with_ruptures(
                current_data.detach().cpu().numpy(),
                pen=cpd_penalty,
                min_gap=min_gap,
            )
        except Exception:
            warnings.warn("Change point detection failed; proceeding without replay")
            drift = False
        if drift:
            drift_detected = True
            with torch.no_grad():
                enc = model.embedding(current_data)
                enc, _, _, _ = model.encoder(enc)
                z_curr = model.fc_latent(enc)
            replay = model.generate_replay_samples(len(current_data), current_z=z_curr)
            if replay is not None:
                replay_samples, replay_weights = replay
                data = torch.cat([current_data, replay_samples], dim=0)
                weights = torch.cat([weights, replay_weights.to(current_data.device)])
                if idx_all is not None:
                    pad = torch.full((len(replay_samples),), -1, dtype=idx_all.dtype, device=idx_all.device)
                    idx_all = torch.cat([idx_all, pad], dim=0)
    else:
        warnings.warn("ruptures not installed; CPD updates will not run")
    recon, _, _, _ = model(data, indices=idx_all)
    loss = model.loss_function(recon, data, weights=weights)
    if replay_consistency_weight > 0 and model.z_bank:
        device = current_data.device
        idx = np.random.choice(
            len(model.z_bank),
            size=min(max_replay_samples, len(model.z_bank)),
            replace=False,
        )
        latents = torch.stack([model.z_bank[i]["z"] for i in idx]).to(device)
        targets = torch.stack([model.z_bank[i]["x"] for i in idx]).to(device)
        recon_bank = model.decoder(latents)
        bank_loss = F.mse_loss(recon_bank, targets)
        loss = loss + replay_consistency_weight * bank_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_ema()
    return loss.item(), drift_detected
