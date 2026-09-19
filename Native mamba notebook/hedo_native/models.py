"""Model definitions. See METHOD.md for the mathematics and the energy theorem.

Proposed model (variant "full_x"):
  * EnergyHEDO  — damped Hamiltonian dynamics H(q,p) = 1/2|p|^2 + V(q) with a nonlinear,
                  L-smooth potential V. Token-wise learned damping. Step size and damping are
                  constrained so H is provably non-increasing per token (Theorem 1). Each token is
                  attenuated by exp(-eta * rho), rho = dissipated fraction of its energy, before the mixer.
  * ExchangeHVSC — per-chunk variational latents with a KL-to-prior bottleneck. In a second
                  mixer pass, each chunk-start token of one modality receives a gated
                  cross-attention message computed from the other modality's chunk latents.
                  Interaction happens only at chunk boundaries; intra-chunk computation is the
                  unchanged native Mamba-2 scan.

Legacy components kept as ablations:
  * HEDO (affine)  — residual token-wise affine map (the earlier implementation).
  * ChunkWiseHVSC  — index-aligned cross-modal KL without information exchange.
"""

import math
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

MAMBA2_MODULE = "mamba_ssm.modules.mamba2"

# CPU tests replace this with a stand-in. train.py / gate.py refuse anything but official Mamba2.
MAMBA2_FACTORY = None


def mamba2_factory():
    if MAMBA2_FACTORY is not None:
        return MAMBA2_FACTORY
    from mamba_ssm import Mamba2
    return Mamba2


@dataclass(frozen=True)
class ModelConfig:
    mixer: str = "mamba2"            # mamba2 | transformer | none
    token_operator: str = "none"     # none | hedo (affine, legacy) | linear | energy
    use_hvsc: bool = False           # legacy index-aligned KL coupling
    hvsc_variational: bool = True
    use_xhvsc: bool = False          # proposed chunk-boundary exchange HVSC
    xhvsc_exchange: bool = True      # False: bottleneck latents only, no cross-modal messages
    xhvsc_heads: int = 4
    prior_beta: float = 1e-3         # weight of KL(q(z|chunk) || N(0, I))
    exchange_weight: float = 1.0     # weight of the pairwise (exchange) InfoNCE
    energy_rank: int = 64
    energy_steps: int = 3
    energy_dt_max: float = 0.5
    adaptive_damping: bool = True
    embed_dim: int = 128
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    n_layers: int = 1
    hvsc_chunk_size: int = 16        # unrelated to Mamba2's internal SSD chunk_size
    d_latent: int = 64
    head_hidden: int = 0
    transformer_ff: int = 256
    transformer_heads: int = 4
    img_len: int = 196
    txt_len: int = 64

    def to_dict(self):
        return asdict(self)

    @property
    def exchanges(self):
        return self.use_xhvsc and self.xhvsc_exchange


PROPOSED = "full_x"
VARIANTS = {
    # legacy 2x2 factorial (affine HEDO, index-KL HVSC)
    "baseline": dict(),
    "hedo": dict(token_operator="hedo"),
    "hvsc": dict(use_hvsc=True),
    "full": dict(token_operator="hedo", use_hvsc=True),
    "full_linear_operator": dict(token_operator="linear", use_hvsc=True),
    "full_hvsc_deterministic": dict(token_operator="hedo", use_hvsc=True, hvsc_variational=False),
    "baseline_param_matched": dict(head_hidden="match:full"),
    "no_mixer_meanpool": dict(mixer="none"),
    "transformer_param_matched": dict(mixer="transformer", transformer_ff="match:baseline"),
    # proposed 2x2 factorial
    "hedo_energy": dict(token_operator="energy"),
    "hvsc_x": dict(use_xhvsc=True),
    "full_x": dict(token_operator="energy", use_xhvsc=True),
    # proposed-model ablations and controls
    "full_x_affine": dict(token_operator="hedo", use_xhvsc=True),
    "full_x_constdamp": dict(token_operator="energy", adaptive_damping=False, use_xhvsc=True),
    "full_x_noexchange": dict(token_operator="energy", use_xhvsc=True, xhvsc_exchange=False),
    "full_x_noprior": dict(token_operator="energy", use_xhvsc=True, prior_beta=0.0),
    "baseline_param_matched_x": dict(head_hidden="match:full_x"),
    "transformer_x": dict(mixer="transformer", token_operator="energy", use_xhvsc=True),
}


# ------------------------------------------------------------------------------
# token operators
# ------------------------------------------------------------------------------

class HEDO(nn.Module):
    """Legacy affine operator: q = W_q x, p = W_p x; q' = q + dt p; p' = c p - dt G q'; out = x + W_o(q' + p')."""

    def __init__(self, d_model=128, dt=0.1, gamma_init=0.1):
        super().__init__()
        self.dt = dt
        self.q_proj = nn.Linear(d_model, d_model)
        self.p_proj = nn.Linear(d_model, d_model)
        self.gamma_raw = nn.Parameter(torch.full((d_model,), math.log(math.expm1(gamma_init))))
        self.grad_v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        q = self.q_proj(x)
        p = self.p_proj(x)
        gamma = F.softplus(self.gamma_raw)
        q_next = q + self.dt * p
        p_next = p * torch.clamp(1.0 - gamma * self.dt, 0.0, 1.0) - self.dt * self.grad_v_proj(q_next)
        return x + self.output_proj(q_next + p_next)


class LinearResidualStack(nn.Module):
    """Control for affine HEDO: same function class and parameter count (4 (d^2 + d) + d)."""

    def __init__(self, d_model=128):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.scale = nn.Parameter(torch.ones(d_model))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.1)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return x + h * self.scale


def _inv_softplus(y):
    return math.log(math.expm1(y))


def logcosh(z):
    return z + F.softplus(-2.0 * z) - math.log(2.0)


class EnergyHEDO(nn.Module):
    """Hamiltonian-inspired energy dissipation operator (Theorem 1 in METHOD.md).

    Per token:  q_0 = x,  p_0 = W_p x
      V(q)    = sum_r w_r logcosh(u_r^T q + b_r),  w_r = softplus(omega_r) >= 0
      grad V  = U^T (w * tanh(U q + b)),  Hessian <= L_hat I,  L_hat = max_r w_r * ||U||_2^2
      dt      = min(dt_max * sigmoid(theta), sqrt((1 - eps) / L_hat))
      c_i     = min(exp(-gamma_i dt), sqrt(1 - L_hat dt^2)),  gamma_i = softplus(g(x_i)) (token-wise)
      p_{k+1} = c_i p_k - dt grad V(q_k),   q_{k+1} = q_k + dt p_{k+1}
      dH_i    = H(q_0, p_0) - H(q_K, p_K) >= 0         (Theorem 1; H >= 0 because V >= 0)
      rho_i   = dH_i / (H(q_0, p_0) + 1) in [0, 1)     (scale-free dissipated fraction)
      out_i   = exp(-eta * rho_i) * q_K                 (attenuation factor in (exp(-eta), 1])
    """

    def __init__(self, d_model=128, rank=64, steps=3, dt_max=0.5, adaptive=True, eps=0.05):
        super().__init__()
        self.steps, self.dt_max, self.adaptive, self.eps = steps, dt_max, adaptive, eps
        self.U = nn.Parameter(torch.randn(rank, d_model) / math.sqrt(d_model))
        self.b = nn.Parameter(torch.zeros(rank))
        self.omega = nn.Parameter(torch.full((rank,), _inv_softplus(0.1)))
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.p_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.p_proj.weight, gain=0.1)
        nn.init.zeros_(self.p_proj.bias)
        if adaptive:
            self.damping = nn.Linear(d_model, 1)
            nn.init.zeros_(self.damping.weight)
            nn.init.constant_(self.damping.bias, _inv_softplus(0.5))
        else:
            self.gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))
        self.eta_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
        self.last_stats = {}

    def potential(self, q):
        w = F.softplus(self.omega)
        return (w * logcosh(q @ self.U.T + self.b)).sum(-1)

    def grad_potential(self, q):
        w = F.softplus(self.omega)
        return (w * torch.tanh(q @ self.U.T + self.b)) @ self.U

    def smoothness_bound(self):
        # detached: a numerical upper bound, not a trainable path (avoids SVD gradients)
        with torch.no_grad():
            sigma = torch.linalg.matrix_norm(self.U.float(), ord=2)
            return (F.softplus(self.omega).max() * sigma.square()).clamp(min=1e-8)

    def trajectory(self, x):
        L_hat = self.smoothness_bound()
        dt = torch.minimum(self.dt_max * torch.sigmoid(self.theta), torch.sqrt((1.0 - self.eps) / L_hat))
        c_max = torch.sqrt((1.0 - L_hat * dt.square()).clamp(min=0.0))
        gamma = F.softplus(self.damping(x)) if self.adaptive else F.softplus(self.gamma_raw).expand(*x.shape[:-1], 1)
        c = torch.minimum(torch.exp(-gamma * dt), c_max)
        q, p = x, self.p_proj(x)
        energies = [0.5 * p.square().sum(-1) + self.potential(q)]
        for _ in range(self.steps):
            p = c * p - dt * self.grad_potential(q)
            q = q + dt * p
            energies.append(0.5 * p.square().sum(-1) + self.potential(q))
        # Theorem 1 gives dH >= 0 exactly; the clamp only removes floating-point round-off.
        dissipated = (energies[0] - energies[-1]).clamp(min=0.0)
        fraction = dissipated / (energies[0] + 1.0)
        attenuation = torch.exp(-F.softplus(self.eta_raw) * fraction)
        return {"out": attenuation.unsqueeze(-1) * q, "energies": torch.stack(energies, -1), "gamma": gamma.squeeze(-1),
                "attenuation": attenuation, "dissipated": dissipated, "dissipated_fraction": fraction, "dt": dt, "L_hat": L_hat, "c_max": c_max}

    def forward(self, x):
        t = self.trajectory(x)
        e = t["energies"].detach()
        self.last_stats = {
            "energy_dt": t["dt"].detach(), "energy_L_hat": t["L_hat"], "energy_gamma_mean": t["gamma"].detach().mean(),
            "energy_attenuation_mean": t["attenuation"].detach().mean(),
            "energy_attenuation_min": t["attenuation"].detach().min(),
            "energy_max_increase": (e[..., 1:] - e[..., :-1]).max(),
        }
        return t["out"]


def build_operator(cfg, d):
    kind = cfg.token_operator
    if kind == "none":
        return nn.Identity()
    if kind == "hedo":
        return HEDO(d)
    if kind == "linear":
        return LinearResidualStack(d)
    if kind == "energy":
        return EnergyHEDO(d, cfg.energy_rank, cfg.energy_steps, cfg.energy_dt_max, cfg.adaptive_damping)
    raise ValueError(kind)


# ------------------------------------------------------------------------------
# sequence mixers
# ------------------------------------------------------------------------------

class SequenceMixer(nn.Module):
    def __init__(self, cfg, max_len):
        super().__init__()
        self.kind = cfg.mixer
        d = cfg.embed_dim
        self.layers = nn.ModuleList()
        if self.kind == "mamba2":
            if d % cfg.headdim:
                raise ValueError("embed_dim must be divisible by headdim")
            Mamba2 = mamba2_factory()
            for _ in range(cfg.n_layers):
                self.layers.append(Mamba2(d_model=d, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                                          headdim=cfg.headdim))
        elif self.kind == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, max_len, d))
            nn.init.normal_(self.pos, std=0.02)
            for _ in range(cfg.n_layers):
                self.layers.append(nn.TransformerEncoderLayer(
                    d, cfg.transformer_heads, dim_feedforward=cfg.transformer_ff, dropout=0.0,
                    batch_first=True, norm_first=True))
        elif self.kind != "none":
            raise ValueError(self.kind)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        if self.kind == "mamba2":
            # Right padding + causal scan: valid-token outputs never see padding.
            for layer in self.layers:
                x = layer(x.contiguous())
        elif self.kind == "transformer":
            x = x + self.pos[:, : x.shape[1]]
            pad = None if mask is None else ~mask
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=pad)
        return self.norm(x)


def masked_mean(y, mask):
    if mask is None:
        return y.mean(dim=1)
    w = mask.unsqueeze(-1).to(y.dtype)
    return (y * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


def chunk_boundaries(y, mask, chunk_size):
    """States at the last valid token of each chunk (right-padded masks). Returns (states [B,K,D], chunk_mask [B,K])."""
    B, L, D = y.shape
    n_chunks = (L + chunk_size - 1) // chunk_size
    ends = torch.clamp(torch.arange(n_chunks, device=y.device) * chunk_size + chunk_size - 1, max=L - 1)
    if mask is None:
        idx = ends.unsqueeze(0).expand(B, -1)
        cmask = torch.ones(B, n_chunks, device=y.device)
    else:
        lengths = mask.long().sum(dim=1, keepdim=True)
        idx = torch.minimum(ends.unsqueeze(0), (lengths - 1).clamp(min=0))
        starts = torch.arange(n_chunks, device=y.device).unsqueeze(0) * chunk_size
        cmask = (lengths > starts).float()
    states = torch.gather(y, 1, idx.unsqueeze(-1).expand(B, n_chunks, D))
    return states, cmask


def gaussian_kl_to_prior(mu, logvar, cmask):
    """KL(N(mu, exp(logvar)) || N(0, I)), per-dim mean, masked mean over chunks, batch mean."""
    kl = 0.5 * (mu.float().square() + logvar.float().exp() - 1.0 - logvar.float()).mean(-1)
    return ((kl * cmask).sum(-1) / cmask.sum(-1).clamp(min=1.0)).mean()


# ------------------------------------------------------------------------------
# HVSC (legacy index-aligned coupling)
# ------------------------------------------------------------------------------

class _GaussianHeads(nn.Module):
    def __init__(self, d_state, d_latent, variational, mu_bound=10.0, logvar_min=-4.0, logvar_max=1.5):
        super().__init__()
        self.mu_bound, self.lv_min, self.lv_max = mu_bound, logvar_min, logvar_max
        self.mu = nn.ModuleDict({m: nn.Linear(d_state, d_latent) for m in ("img", "txt")})
        self.logvar = nn.ModuleDict({m: nn.Linear(d_state, d_latent) for m in ("img", "txt")}) if variational else None
        lv_bias = math.log((-1.0 - logvar_min) / (logvar_max + 1.0))  # logvar starts at -1
        for m in ("img", "txt"):
            nn.init.xavier_uniform_(self.mu[m].weight, gain=0.2)
            nn.init.zeros_(self.mu[m].bias)
            if variational:
                nn.init.zeros_(self.logvar[m].weight)
                nn.init.constant_(self.logvar[m].bias, lv_bias)

    def posterior(self, modality, states):
        mu = self.mu_bound * torch.tanh(self.mu[modality](states) / self.mu_bound)
        if self.logvar is None:
            return mu, None
        logvar = self.lv_min + (self.lv_max - self.lv_min) * torch.sigmoid(self.logvar[modality](states))
        return mu, logvar


class ChunkWiseHVSC(_GaussianHeads):
    def __init__(self, d_state=128, d_latent=64, variational=True):
        super().__init__(d_state, d_latent, variational, logvar_min=-1.5, logvar_max=1.5)
        self.variational = variational
        self.proj = nn.ModuleDict({m: nn.Linear(d_latent, d_state) for m in ("img", "txt")})
        for m in ("img", "txt"):
            nn.init.xavier_uniform_(self.proj[m].weight, gain=0.2)
            nn.init.zeros_(self.proj[m].bias)

    def encode(self, modality, states, chunk_mask, sample):
        mu, logvar = self.posterior(modality, states)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if (sample and self.variational) else mu
        w = chunk_mask.unsqueeze(-1).to(z.dtype)
        pooled = (self.proj[modality](z) * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return pooled, mu, logvar

    @staticmethod
    def symmetric_kl(mu_i, lv_i, m_i, mu_t, lv_t, m_t):
        K = min(mu_i.shape[1], mu_t.shape[1])
        mu_i, lv_i, mu_t, lv_t = mu_i[:, :K].float(), lv_i[:, :K].float(), mu_t[:, :K].float(), lv_t[:, :K].float()
        # 1/2 [KL(i||t) + KL(t||i)] = 1/4 [e^(li-lt) + e^(lt-li) - 2 + d^2 (e^-li + e^-lt)]
        # (log-variance terms cancel; variance ratios are formed from log differences, never divisions)
        d2 = (mu_i - mu_t).pow(2)
        diff = lv_i - lv_t
        sym = 0.25 * (torch.exp(diff) + torch.exp(-diff) - 2.0 + d2 * (torch.exp(-lv_i) + torch.exp(-lv_t))).mean(-1)
        joint = m_i[:, :K] * m_t[:, :K]
        per_pair = (sym * joint).sum(-1) / joint.sum(-1).clamp(min=1.0)
        return per_pair.mean()


# ------------------------------------------------------------------------------
# HVSC-X (proposed chunk-boundary exchange with variational bottleneck)
# ------------------------------------------------------------------------------

class ExchangeHVSC(_GaussianHeads):
    """Chunk latents z_k ~ q(z | s_k) (bottleneck, KL to N(0, I)).

    Message to chunk k of the receiver: gated multi-head attention whose query is the receiver's
    pass-1 boundary state of chunk k-1 (a learned query for k = 0) and whose keys/values are the
    sender's chunk latents. The message is added only at the first token of chunk k before the
    second mixer pass. Chunks beyond a caption's length receive nothing.
    """

    def __init__(self, d_state=128, d_latent=64, heads=4, chunk_size=16, exchange=True):
        super().__init__(d_state, d_latent, variational=True)
        self.chunk_size = chunk_size
        self.readout = nn.ModuleDict({m: nn.Linear(d_latent, d_state) for m in ("img", "txt")})
        if exchange:  # the no-exchange ablation keeps only the bottleneck
            self.q0 = nn.ParameterDict({m: nn.Parameter(torch.zeros(d_state)) for m in ("img", "txt")})
            self.attn = nn.ModuleDict({m: nn.MultiheadAttention(d_state, heads, kdim=d_latent, vdim=d_latent,
                                                                batch_first=True) for m in ("img", "txt")})
            # tanh(gate) = 0 at init: the exchange starts as an exact no-op and is learned
            self.gate = nn.ParameterDict({m: nn.Parameter(torch.zeros(())) for m in ("img", "txt")})
        for m in ("img", "txt"):
            nn.init.xavier_uniform_(self.readout[m].weight, gain=0.2)
            nn.init.zeros_(self.readout[m].bias)

    def latents(self, modality, states, cmask, sample):
        mu, logvar = self.posterior(modality, states)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        w = cmask.unsqueeze(-1).to(z.dtype)
        pooled = (self.readout[modality](z) * w).sum(1) / w.sum(1).clamp(min=1.0)
        return z, mu, logvar, pooled

    def inject(self, receiver, x, states, cmask, sender_z, sender_cmask, scale=1.0):
        B, K, D = states.shape
        query = torch.cat([self.q0[receiver].expand(B, 1, D), states[:, :-1]], dim=1)
        msg, _ = self.attn[receiver](query, sender_z, sender_z, key_padding_mask=sender_cmask <= 0, need_weights=False)
        msg = msg * cmask.unsqueeze(-1) * torch.tanh(self.gate[receiver]) * scale
        starts = torch.arange(K, device=x.device) * self.chunk_size
        return x + torch.zeros_like(x).index_copy(1, starts, msg.to(x.dtype))


# ------------------------------------------------------------------------------
# full retrieval model
# ------------------------------------------------------------------------------

def _head(d, hidden):
    if hidden <= 0:
        return nn.Linear(d, d)
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))


class RetrievalModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.use_hvsc and cfg.use_xhvsc:
            raise ValueError("use_hvsc and use_xhvsc are exclusive")
        self.cfg = cfg
        d = cfg.embed_dim
        self.proj = nn.ModuleDict({"img": nn.Linear(768, d), "txt": nn.Linear(768, d)})
        self.operator = nn.ModuleDict({m: build_operator(cfg, d) for m in ("img", "txt")})
        self.mixer = nn.ModuleDict({"img": SequenceMixer(cfg, cfg.img_len), "txt": SequenceMixer(cfg, cfg.txt_len)})
        self.hvsc = ChunkWiseHVSC(d, cfg.d_latent, cfg.hvsc_variational) if cfg.use_hvsc else None
        self.xhvsc = ExchangeHVSC(d, cfg.d_latent, cfg.xhvsc_heads, cfg.hvsc_chunk_size, cfg.xhvsc_exchange) if cfg.use_xhvsc else None
        self.head = nn.ModuleDict({m: _head(d, cfg.head_hidden) for m in ("img", "txt")})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.exchange_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07))) if cfg.exchanges else None

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)

    def exchange_scale(self):
        return self.exchange_logit_scale.exp().clamp(max=100.0)

    # pass 1: independent (dual-encoder) encoding --------------------------------
    def pass1(self, modality, feats, mask, sample):
        h0 = self.proj[modality](feats)
        if h0.requires_grad:
            h0.retain_grad()
        x = self.operator[modality](h0)
        y = self.mixer[modality](x, mask)
        pooled = masked_mean(y, mask)
        aux = {"proj": h0, "x": x}
        if self.hvsc is not None:
            states, cmask = chunk_boundaries(y, mask, self.cfg.hvsc_chunk_size)
            z_var, mu, logvar = self.hvsc.encode(modality, states, cmask, sample)
            pooled = pooled + z_var
            aux.update(mu=mu, logvar=logvar, cmask=cmask)
        if self.xhvsc is not None:
            states, cmask = chunk_boundaries(y, mask, self.cfg.hvsc_chunk_size)
            z, mu, logvar, z_pooled = self.xhvsc.latents(modality, states, cmask, sample)
            pooled = pooled + z_pooled
            aux.update(states=states, cmask=cmask, z=z, mu=mu, logvar=logvar, z_pooled=z_pooled)
        h = self.head[modality](pooled)
        return F.normalize(h, dim=-1), aux, h.norm(dim=-1)

    # pass 2: chunk-boundary exchange conditioned on a partner --------------------
    def pass2(self, modality, aux, mask, partner_aux, message_scale=1.0):
        other = "txt" if modality == "img" else "img"
        x2 = self.xhvsc.inject(modality, aux["x"], aux["states"], aux["cmask"], partner_aux["z"], partner_aux["cmask"],
                               message_scale)
        y2 = self.mixer[modality](x2, mask)
        return F.normalize(self.head[modality](masked_mean(y2, mask) + aux["z_pooled"]), dim=-1)

    def pair_similarity(self, aux_img, aux_txt, txt_mask, message_scale=(1.0, 1.0)):
        """Exchange score for aligned rows: aux_img[r] paired with aux_txt[r]."""
        zi = self.pass2("img", aux_img, None, aux_txt, message_scale[0])
        zt = self.pass2("txt", aux_txt, txt_mask, aux_img, message_scale[1])
        return (zi * zt).sum(-1)

    @torch.no_grad()
    def encode_image(self, feats):
        return self.pass1("img", feats, None, sample=False)[0]

    @torch.no_grad()
    def encode_text(self, feats, mask):
        return self.pass1("txt", feats, mask, sample=False)[0]

    def operator_stats(self):
        stats = {}
        for m in ("img", "txt"):
            for k, v in getattr(self.operator[m], "last_stats", {}).items():
                stats[f"{m}_{k}"] = v
        if self.cfg.exchanges:
            for m in ("img", "txt"):
                stats[f"{m}_exchange_gate"] = torch.tanh(self.xhvsc.gate[m]).detach()
        return stats

    def forward(self, img_feats, txt_feats, txt_mask, pair_index, sample=True):
        """img_feats [U,196,768] unique images; txt_feats [N,64,768]; pair_index [N] caption -> image row."""
        z_img, aux_i, norm_i = self.pass1("img", img_feats, None, sample)
        z_txt, aux_t, norm_t = self.pass1("txt", txt_feats, txt_mask, sample)
        out = {"z_img": z_img, "z_txt": z_txt, "aux_img": aux_i, "aux_txt": aux_t,
               "coupling_kl": img_feats.new_zeros(()), "prior_kl": img_feats.new_zeros(()), "pair_sim": None}
        stats = {"pre_norm_img_mean": norm_i.mean().detach(), "pre_norm_txt_mean": norm_t.mean().detach()}
        if self.hvsc is not None and self.cfg.hvsc_variational:
            mu_i, lv_i, m_i = aux_i["mu"][pair_index], aux_i["logvar"][pair_index], aux_i["cmask"][pair_index]
            out["coupling_kl"] = self.hvsc.symmetric_kl(mu_i, lv_i, m_i, aux_t["mu"], aux_t["logvar"], aux_t["cmask"])
        if self.xhvsc is not None:
            out["prior_kl"] = 0.5 * (gaussian_kl_to_prior(aux_i["mu"], aux_i["logvar"], aux_i["cmask"])
                                     + gaussian_kl_to_prior(aux_t["mu"], aux_t["logvar"], aux_t["cmask"]))
            if self.cfg.xhvsc_exchange:
                U, N = z_img.shape[0], z_txt.shape[0]
                ui = torch.arange(U, device=z_img.device).repeat_interleave(N)
                tn = torch.arange(N, device=z_img.device).repeat(U)
                pi = {k: v[ui] for k, v in aux_i.items() if k != "proj"}
                pt = {k: v[tn] for k, v in aux_t.items() if k != "proj"}
                out["pair_sim"] = self.pair_similarity(pi, pt, txt_mask[tn]).view(U, N)
        stats.update(self.operator_stats())
        out["stats"] = stats
        return out


def multipositive_infonce(z_img, z_txt, pair_index, scale):
    return multipositive_infonce_sim(z_img @ z_txt.T, pair_index, scale)


def multipositive_infonce_sim(sim, pair_index, scale):
    """Symmetric InfoNCE on a U x N similarity: each image has all its captions as positives."""
    sim = sim * scale
    pos = F.one_hot(pair_index, sim.shape[0]).T.float()             # [U, N]
    loss_i2t = -(F.log_softmax(sim, dim=1) * pos / pos.sum(1, keepdim=True).clamp(min=1.0)).sum(1).mean()
    loss_t2i = -(F.log_softmax(sim.T, dim=1) * pos.T).sum(1).mean()  # exactly one positive per caption
    return 0.5 * (loss_i2t + loss_t2i)


def total_loss(model, out, pair_index, coupling_kl_weight):
    """Returns (loss, parts). Loss = InfoNCE(pass 1) + w_x InfoNCE(exchange) + w_kl coupling KL + beta prior KL."""
    cfg = model.cfg
    parts = {"infonce": multipositive_infonce(out["z_img"], out["z_txt"], pair_index, model.scale())}
    loss = parts["infonce"] + coupling_kl_weight * out["coupling_kl"] + cfg.prior_beta * out["prior_kl"]
    if out["pair_sim"] is not None:
        parts["infonce_exchange"] = multipositive_infonce_sim(out["pair_sim"], pair_index, model.exchange_scale())
        loss = loss + cfg.exchange_weight * parts["infonce_exchange"]
    parts.update(coupling_kl=out["coupling_kl"], prior_kl=out["prior_kl"])
    return loss, parts


# ------------------------------------------------------------------------------
# configuration resolution / parameter matching
# ------------------------------------------------------------------------------

def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _solve_width(cfg, field, target):
    """Parameter count is affine in head_hidden / transformer_ff; solve for the closest integer width."""
    p1 = count_trainable(RetrievalModel(replace(cfg, **{field: 64})))
    p2 = count_trainable(RetrievalModel(replace(cfg, **{field: 128})))
    width = max(1, round(64 + (target - p1) * 64 / (p2 - p1)))
    return replace(cfg, **{field: width})


def build_config(variant, **overrides):
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant}; choose from {sorted(VARIANTS)}")
    spec = dict(VARIANTS[variant])
    matches = {k: v.split(":", 1)[1] for k, v in spec.items() if isinstance(v, str) and v.startswith("match:")}
    for k in matches:
        spec.pop(k)
    cfg = ModelConfig(**{**spec, **overrides})
    for field, reference in matches.items():
        target = count_trainable(RetrievalModel(build_config(reference, **overrides)))
        cfg = _solve_width(cfg, field, target)
    return cfg


def assert_native_mamba2(model):
    found = [m for m in model.modules() if type(m).__module__ == MAMBA2_MODULE and type(m).__name__ == "Mamba2"]
    if model.cfg.mixer == "mamba2" and len(found) != 2 * model.cfg.n_layers:
        raise RuntimeError(f"expected {2 * model.cfg.n_layers} native mamba_ssm Mamba2 modules, found {len(found)}")
    return len(found)
