"""HEDO-HVSC native Mamba-2 training runner (numerically audited revision).

Changes relative to the run that diverged (epoch 2 NaN), all documented in the notebook:
  * trainable stack runs in FP32; frozen backbones keep FP16 autocast for feature extraction
  * GradScaler removed (it is only needed for FP16 gradients)
  * HVSC log-variance range [-10, 10] -> [LOGVAR_MIN, LOGVAR_MAX] = [-6, 4] (env-configurable),
    std clamped to [1e-3, 20], variance floor 1e-6, squared mean difference bounded by 1e4,
    KL computed in FP32, finite and non-negative check with diagnostics
  * KL weight 1e-4 -> 1e-3 (HEDO_KL_WEIGHT), learning rate 1e-3 -> 1e-4 (HEDO_LR)
  * bounded logit scale in [1, 100]
  * finite checks after every stage, gradients checked before every optimizer step
  * RoBERTa loaded as RobertaModel without the (unused, randomly initialised) pooler
  * official IMAGENET1K_V1 preprocessing; ViT extraction path verified against vit(x) logits
  * 14 GO/NO-GO gates (gates mode) before any training; training refuses to start without GO
  * best (validation-only) and last checkpoints with provenance; test evaluated once on best
  * results/seed{S}/{config}/ run layout, loud structured failures in run_status.json
Terminology: "Hamiltonian-inspired discrete dissipative coordinate-momentum transformation";
"HVSC aligns modality-specific boundary-state posterior distributions using symmetric KL regularization."
"""

import os
import sys
import copy
import json
import math
import time
import random
import hashlib
import inspect

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torch.utils.data import Dataset, DataLoader, Sampler

from torchvision import transforms
import torchvision.models as tv_models

from transformers import (
    AutoTokenizer,
    AutoModel,
)

from mamba_ssm import Mamba2


# ==============================================================================
# NATIVE DEVICE
# ==============================================================================

DEVICE = torch.device(os.environ.get("HEDO_DEVICE", "cuda"))  # "cpu" only for local debugging

if DEVICE.type == "cuda" and not torch.cuda.is_available():

    raise RuntimeError(
        "CRITICAL: CUDA is unavailable in the native Python environment."
    )


# ==============================================================================
# NATIVE MAMBA-2 HARD GATE
# ==============================================================================

print("=" * 80)
print("NATIVE TRAINING RUNNER")
print("=" * 80)

print("Python       :", sys.executable)
print("Python ver.  :", sys.version.split()[0])
print("Torch        :", torch.__version__)
print("CUDA         :", torch.version.cuda)
print("GPU          :", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu (debug)")

print("Mamba2       :", Mamba2)
print("Mamba2 module:", Mamba2.__module__)
print("Mamba2 source :", inspect.getfile(Mamba2))

if Mamba2.__module__ != "mamba_ssm.modules.mamba2":

    raise RuntimeError(
        "CRITICAL: Native mamba_ssm.modules.mamba2.Mamba2 "
        "was not loaded."
    )


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_all_seeds(seed=42):

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except Exception:

        pass


# ==============================================================================
# FINITE-VALUE MONITOR
# ==============================================================================

class NonFiniteError(FloatingPointError):
    pass


def tensor_stats(t):
    t = t.detach()
    f = t.float()
    finite = torch.isfinite(f)
    ok = f[finite]
    return {
        "shape": tuple(t.shape), "dtype": str(t.dtype), "device": str(t.device),
        "min": float(ok.min()) if ok.numel() else float("nan"),
        "max": float(ok.max()) if ok.numel() else float("nan"),
        "mean": float(ok.mean()) if ok.numel() else float("nan"),
        "std": float(ok.std()) if ok.numel() > 1 else float("nan"),
        "finite": bool(finite.all()), "nan_count": int(torch.isnan(f).sum()),
        "inf_count": int(torch.isinf(f).sum()),
    }


class FiniteMonitor:
    """Records tensors at named stages in execution order; check() costs one GPU sync.

    The first failing entry in execution order is the first non-finite tensor.
    Inactive by default so module-level architecture gates are unaffected.
    """

    def __init__(self):
        self.active = False
        self.items = []

    def add(self, name, t):
        if self.active and t is not None:
            self.items.append((name, t.detach()))
        return t

    def check(self, context, verbose=False):
        items, self.items = self.items, []
        if not items:
            return
        flags = torch.stack([torch.isfinite(t).all() for _, t in items]).cpu().tolist()
        if verbose:
            print(f"[FINITE CHECK] {context}", flush=True)
            for (name, _), ok in zip(items, flags):
                print(f"  {name:<24}: {'PASS' if ok else 'FAIL'}", flush=True)
        bad = [(name, t) for (name, t), ok in zip(items, flags) if not ok]
        if bad:
            print(f"[FINITE CHECK] FAILURE during {context}", flush=True)
            for name, t in bad:
                print(f"  {name}: {tensor_stats(t)}", flush=True)
            exc = NonFiniteError(f"first non-finite tensor during {context}: {bad[0][0]}")
            exc.tensor_name, exc.tensor_stats = bad[0][0], tensor_stats(bad[0][1])
            raise exc


MONITOR = FiniteMonitor()


# ==============================================================================
# HEDO
# ==============================================================================

HEDO_GAMMA_INIT = 0.1


class HEDO(nn.Module):

    """Hamiltonian-inspired discrete dissipative coordinate-momentum transformation.

    q = W_q x, p = W_p x, gamma = sigmoid(g), q' = q + p, p' = p - gamma * q', out = W_o(q' + p').
    No energy conservation, symplecticity or guaranteed dissipation is claimed.
    """

    def __init__(
        self,
        d_model=128,
        tag="HEDO",
    ):

        super().__init__()

        self.tag = tag
        self.last_stats = {}

        self.q_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.p_proj = nn.Linear(
            d_model,
            d_model,
        )

        # sigmoid(logit(0.1)) = 0.1: initial gamma equals the documented value
        self.gamma = nn.Parameter(
            torch.full(
                (d_model,),
                math.log(HEDO_GAMMA_INIT / (1.0 - HEDO_GAMMA_INIT)),
            )
        )

        self.output_proj = nn.Linear(
            d_model,
            d_model,
        )

    def forward(self, x):

        t = self.tag

        q = MONITOR.add(f"{t} q", self.q_proj(x))

        p = MONITOR.add(f"{t} p", self.p_proj(x))

        gamma = MONITOR.add(f"{t} gamma", torch.sigmoid(
            self.gamma
        ).view(
            1,
            1,
            -1,
        ))

        q_new = MONITOR.add(f"{t} q_new", q + p)

        p_new = MONITOR.add(f"{t} p_new", (
            p
            - gamma * q_new
        ))

        h = MONITOR.add(f"{t} h", q_new + p_new)

        if MONITOR.active:
            with torch.no_grad():
                # empirical quadratic energy E = 1/2 (|q|^2 + |p|^2), before and after the update
                e_in = 0.5 * (q.float().pow(2).sum(-1) + p.float().pow(2).sum(-1))
                e_out = 0.5 * (q_new.float().pow(2).sum(-1) + p_new.float().pow(2).sum(-1))
                self.last_stats = {
                    "energy_in_mean": e_in.mean(),
                    "energy_out_mean": e_out.mean(),
                    "energy_ratio_mean": (e_out / e_in.clamp_min(1e-12)).mean(),
                    "frac_tokens_energy_increase": (e_out > e_in).float().mean(),
                    "gamma_min": gamma.min(),
                    "gamma_max": gamma.max(),
                }

        return MONITOR.add(f"{t}", self.output_proj(
            h
        ))


# ==============================================================================
# NATIVE MAMBA-2 SEQUENCE BLOCK
# ==============================================================================

class NativeMamba2SequenceBlock(nn.Module):

    def __init__(
        self,
        d_model=128,
        d_state=64,
        chunk_size=16,
        d_conv=4,
        expand=2,
        headdim=64,
        tag="Mamba",
    ):

        super().__init__()

        self.tag = tag

        if d_model % headdim != 0:

            raise ValueError(
                f"d_model={d_model} must be divisible "
                f"by headdim={headdim}"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        x,
        mask=None,
        return_boundary_states=False,
    ):

        if x.ndim != 3:

            raise ValueError(
                f"Expected [B,L,D], got {tuple(x.shape)}"
            )

        B, L, D = x.shape

        if D != self.d_model:

            raise ValueError(
                f"Expected D={self.d_model}, got D={D}"
            )

        # ----------------------------------------------------------------------
        # MASK
        # ----------------------------------------------------------------------

        if mask is None:

            mask_bool = None
            x_in = x

        else:

            if tuple(mask.shape) != (B, L):

                raise ValueError(
                    f"Mask shape {tuple(mask.shape)} "
                    f"does not match {(B, L)}"
                )

            mask_bool = (
                mask
                .to(device=x.device)
                .bool()
            )

            x_in = (
                x
                *
                mask_bool.unsqueeze(-1)
                .to(dtype=x.dtype)
            )

        # ----------------------------------------------------------------------
        # REAL NATIVE MAMBA-2 COMPUTATION
        # ----------------------------------------------------------------------

        y = MONITOR.add(f"{self.tag} raw", self.mamba(
            x_in
        ))

        y = MONITOR.add(f"{self.tag}", self.norm(
            y
        ))

        if not return_boundary_states:

            return y

        # ----------------------------------------------------------------------
        # CHUNK BOUNDARIES
        # ----------------------------------------------------------------------

        n_chunks = (
            L
            + self.chunk_size
            - 1
        ) // self.chunk_size

        chunk_starts = (
            torch.arange(
                n_chunks,
                device=x.device,
                dtype=torch.long,
            )
            * self.chunk_size
        )

        end_indices = torch.clamp(
            chunk_starts
            + self.chunk_size
            - 1,
            max=L - 1,
        )

        # ----------------------------------------------------------------------
        # UNMASKED
        # ----------------------------------------------------------------------

        if mask_bool is None:

            boundary_indices = (
                end_indices
                .unsqueeze(0)
                .expand(B, -1)
            )

            boundary_mask = torch.ones(
                B,
                n_chunks,
                device=x.device,
                dtype=torch.float32,
            )

        # ----------------------------------------------------------------------
        # MASKED
        # ----------------------------------------------------------------------

        else:

            lengths = (
                mask_bool.long()
                .sum(dim=1)
            )

            boundary_indices = torch.minimum(
                end_indices.unsqueeze(0),
                torch.clamp(
                    lengths.unsqueeze(1) - 1,
                    min=0,
                ),
            )

            chunk_ids = (
                torch.arange(
                    n_chunks,
                    device=x.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
            )

            boundary_mask = (
                lengths.unsqueeze(1)
                >
                chunk_ids
                * self.chunk_size
            ).float()

        # ----------------------------------------------------------------------
        # SAFE BATCH GATHER
        # ----------------------------------------------------------------------

        gather_index = (
            boundary_indices
            .unsqueeze(-1)
            .expand(
                B,
                n_chunks,
                D,
            )
        )

        boundary_states = MONITOR.add(f"{self.tag} boundary", torch.gather(
            y,
            dim=1,
            index=gather_index,
        ))

        return (
            y,
            boundary_states,
            boundary_mask,
        )


# ==============================================================================
# HVSC
# ==============================================================================

LOGVAR_MIN = float(os.environ.get("HEDO_LOGVAR_MIN", "-6.0"))
LOGVAR_MAX = float(os.environ.get("HEDO_LOGVAR_MAX", "4.0"))
STD_MIN, STD_MAX = 1e-3, 20.0
VAR_FLOOR = 1e-6
# Bound on (mu1 - mu2)^2 per latent dimension. Posterior means are linear maps of
# LayerNorm-ed boundary states, so |mu| stays O(10) in normal training; 1e4 (|diff| <= 100)
# is only reached by divergent parameters. Saturation is logged every batch.
MAX_SQ_MEAN_DIFF = 1e4
KL_NEG_TOL = 1e-4


class ChunkWiseHVSC(nn.Module):
    """HVSC aligns modality-specific boundary-state posterior distributions using symmetric KL regularization."""

    def __init__(
        self,
        d_state=128,
        d_latent=128,
        logvar_min=LOGVAR_MIN,
        logvar_max=LOGVAR_MAX,
    ):

        super().__init__()

        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.last_stats = {}

        self.img_mu = nn.Linear(
            d_state,
            d_latent,
        )

        self.img_logvar = nn.Linear(
            d_state,
            d_latent,
        )

        self.txt_mu = nn.Linear(
            d_state,
            d_latent,
        )

        self.txt_logvar = nn.Linear(
            d_state,
            d_latent,
        )

    def pool_boundary_states(
        self,
        states,
        mask=None,
    ):

        if mask is None:

            return states.mean(
                dim=1
            )

        weights = (
            mask
            .to(dtype=states.dtype)
            .unsqueeze(-1)
        )

        pooled = (
            states * weights
        ).sum(
            dim=1
        )

        denominator = (
            weights.sum(
                dim=1
            )
            .clamp(
                min=1.0
            )
        )

        return pooled / denominator

    def _posterior(
        self,
        mu_layer,
        logvar_layer,
        h,
    ):

        mu = mu_layer(h).float()

        logvar = logvar_layer(
            h
        ).float().clamp(
            self.logvar_min,
            self.logvar_max,
        )

        return (
            mu,
            logvar,
        )

    @staticmethod
    def _kl_diag_gaussian(
        mu1,
        logvar1,
        mu2,
        logvar2,
    ):

        mu1 = mu1.float()
        logvar1 = logvar1.float()
        mu2 = mu2.float()
        logvar2 = logvar2.float()

        var1 = torch.exp(logvar1).clamp_min(VAR_FLOOR)
        var2 = torch.exp(logvar2).clamp_min(VAR_FLOOR)

        sq_diff = (mu1 - mu2).pow(2).clamp(max=MAX_SQ_MEAN_DIFF)

        kl = 0.5 * (
            logvar2
            - logvar1
            +
            (
                var1
                +
                sq_diff
            )
            / var2
            - 1.0
        )

        return kl.sum(
            dim=-1
        ).mean()

    @staticmethod
    def _sample(
        mu,
        logvar,
        sample=True,
    ):

        if not sample:

            return mu

        eps = torch.randn_like(
            mu
        )

        std = torch.exp(
            0.5 * logvar.float()
        ).clamp(
            min=STD_MIN,
            max=STD_MAX,
        )

        return (
            mu
            +
            eps
            *
            std
        )

    def forward(
        self,
        img_states,
        txt_states,
        img_mask=None,
        txt_mask=None,
        sample_posterior=True,
    ):

        h_img = self.pool_boundary_states(
            img_states,
            img_mask,
        )

        h_txt = self.pool_boundary_states(
            txt_states,
            txt_mask,
        )

        mu_img, logvar_img = (
            self._posterior(
                self.img_mu,
                self.img_logvar,
                h_img,
            )
        )

        mu_txt, logvar_txt = (
            self._posterior(
                self.txt_mu,
                self.txt_logvar,
                h_txt,
            )
        )

        MONITOR.add("HVSC mu image", mu_img)
        MONITOR.add("HVSC logvar image", logvar_img)
        MONITOR.add("HVSC mu text", mu_txt)
        MONITOR.add("HVSC logvar text", logvar_txt)
        if MONITOR.active:
            MONITOR.add("HVSC variance image", torch.exp(logvar_img.float()).clamp_min(VAR_FLOOR))
            MONITOR.add("HVSC variance text", torch.exp(logvar_txt.float()).clamp_min(VAR_FLOOR))
            MONITOR.add("HVSC std image", torch.exp(0.5 * logvar_img.float()).clamp(min=STD_MIN, max=STD_MAX))
            MONITOR.add("HVSC std text", torch.exp(0.5 * logvar_txt.float()).clamp(min=STD_MIN, max=STD_MAX))

        kl_i2t = self._kl_diag_gaussian(
            mu_img,
            logvar_img,
            mu_txt,
            logvar_txt,
        )

        kl_t2i = self._kl_diag_gaussian(
            mu_txt,
            logvar_txt,
            mu_img,
            logvar_img,
        )

        symmetric_kl = MONITOR.add("HVSC KL", 0.5 * (
            kl_i2t
            +
            kl_t2i
        ))

        if MONITOR.active:
            with torch.no_grad():
                sq = (mu_img.float() - mu_txt.float()).pow(2)
                self.last_stats = {
                    "kl_raw": symmetric_kl.detach(),
                    "mu_absmax": torch.maximum(mu_img.abs().max(), mu_txt.abs().max()).float(),
                    "logvar_min": torch.minimum(logvar_img.min(), logvar_txt.min()),
                    "logvar_max": torch.maximum(logvar_img.max(), logvar_txt.max()),
                    "sq_diff_max": sq.max(),
                    "sq_diff_saturated_frac": (sq > MAX_SQ_MEAN_DIFF).float().mean(),
                }
            finite = torch.isfinite(symmetric_kl)
            nonneg = symmetric_kl >= -KL_NEG_TOL
            if not bool(finite & nonneg):
                with torch.no_grad():
                    diag = {
                        "kl": float(symmetric_kl), "kl_i2t": float(kl_i2t), "kl_t2i": float(kl_t2i),
                        "mu_img": tensor_stats(mu_img), "mu_txt": tensor_stats(mu_txt),
                        "logvar_img": tensor_stats(logvar_img), "logvar_txt": tensor_stats(logvar_txt),
                        "var_img": tensor_stats(logvar_img.float().exp()),
                        "var_txt": tensor_stats(logvar_txt.float().exp()),
                        "sq_mean_diff": tensor_stats(sq),
                    }
                print("[HVSC KL DIAGNOSTIC]", json.dumps(diag, indent=2), flush=True)
                raise NonFiniteError(
                    "HVSC symmetric KL is " + ("non-finite" if not bool(finite) else "negative beyond tolerance"))

        z_img = self._sample(
            mu_img,
            logvar_img,
            sample_posterior,
        )

        z_txt = self._sample(
            mu_txt,
            logvar_txt,
            sample_posterior,
        )

        MONITOR.add("HVSC z image (sampled)" if sample_posterior else "HVSC z image (mean)", z_img)
        MONITOR.add("HVSC z text (sampled)" if sample_posterior else "HVSC z text (mean)", z_txt)

        if MONITOR.active:
            with torch.no_grad():
                for name, z, mu in (("img", z_img, mu_img), ("txt", z_txt, mu_txt)):
                    z, mu = z.float(), mu.float()
                    self.last_stats[f"z_mu_cosine_{name}"] = F.cosine_similarity(z, mu, dim=-1).mean()
                    self.last_stats[f"noise_to_signal_{name}"] = ((z - mu).norm(dim=-1)
                                                                  / mu.norm(dim=-1).clamp_min(1e-12)).mean()

        return (
            z_img,
            z_txt,
            symmetric_kl,
        )


# ==============================================================================
# FULL MODEL
# ==============================================================================

class HEDO_HVSC_Model(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        d_state=64,
        chunk_size=16,
        use_hedo=True,
        use_hvsc=True,
    ):

        super().__init__()

        self.use_hedo = use_hedo
        self.use_hvsc = use_hvsc

        self.img_proj = nn.Linear(
            768,
            embed_dim,
        )

        self.txt_proj = nn.Linear(
            768,
            embed_dim,
        )

        if use_hedo:

            self.hedo_img = HEDO(
                d_model=embed_dim,
                tag="HEDO image",
            )

            self.hedo_txt = HEDO(
                d_model=embed_dim,
                tag="HEDO text",
            )

        self.mamba2_img = (
            NativeMamba2SequenceBlock(
                d_model=embed_dim,
                d_state=d_state,
                chunk_size=chunk_size,
                d_conv=4,
                expand=2,
                headdim=64,
                tag="Mamba image",
            )
        )

        self.mamba2_txt = (
            NativeMamba2SequenceBlock(
                d_model=embed_dim,
                d_state=d_state,
                chunk_size=chunk_size,
                d_conv=4,
                expand=2,
                headdim=64,
                tag="Mamba text",
            )
        )

        if use_hvsc:

            self.hvsc = ChunkWiseHVSC(
                d_state=embed_dim,
                d_latent=64,
            )
            # e_M = LayerNorm_M(h_pool,M + alpha * W_out,M z_M), alpha = 0.1
            self.hvsc_out_img = nn.Linear(64, embed_dim)
            self.hvsc_out_txt = nn.Linear(64, embed_dim)
            self.hvsc_norm_img = nn.LayerNorm(embed_dim)
            self.hvsc_norm_txt = nn.LayerNorm(embed_dim)
            self.hvsc_alpha = 0.1

        self.head_img = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.head_txt = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(1.0 / 0.07)
            )
        )

    def forward(
        self,
        image_features,
        text_features,
        image_mask=None,
        text_mask=None,
        sample_posterior=True,
    ):

        x_img = self.img_proj(
            image_features
        )

        x_txt = self.txt_proj(
            text_features
        )

        if self.use_hedo:

            x_img = self.hedo_img(
                x_img
            )

            x_txt = self.hedo_txt(
                x_txt
            )

        (
            y_img,
            boundary_img,
            boundary_mask_img,
        ) = self.mamba2_img(
            x_img,
            mask=image_mask,
            return_boundary_states=True,
        )

        (
            y_txt,
            boundary_txt,
            boundary_mask_txt,
        ) = self.mamba2_txt(
            x_txt,
            mask=text_mask,
            return_boundary_states=True,
        )

        # mask-weighted mean of the full Mamba-2 sequence (all configurations)
        if image_mask is None:
            pooled_img = y_img.mean(dim=1)
        else:
            w_img = image_mask.to(dtype=y_img.dtype).unsqueeze(-1)
            pooled_img = (y_img * w_img).sum(dim=1) / w_img.sum(dim=1).clamp(min=1.0)
        if text_mask is None:
            pooled_txt = y_txt.mean(dim=1)
        else:
            w_txt = text_mask.to(dtype=y_txt.dtype).unsqueeze(-1)
            pooled_txt = (y_txt * w_txt).sum(dim=1) / w_txt.sum(dim=1).clamp(min=1.0)
        MONITOR.add("pooled image", pooled_img)
        MONITOR.add("pooled text", pooled_txt)

        if self.use_hvsc:

            lat_img, lat_txt, symmetric_kl = (
                self.hvsc(
                    boundary_img,
                    boundary_txt,
                    boundary_mask_img,
                    boundary_mask_txt,
                    sample_posterior=sample_posterior,
                )
            )
            z_img = MONITOR.add("HVSC fused image", self.hvsc_norm_img(
                pooled_img + self.hvsc_alpha * self.hvsc_out_img(lat_img)))
            z_txt = MONITOR.add("HVSC fused text", self.hvsc_norm_txt(
                pooled_txt + self.hvsc_alpha * self.hvsc_out_txt(lat_txt)))

        else:

            z_img = pooled_img
            z_txt = pooled_txt

            symmetric_kl = torch.zeros(
                (),
                device=image_features.device,
            )

        z_img = MONITOR.add("z image", F.normalize(
            self.head_img(z_img),
            dim=-1,
        ))

        z_txt = MONITOR.add("z text", F.normalize(
            self.head_txt(z_txt),
            dim=-1,
        ))

        return (
            z_img,
            z_txt,
            symmetric_kl,
        )


# ==============================================================================
# MULTI-POSITIVE INFONCE
# ==============================================================================

class SymmetricMultiPositiveInfoNCELoss(
    nn.Module
):

    def forward(
        self,
        z_img,
        z_txt,
        image_ids,
        logit_scale,
    ):

        unique_indices = []
        unique_image_ids = []

        seen = set()

        for idx, iid in enumerate(
            image_ids
        ):

            iid = str(iid)

            if iid not in seen:

                seen.add(iid)

                unique_indices.append(
                    idx
                )

                unique_image_ids.append(
                    iid
                )

        if len(unique_indices) == 0:

            raise ValueError(
                "No unique image IDs."
            )

        unique_idx = torch.tensor(
            unique_indices,
            device=z_img.device,
            dtype=torch.long,
        )

        unique_z_img = z_img.index_select(
            0,
            unique_idx,
        )

        # ----------------------------------------------------------------------
        # IMPORTANT:
        # 40 captions, 8 unique images
        # => similarity = 8 x 40
        # ----------------------------------------------------------------------

        sim = MONITOR.add("similarity", (
            unique_z_img.float()
            @ z_txt.float().T
        ) * logit_scale.float().clamp(
            max=100.0
        ))

        # ----------------------------------------------------------------------
        # 8 unique images x 40 captions; each image row has exactly its 5
        # captions as positives, each caption column exactly one positive image.
        # ----------------------------------------------------------------------
        pos_mask = torch.tensor(
            [
                [
                    uid == str(cid)
                    for cid in image_ids
                ]
                for uid in unique_image_ids
            ],
            device=z_img.device,
            dtype=torch.float32,
        )

        row_pos = pos_mask.sum(dim=1)
        col_pos = pos_mask.sum(dim=0)
        if not (bool((row_pos >= 1).all()) and bool((col_pos == 1).all())):
            raise ValueError(
                f"Invalid positive mask: per-image positives {row_pos.tolist()}, "
                f"per-caption positives {col_pos.tolist()}")

        log_prob_rows = MONITOR.add("row log-probabilities", F.log_softmax(sim, dim=1))
        log_prob_cols = MONITOR.add("column log-probabilities", F.log_softmax(sim.T, dim=1))

        loss_i2t = -(
            log_prob_rows
            * (pos_mask / row_pos.unsqueeze(1))
        ).sum(
            dim=1
        ).mean()

        loss_t2i = -(
            log_prob_cols
            * pos_mask.T
        ).sum(
            dim=1
        ).mean()

        return MONITOR.add("InfoNCE", 0.5 * (
            loss_i2t
            +
            loss_t2i
        ))


# ==============================================================================
# HARD ARCHITECTURE TEST
# ==============================================================================

print()
print("=" * 80)
print("ARCHITECTURE HARD GATE")
print("=" * 80)

test_x = torch.randn(
    2,
    64,
    128,
    device=DEVICE,
)

native_block = NativeMamba2SequenceBlock(
    d_model=128,
    d_state=64,
    chunk_size=16,
    d_conv=4,
    expand=2,
    headdim=64,
).to(DEVICE)

native_block.eval()

with torch.no_grad():

    test_output = native_block(
        test_x,
        return_boundary_states=True,
    )

test_y = test_output[0]
test_boundaries = test_output[1]
test_boundary_mask = test_output[2]

assert test_y.shape == (
    2,
    64,
    128,
)

assert test_boundaries.shape == (
    2,
    4,
    128,
)

assert test_boundary_mask.shape == (
    2,
    4,
)

assert torch.isfinite(
    test_y
).all()

assert torch.isfinite(
    test_boundaries
).all()

print(
    "Sequence output :",
    tuple(test_y.shape)
)

print(
    "Boundary output :",
    tuple(test_boundaries.shape)
)

print(
    "Boundary mask   :",
    tuple(test_boundary_mask.shape)
)

print(
    "NATIVE_BOUNDARY_GATE: PASS"
)


# ==============================================================================
# MASKED GATE
# ==============================================================================

masked_input = torch.randn(
    2,
    64,
    128,
    device=DEVICE,
)

masked_mask = torch.ones(
    2,
    64,
    device=DEVICE,
)

masked_mask[1, 48:] = 0

with torch.no_grad():

    masked_result = native_block(
        masked_input,
        mask=masked_mask,
        return_boundary_states=True,
    )

assert masked_result[0].shape == (
    2,
    64,
    128,
)

assert masked_result[1].shape == (
    2,
    4,
    128,
)

assert masked_result[2].shape == (
    2,
    4,
)

print(
    "NATIVE_MASKED_GATE: PASS"
)


# ==============================================================================
# FULL MODEL GATE
# ==============================================================================

baseline_model = HEDO_HVSC_Model(
    use_hedo=False,
    use_hvsc=False,
).to(DEVICE)

full_model = HEDO_HVSC_Model(
    use_hedo=True,
    use_hvsc=True,
).to(DEVICE)

for model_name, model in [
    ("baseline", baseline_model),
    ("full", full_model),
]:

    native_modules = [
        name
        for name, child
        in model.named_modules()
        if isinstance(
            child,
            Mamba2,
        )
    ]

    print(
        f"{model_name} native Mamba2:",
        native_modules,
    )

    if not native_modules:

        raise RuntimeError(
            f"{model_name} model has no native Mamba-2."
        )

print(
    "FULL_MODEL_NATIVE_MAMBA2: PASS"
)


# ==============================================================================
# LOSS GATE
# ==============================================================================

loss_fn = (
    SymmetricMultiPositiveInfoNCELoss()
)

dummy_images = F.normalize(
    torch.randn(
        40,
        128,
        device=DEVICE,
    ),
    dim=-1,
)

dummy_text = F.normalize(
    torch.randn(
        40,
        128,
        device=DEVICE,
    ),
    dim=-1,
)

dummy_ids = [
    f"image_{i // 5}"
    for i in range(40)
]

dummy_scale = torch.tensor(
    math.log(1.0 / 0.07),
    device=DEVICE,
)

dummy_loss = loss_fn(
    dummy_images,
    dummy_text,
    dummy_ids,
    dummy_scale,
)

if not torch.isfinite(
    dummy_loss
):

    raise RuntimeError(
        "Multi-positive InfoNCE returned non-finite loss."
    )

print(
    "8x40 MULTIPOSITIVE INFONCE: PASS"
)

print(
    "Dummy loss:",
    float(
        dummy_loss.detach().cpu()
    ),
)


# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

BENCHMARK_ROOT = Path(
    "/content/HEDO_HVSC_NATIVE_MAMBA2_BENCHMARK_V6"
)

RUN_ROOT = (
    BENCHMARK_ROOT
    / "runs"
)

DATA_ROOT = Path(
    "/content/data/flickr8k"
)

RUN_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 40
CAPTIONS_PER_IMAGE = 5
UNIQUE_IMAGES_PER_BATCH = 8

EPOCHS = int(
    os.environ.get(
        "HEDO_EPOCHS",
        "10",
    )
)

LEARNING_RATE = float(os.environ.get("HEDO_LR", "5e-5"))
WEIGHT_DECAY = float(os.environ.get("HEDO_WEIGHT_DECAY", "1e-4"))
KL_WEIGHT = float(os.environ.get("HEDO_KL_WEIGHT", "0.001"))
GRAD_CLIP = float(os.environ.get("HEDO_GRAD_CLIP", "0.5"))
ADAM_EPS = 1e-8
TRAIN_SAMPLE_POSTERIOR = os.environ.get("HEDO_TRAIN_SAMPLE_POSTERIOR", "0") == "1"
HVSC_Z_DIM = 64
HVSC_FUSION_ALPHA = 0.1
HEDO_GAMMA_INIT = 0.1
STABILITY_EPOCHS = int(os.environ.get("HEDO_STABILITY_EPOCHS", "1"))

BENCHMARK_VERSION = (
    "HEDO_HVSC_NATIVE_MAMBA2_V6_1_STABLE_MEAN"
)

LOSS_GEOMETRY_VERSION = (
    "unique_images_8x40_multipositive_v2"
)


# ==============================================================================
# MANIFEST
# ==============================================================================

def load_flickr8k():

    manifest_path = (
        DATA_ROOT
        / "manifest.json"
    )

    if not manifest_path.is_file():

        raise FileNotFoundError(
            f"Flickr8k manifest not found:\n{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    image_dir = Path(
        manifest["image_dir"]
    )

    caption_file = Path(
        manifest["caption_file"]
    )

    def read_split(path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return {
                x.strip()
                for x in f
                if x.strip()
            }

    split_ids = {

        "train": read_split(
            manifest["train_split"]
        ),

        "val": read_split(
            manifest["validation_split"]
        ),

        "test": read_split(
            manifest["test_split"]
        ),
    }

    records = []

    with open(
        caption_file,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if "\t" not in line:
                continue

            image_caption_id, caption = (
                line.split(
                    "\t",
                    1,
                )
            )

            image_id = (
                image_caption_id
                .split(
                    "#",
                    1,
                )[0]
                .strip()
            )

            caption = caption.strip()

            if image_id and caption:

                records.append(
                    (
                        image_id,
                        caption,
                    )
                )

    df = pd.DataFrame(
        records,
        columns=[
            "image_id",
            "caption",
        ],
    )

    tables = {}

    for split, ids in split_ids.items():

        tables[split] = (
            df[
                df["image_id"].isin(ids)
            ]
            .reset_index(drop=True)
        )

    # --------------------------------------------------------------------------
    # Flickr8k official counts expected by this benchmark
    # --------------------------------------------------------------------------

    assert len(split_ids["train"]) == 6000
    assert len(split_ids["val"]) == 1000
    assert len(split_ids["test"]) == 1000

    assert len(tables["train"]) == 30000
    assert len(tables["val"]) == 5000
    assert len(tables["test"]) == 5000

    for split in [
        "train",
        "val",
        "test",
    ]:

        counts = (
            tables[split]
            .groupby("image_id")
            .size()
        )

        assert (
            len(counts)
            == len(split_ids[split])
        )

        assert (
            counts == 5
        ).all()

    return (
        image_dir,
        tables,
    )


# ==============================================================================
# TOKENIZER / TRANSFORM
# ==============================================================================

IMAGE_TRANSFORM = tv_models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
# == ImageClassification(crop_size=224, resize_size=256, bilinear, mean=(0.485, 0.456, 0.406),
#                        std=(0.229, 0.224, 0.225)): the exact preprocessing the pretrained weights expect.

# ==============================================================================
# DATASET
# ==============================================================================

class FlickrDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        tokenizer,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.image_dir = Path(
            image_dir
        )

        self.tokens = tokenizer(
            self.df["caption"]
            .astype(str)
            .tolist(),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[index]

        image_id = str(
            row["image_id"]
        )

        image_path = (
            self.image_dir
            / image_id
        )

        if not image_path.is_file():

            raise FileNotFoundError(
                image_path
            )

        with Image.open(
            image_path
        ) as image:

            image = IMAGE_TRANSFORM(
                image.convert("RGB")
            )

        return {

            "image": image,

            "input_ids":
                self.tokens[
                    "input_ids"
                ][index],

            "attention_mask":
                self.tokens[
                    "attention_mask"
                ][index],

            "image_id":
                image_id,
        }


# ==============================================================================
# GROUPED BATCH SAMPLER
# ==============================================================================

class AtomicGroupedBatchSampler(
    Sampler
):

    def __init__(
        self,
        dataframe,
        batch_size=40,
        captions_per_image=5,
        shuffle=True,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.batch_size = int(
            batch_size
        )

        self.k = int(
            captions_per_image
        )

        self.shuffle = bool(
            shuffle
        )

        if (
            self.batch_size
            % self.k
            != 0
        ):

            raise ValueError(
                "Batch size must be divisible "
                "by captions per image."
            )

        self.images_per_batch = (
            self.batch_size
            // self.k
        )

        self.image_to_indices = (
            defaultdict(list)
        )

        for idx, row in (
            self.df.iterrows()
        ):

            self.image_to_indices[
                str(row["image_id"])
            ].append(
                idx
            )

        self.unique_images = sorted(
            self.image_to_indices
        )

        self.num_batches = (
            len(
                self.unique_images
            )
            //
            self.images_per_batch
        )

    def __len__(self):

        return self.num_batches

    def __iter__(self):

        images = list(
            self.unique_images
        )

        if self.shuffle:

            random.shuffle(
                images
            )

        usable = (
            self.num_batches
            *
            self.images_per_batch
        )

        for start in range(
            0,
            usable,
            self.images_per_batch,
        ):

            selected = images[
                start:
                start
                +
                self.images_per_batch
            ]

            batch = []

            for image_id in selected:

                indices = (
                    self.image_to_indices[
                        image_id
                    ]
                )

                if len(indices) != self.k:

                    raise RuntimeError(
                        f"{image_id}: "
                        f"expected {self.k} captions."
                    )

                chosen = (
                    random.sample(
                        indices,
                        self.k,
                    )
                    if self.shuffle
                    else indices[:self.k]
                )

                batch.extend(
                    chosen
                )

            assert (
                len(batch)
                == self.batch_size
            )

            yield batch


# ==============================================================================
# FROZEN VISION BACKBONE
# ==============================================================================

class FrozenVisionBackbone(nn.Module):
    """Pretrained torchvision ViT-B/16 (IMAGENET1K_V1), official forward path up to the encoder output.

    torchvision: VisionTransformer.forward(x) = heads(encoder(cat(class_token, _process_input(x)))[:, 0])
                 Encoder.forward(x) = ln(layers(dropout(x + pos_embedding)))
    forward() returns the 196 patch tokens of exactly that encoder output (class token dropped).
    forward_logits() reuses the same path plus heads; the fidelity gate requires it to equal vit(x).
    """

    WEIGHTS = tv_models.ViT_B_16_Weights.IMAGENET1K_V1

    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = tv_models.vit_b_16(weights=self.WEIGHTS if pretrained else None)
        assert self.vit.encoder.pos_embedding.shape == (1, 197, 768)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def _encode_all_tokens(self, x):
        x = self.vit._process_input(x)
        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        return self.vit.encoder(torch.cat([cls, x], dim=1))

    @torch.no_grad()
    def forward(self, x):
        return self._encode_all_tokens(x)[:, 1:, :]

    @torch.no_grad()
    def forward_logits(self, x):
        return self.vit.heads(self._encode_all_tokens(x)[:, 0])


# ==============================================================================
# FROZEN TEXT BACKBONE
# ==============================================================================

class FrozenLanguageBackbone(
    nn.Module
):

    def __init__(self):

        super().__init__()

        # The retrieval encoder uses last_hidden_state only. The roberta-base checkpoint
        # is a masked-LM checkpoint: it has lm_head.* (not needed) and no pooler.dense.*
        # (which would otherwise be randomly initialised). add_pooling_layer=False builds
        # exactly the pretrained encoder, so every parameter comes from the checkpoint.
        self.roberta, info = (
            AutoModel
            .from_pretrained(
                "roberta-base",
                add_pooling_layer=False,
                output_loading_info=True,
            )
        )

        missing = sorted(info.get("missing_keys", []))
        unexpected = sorted(info.get("unexpected_keys", []))
        print("ROBERTA_BACKBONE_CLASS     :", type(self.roberta).__name__, flush=True)
        print("ROBERTA_CONFIG             :", {k: getattr(self.roberta.config, k) for k in (
            "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size", "max_position_embeddings")},
            flush=True)
        print("ROBERTA_PRETRAINED_WEIGHTS :", "roberta-base",
              getattr(self.roberta.config, "_commit_hash", None), flush=True)
        print("ROBERTA_UNEXPECTED_KEYS    :", unexpected, flush=True)
        print("ROBERTA_MISSING_KEYS       :", missing, flush=True)
        if missing:
            raise RuntimeError(f"RoBERTa encoder has randomly initialised weights: {missing}")
        if any(not k.startswith("lm_head.") for k in unexpected):
            raise RuntimeError(f"Unexpected non-LM-head checkpoint keys: {unexpected}")
        print("ROBERTA_LOAD_AUDIT: PASS (all encoder weights pretrained; lm_head.* intentionally unused)",
              flush=True)

        for p in self.parameters():

            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

@torch.no_grad()
def get_features(
    batch,
    vision,
    language,
):

    images = (
        batch["image"]
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    input_ids = (
        batch["input_ids"]
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    attention_mask = (
        batch["attention_mask"]
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        image_features = vision(
            images
        )

        text_features = language(
            input_ids,
            attention_mask,
        )

    return (
        image_features.float(),
        text_features.float(),
        attention_mask,
    )


# ==============================================================================
# RETRIEVAL EXTRACTION
# ==============================================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    vision,
    language,
    sample_posterior=False,
    noise_seed=None,
):

    model.eval()
    assert not torch.is_grad_enabled(), "evaluation must never build a graph"
    if noise_seed is not None:
        torch.manual_seed(noise_seed)

    all_img = []
    all_txt = []
    all_ids = []

    for batch in loader:

        image_features, text_features, mask = (
            get_features(
                batch,
                vision,
                language,
            )
        )

        z_img, z_txt, _ = model(
            image_features,
            text_features,
            text_mask=mask,
            sample_posterior=sample_posterior,
        )

        all_img.append(
            z_img.float()
            .cpu()
            .numpy()
        )

        all_txt.append(
            z_txt.float()
            .cpu()
            .numpy()
        )

        all_ids.extend(
            str(x)
            for x
            in batch["image_id"]
        )

    raw_img = np.concatenate(
        all_img,
        axis=0,
    )

    txt = np.concatenate(
        all_txt,
        axis=0,
    )

    unique_ids = []
    unique_img = []
    seen = set()

    for idx, iid in enumerate(
        all_ids
    ):

        if iid not in seen:

            seen.add(iid)

            unique_ids.append(
                iid
            )

            unique_img.append(
                raw_img[idx]
            )

    unique_img = np.asarray(
        unique_img,
        dtype=np.float32,
    )

    txt = np.asarray(
        txt,
        dtype=np.float32,
    )

    assert (
        unique_img.shape
        ==
        (1000, 128)
    )

    assert (
        txt.shape
        ==
        (5000, 128)
    )

    return {
        "image_embeddings":
            unique_img,

        "text_embeddings":
            txt,

        "image_ids":
            unique_ids,

        "caption_image_ids":
            all_ids,
    }


# ==============================================================================
# RETRIEVAL METRICS
# ==============================================================================

def compute_retrieval_metrics(
    extracted
):

    images = extracted[
        "image_embeddings"
    ]

    texts = extracted[
        "text_embeddings"
    ]

    image_ids = extracted[
        "image_ids"
    ]

    caption_ids = extracted[
        "caption_image_ids"
    ]

    similarity = (
        images
        @
        texts.T
    )

    image_index = {
        iid: idx
        for idx, iid
        in enumerate(image_ids)
    }

    image_to_captions = (
        defaultdict(list)
    )

    for idx, iid in enumerate(
        caption_ids
    ):

        image_to_captions[
            iid
        ].append(
            idx
        )

    i2t_ranks = []

    for i, iid in enumerate(
        image_ids
    ):

        correct = set(
            image_to_captions[
                iid
            ]
        )

        ranking = np.argsort(
            -similarity[i]
        )

        rank = next(
            r
            for r, caption_index
            in enumerate(ranking)
            if caption_index
            in correct
        )

        i2t_ranks.append(
            rank
        )

    t2i_ranks = []

    for c, iid in enumerate(
        caption_ids
    ):

        target = image_index[
            iid
        ]

        ranking = np.argsort(
            -similarity[:, c]
        )

        rank = int(
            np.where(
                ranking == target
            )[0][0]
        )

        t2i_ranks.append(
            rank
        )

    i2t = np.asarray(
        i2t_ranks
    )

    t2i = np.asarray(
        t2i_ranks
    )

    metrics = {

        "i2t_r1":
            float(
                np.mean(
                    i2t < 1
                ) * 100
            ),

        "i2t_r5":
            float(
                np.mean(
                    i2t < 5
                ) * 100
            ),

        "i2t_r10":
            float(
                np.mean(
                    i2t < 10
                ) * 100
            ),

        "i2t_medr":
            float(
                np.median(
                    i2t + 1
                )
            ),

        "i2t_meanr":
            float(
                np.mean(
                    i2t + 1
                )
            ),

        "t2i_r1":
            float(
                np.mean(
                    t2i < 1
                ) * 100
            ),

        "t2i_r5":
            float(
                np.mean(
                    t2i < 5
                ) * 100
            ),

        "t2i_r10":
            float(
                np.mean(
                    t2i < 10
                ) * 100
            ),

        "t2i_medr":
            float(
                np.median(
                    t2i + 1
                )
            ),

        "t2i_meanr":
            float(
                np.mean(
                    t2i + 1
                )
            ),
    }

    metrics["mean_recall"] = float(
        (
            metrics["i2t_r1"]
            +
            metrics["i2t_r5"]
            +
            metrics["i2t_r10"]
            +
            metrics["t2i_r1"]
            +
            metrics["t2i_r5"]
            +
            metrics["t2i_r10"]
        )
        / 6.0
    )

    return metrics


# ==============================================================================
# TRAINING, GATES AND ORCHESTRATION
# ==============================================================================
# Modes (HEDO_MODE):
#   gates : run all 14 GO/NO-GO gates (includes a 1-epoch stability run, train/inference
#           consistency and checkpoint reload) and write results/GO_NO_GO.json. No results.
#   train : refuses to start unless GO_NO_GO.json says GO for the same methodology hash;
#           re-runs the cheap gates, trains, selects the best checkpoint on validation only,
#           evaluates the test split exactly once.
# Precision: FP32 trainable stack (no autocast, no GradScaler); frozen backbones FP16 autocast.
# ==============================================================================

import traceback

LOGIT_SCALE_MIN = math.log(1.0)
LOGIT_SCALE_MAX = math.log(100.0)
FAILED_STATUS = "FAILED_NUMERICAL_STABILITY"
RESULTS_ROOT = Path(os.environ.get("HEDO_RESULTS_ROOT", str(BENCHMARK_ROOT / "results_v6_1")))
CONSISTENCY_MIN_COSINE = float(os.environ.get("HEDO_CONSISTENCY_MIN_COSINE", "0.90"))
CONFIGS = {
    "baseline": {"use_hedo": False, "use_hvsc": False},
    "mamba2_hedo": {"use_hedo": True, "use_hvsc": False},
    "mamba2_hvsc": {"use_hedo": False, "use_hvsc": True},
    "full_hedo_hvsc": {"use_hedo": True, "use_hvsc": True},
}
BEST_CHECKPOINT_SELECTION_CRITERION = "max validation mean_recall (validation split only; ties keep the earlier epoch)"
RUN_STATE = {"stage": None, "epoch": None, "batch": None, "lr": None, "last_grad_norm": None}
PARAM_GROUP_OF = {}


class GateFailure(RuntimeError):
    def __init__(self, gate, message):
        super().__init__(f"{gate}: {message}")
        self.gate = gate


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def shared_hyperparameters():
    """Everything that must be identical across seeds and ablation configurations."""
    return {
        "embed_dim": 128, "d_state": 64, "chunk_size": 16,
        "mamba2": {"d_model": 128, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 64},
        "batch_geometry": {"unique_images": UNIQUE_IMAGES_PER_BATCH, "captions_per_image": CAPTIONS_PER_IMAGE,
                           "captions": BATCH_SIZE},
        "optimizer": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR per step, T_max = steps_per_epoch * epochs",
        "grad_clip_norm": GRAD_CLIP, "epochs": EPOCHS, "kl_weight": KL_WEIGHT,
        "logvar_range": [LOGVAR_MIN, LOGVAR_MAX], "std_range": [STD_MIN, STD_MAX], "var_floor": VAR_FLOOR,
        "max_sq_mean_diff": MAX_SQ_MEAN_DIFF, "logit_scale_range": [1.0, 100.0],
        "precision": "FP32 trainable stack; FP16 autocast frozen backbones",
        "image_preprocessing": repr(IMAGE_TRANSFORM), "text": "roberta-base tokenizer, max_length 64, padding max_length",
        "vision_backbone": "torchvision vit_b_16 IMAGENET1K_V1 (frozen), 196 patch tokens",
        "text_backbone": "roberta-base RobertaModel without pooler (frozen), last_hidden_state",
        "checkpoint_selection": BEST_CHECKPOINT_SELECTION_CRITERION,
        "methodology_version": "HEDO_HVSC_NATIVE_MAMBA2_V6_1_STABLE_MEAN",
        "adam_eps": ADAM_EPS, "grad_clip_error_if_nonfinite": True, "hedo_gamma_init": HEDO_GAMMA_INIT,
        "hvsc_z_dim": HVSC_Z_DIM, "hvsc_fusion": "normalize(head(LayerNorm(h_pool + 0.1 * W_out z)))",
        "hvsc_fusion_alpha": HVSC_FUSION_ALPHA,
        "training_posterior_policy": "sampled z" if TRAIN_SAMPLE_POSTERIOR else "posterior mean (z = mu)",
        "inference_policy": "posterior mean (sample_posterior=False), model.eval(), no_grad",
        "consistency_min_cosine": CONSISTENCY_MIN_COSINE,
    }


def methodology_hash():
    return sha256_json({"runner_sha256": sha256_path(Path(__file__).resolve()), "shared": shared_hyperparameters()})


# ------------------------------------------------------------------------------
# optimisation
# ------------------------------------------------------------------------------

def bounded_logit_scale(model):
    return MONITOR.add("logit scale", model.logit_scale.clamp(min=LOGIT_SCALE_MIN, max=LOGIT_SCALE_MAX).exp())


def build_optimizer(model, verbose=False):
    groups = {"hedo": [], "hvsc": [], "mamba2": [], "projection_head_temperature": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("hedo_"):
            groups["hedo"].append((name, p))
        elif name.startswith("hvsc"):
            groups["hvsc"].append((name, p))
        elif name.startswith("mamba2_"):
            groups["mamba2"].append((name, p))
        else:
            groups["projection_head_temperature"].append((name, p))
    param_groups = [{"name": k, "params": [p for _, p in v], "param_names": [n for n, _ in v],
                     "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY} for k, v in groups.items() if v]
    PARAM_GROUP_OF.update({n: g["name"] for g in param_groups for n in g["param_names"]})
    in_optimizer = sum(p.numel() for g in param_groups for p in g["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if in_optimizer != trainable:
        raise GateFailure("OPTIMIZER_MEMBERSHIP", f"{in_optimizer} params in optimizer vs {trainable} trainable")
    if verbose:
        for g in param_groups:
            print(f"  param group {g['name']:<30} {sum(p.numel() for p in g['params']):>9,} params", flush=True)
    return torch.optim.AdamW([{k: v for k, v in g.items() if k != "param_names"} | {"name": g["name"]}
                              for g in param_groups], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                             eps=ADAM_EPS), param_groups


def training_step(model, batch, vision, language, optimizer, context, verbose=False, keep_grads=False):
    optimizer.zero_grad(set_to_none=True)
    image_features, text_features, mask = get_features(batch, vision, language)
    MONITOR.add("image features", image_features)
    MONITOR.add("text features", text_features)
    image_ids = [str(x) for x in batch["image_id"]]

    RUN_STATE["lr"] = optimizer.param_groups[0]["lr"]
    z_img, z_txt, kl_raw = model(image_features, text_features, text_mask=mask,
                                 sample_posterior=TRAIN_SAMPLE_POSTERIOR)
    scale = bounded_logit_scale(model)
    info_loss = loss_fn(z_img, z_txt, image_ids, scale)
    kl_scaled = KL_WEIGHT * kl_raw
    total_loss = MONITOR.add("total loss", info_loss + kl_scaled)
    MONITOR.check(f"{context} forward", verbose=verbose)
    if verbose:
        print("REAL_BATCH_FORWARD: PASS\nREAL_BATCH_LOSS: PASS", flush=True)

    total_loss.backward()
    named_grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    for n, g in named_grads:
        MONITOR.add(f"grad {n}", g)
    MONITOR.check(f"{context} gradients")          # every gradient finite BEFORE clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRAD_CLIP,
        error_if_nonfinite=True,
    )
    RUN_STATE["last_grad_norm"] = float(grad_norm)
    if verbose:
        print(f"REAL_BATCH_BACKWARD: PASS ({len(named_grads)} gradient tensors, pre-clip norm {float(grad_norm):.4f})",
              flush=True)
    grads_snapshot = {n: g.detach().clone() for n, g in named_grads} if keep_grads else None

    optimizer.step()
    with torch.no_grad():
        model.logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)
    for n, p in model.named_parameters():
        MONITOR.add(f"param {n}", p)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.is_floating_point():
                MONITOR.add(f"optimizer state {k}", v)
    MONITOR.check(f"{context} optimizer step")
    if verbose:
        print("REAL_BATCH_OPTIMIZER_STEP: PASS", flush=True)

    record = {"loss": total_loss.item(), "infonce": info_loss.item(), "kl_raw": kl_raw.item(),
              "kl_weighted": kl_scaled.item(), "grad_norm": float(grad_norm), "logit_scale": scale.item()}
    if getattr(model, "use_hvsc", False):
        record.update({f"hvsc_{k}": float(v) for k, v in model.hvsc.last_stats.items()})
    if getattr(model, "use_hedo", False):
        record.update({f"hedo_img_{k}": float(v) for k, v in model.hedo_img.last_stats.items()})
        record.update({f"hedo_txt_{k}": float(v) for k, v in model.hedo_txt.last_stats.items()})
    return record, grads_snapshot


def train_single_epoch(model, loader, vision, language, optimizer, scheduler, epoch, log=None):
    model.train()
    totals, n, grad_norm_max = defaultdict(float), 0, 0.0
    t_start = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        RUN_STATE.update(epoch=epoch, batch=batch_idx)
        lr = scheduler.get_last_lr()[0]
        record, _ = training_step(model, batch, vision, language, optimizer,
                                  context=f"epoch {epoch} batch {batch_idx}", verbose=(batch_idx == 1))
        scheduler.step()
        for k, v in record.items():
            totals[k] += v
        grad_norm_max = max(grad_norm_max, record["grad_norm"])
        n += 1
        if log is not None:
            log.write(json.dumps({"epoch": epoch, "batch": batch_idx, "lr": lr, **record}) + "\n")
            log.flush()
        total_batches = len(loader)
        log_freq = max(1, int(os.environ.get("HEDO_LOG_INTERVAL", "10")))
        if batch_idx == 1 or batch_idx % log_freq == 0 or batch_idx == total_batches:
            now = time.time()
            elapsed = now - t_start
            speed = batch_idx / max(1e-5, elapsed)
            eta_sec = (total_batches - batch_idx) / max(1e-5, speed)
            eta_m, eta_s = divmod(int(eta_sec), 60)
            pct = (batch_idx / total_batches) * 100
            print(f"  [Epoch {epoch:02d}/{EPOCHS} | Batch {batch_idx:03d}/{total_batches} ({pct:5.1f}%)] "
                  f"loss: {record['loss']:.4f} (info: {record['infonce']:.4f}, kl: {record['kl_raw']:.4f}) | "
                  f"grad: {record['grad_norm']:.3f} | lr: {lr:.2e} | {speed:.1f} b/s | ETA: {eta_m:02d}:{eta_s:02d}",
                  flush=True)
    out = {k: v / max(1, n) for k, v in totals.items()}
    out.update(grad_norm_max=grad_norm_max, lr_end=scheduler.get_last_lr()[0],
               logit_scale_end=float(model.logit_scale.clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()))
    return out


def evaluate_split(model, loader, vision, language, expected_image_ids, context, sample_posterior=False,
                   noise_seed=None):
    """Full-split retrieval with deterministic posterior means unless explicitly sampling for the consistency gate."""
    print(f"  --> Evaluating {context} ({len(loader)} batches)...", flush=True)
    embeddings = extract_embeddings(model, loader, vision, language, sample_posterior=sample_posterior,
                                    noise_seed=noise_seed)
    for key in ("image_embeddings", "text_embeddings"):
        if not np.isfinite(embeddings[key]).all():
            raise NonFiniteError(f"non-finite {key} during {context}")
    ids, caps = embeddings["image_ids"], embeddings["caption_image_ids"]
    if set(ids) != set(expected_image_ids) or len(ids) != len(expected_image_ids) or len(caps) != 5 * len(ids):
        raise GateFailure("RETRIEVAL_EVALUATOR", f"{context}: evaluated ids do not match the split")
    counts = defaultdict(int)
    for c in caps:
        counts[c] += 1
    if set(counts.values()) != {CAPTIONS_PER_IMAGE}:
        raise GateFailure("RETRIEVAL_EVALUATOR", f"{context}: not exactly 5 captions per evaluated image")
    metrics = compute_retrieval_metrics(embeddings)
    print(f"  --> {context} complete: i2t_r1={metrics['i2t_r1']:.1f}% | t2i_r1={metrics['t2i_r1']:.1f}% | mean_recall={metrics['mean_recall']:.2f}%", flush=True)
    return embeddings, metrics


# ------------------------------------------------------------------------------
# gates
# ------------------------------------------------------------------------------

class GateReport:
    def __init__(self):
        self.gates = {}

    def record(self, name, passed, evidence):
        self.gates[name] = {"status": "PASS" if passed else "FAIL", "evidence": evidence}
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            raise GateFailure(name, json.dumps(evidence, default=str)[:2000])


def dataset_gate(report, tables):
    """Recompute every split property from the raw files; the manifest's own flags are not trusted."""
    manifest = json.load(open(DATA_ROOT / "manifest.json", encoding="utf-8"))
    image_dir = Path(manifest["image_dir"])
    splits = {}
    for split, key in (("train", "train_split"), ("val", "validation_split"), ("test", "test_split")):
        lines = [x.strip() for x in open(manifest[key], encoding="utf-8") if x.strip()]
        splits[split] = lines
    ids = {s: set(v) for s, v in splits.items()}
    raw_rows = []
    with open(manifest["caption_file"], encoding="utf-8", errors="strict") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                key, caption = line.split("\t", 1)
                raw_rows.append((key.split("#", 1)[0].strip(), caption.strip()))
    by_image = defaultdict(list)
    for iid, cap in raw_rows:
        by_image[iid].append(cap)
    evidence = {
        "split_sizes": {s: len(v) for s, v in ids.items()},
        "duplicate_ids_within_split": {s: len(splits[s]) - len(ids[s]) for s in ids},
        "overlaps": {"train_val": len(ids["train"] & ids["val"]), "train_test": len(ids["train"] & ids["test"]),
                     "val_test": len(ids["val"] & ids["test"])},
        "caption_counts": {s: int(sum(len(by_image[i]) for i in ids[s])) for s in ids},
        "images_without_5_captions": {s: sorted(i for i in ids[s] if len(by_image[i]) != 5)[:5] for s in ids},
        "missing_image_files": sorted(i for s in ids for i in ids[s] if not (image_dir / i).is_file())[:5],
        "loader_tables_match_raw": all(
            sorted(zip(tables[s]["image_id"], tables[s]["caption"])) ==
            sorted((i, c) for i in ids[s] for c in by_image[i]) for s in ids),
    }
    ok = (evidence["split_sizes"] == {"train": 6000, "val": 1000, "test": 1000}
          and not any(evidence["duplicate_ids_within_split"].values())
          and not any(evidence["overlaps"].values())
          and evidence["caption_counts"] == {"train": 30000, "val": 5000, "test": 5000}
          and not any(evidence["images_without_5_captions"].values())
          and not evidence["missing_image_files"] and evidence["loader_tables_match_raw"])
    fingerprint = {
        "caption_file_sha256": sha256_path(manifest["caption_file"]),
        "split_file_sha256": {s: sha256_path(manifest[k]) for s, k in
                              (("train", "train_split"), ("val", "validation_split"), ("test", "test_split"))},
        "sorted_split_ids_sha256": {s: sha256_json(sorted(ids[s])) for s in ids},
        "caption_table_sha256": {s: sha256_json(sorted(zip(tables[s]["image_id"], tables[s]["caption"])))
                                 for s in ids},
        "counts": evidence["split_sizes"], "caption_counts": evidence["caption_counts"],
    }
    fingerprint["fingerprint_sha256"] = sha256_json(fingerprint)
    report.record("DATASET_GATE", ok, evidence)
    return manifest, ids, fingerprint


def vision_fidelity_gate(report, vision, images):
    x = images.to(DEVICE).float()
    with torch.no_grad():
        reference = vision.vit(x)
        extracted = vision.forward_logits(x)
    diff = float((reference - extracted).abs().max())
    expected_transform = repr(tv_models.ViT_B_16_Weights.IMAGENET1K_V1.transforms())
    evidence = {"max_abs_logit_diff_vs_vit_forward": diff, "transform": repr(IMAGE_TRANSFORM),
                "transform_is_official": repr(IMAGE_TRANSFORM) == expected_transform,
                "weights": str(FrozenVisionBackbone.WEIGHTS), "patch_tokens_shape": list(vision(x).shape),
                "frozen": all(not p.requires_grad for p in vision.parameters()), "eval_mode": not vision.training}
    ok = (diff <= 1e-4 and evidence["transform_is_official"] and evidence["patch_tokens_shape"][1:] == [196, 768]
          and evidence["frozen"] and evidence["eval_mode"])
    report.record("VISION_BACKBONE_FIDELITY_GATE", ok, evidence)


def roberta_fidelity_gate(report, language, input_ids, attention_mask):
    ids, mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    reference = AutoModel.from_pretrained("roberta-base").to(DEVICE).eval()   # default class, with pooler
    ours_state = language.roberta.state_dict()
    ref_state = reference.state_dict()
    mismatched = [k for k, v in ours_state.items() if k not in ref_state or not torch.equal(v, ref_state[k].to(v.device))]
    with torch.no_grad():
        a = language(ids, mask).float()
        b = reference(input_ids=ids, attention_mask=mask).last_hidden_state.float()
    diff = float((a - b).abs().max())
    evidence = {"class": type(language.roberta).__name__, "has_pooler": getattr(language.roberta, "pooler", None) is not None,
                "encoder_tensors_differing_from_pretrained": mismatched[:10], "n_encoder_tensors": len(ours_state),
                "max_abs_last_hidden_state_diff_vs_default_model": diff, "tokens_shape": list(ids.shape),
                "max_length_64": ids.shape[1] == 64, "every_caption_has_tokens": bool((mask.sum(1) > 0).all()),
                "frozen": all(not p.requires_grad for p in language.parameters()), "eval_mode": not language.training,
                "output_used": "last_hidden_state (token sequence); pooling done by the model after Mamba-2"}
    del reference
    ok = (not mismatched and diff <= 1e-4 and not evidence["has_pooler"] and evidence["max_length_64"]
          and evidence["every_caption_has_tokens"] and evidence["frozen"] and evidence["eval_mode"])
    report.record("ROBERTA_BACKBONE_FIDELITY_GATE", ok, evidence)


def native_mamba2_gate(report, model, grads):
    found = {n: m for n, m in model.named_modules() if isinstance(m, Mamba2)}
    expected = {"mamba2_img.mamba", "mamba2_txt.mamba"}
    ctor = {n: {k: getattr(m, k, None) for k in ("d_model", "d_state", "d_conv", "expand", "headdim")}
            for n, m in found.items()}
    params = {n: p for n, p in model.named_parameters() if n.startswith(("mamba2_img.mamba.", "mamba2_txt.mamba."))}
    grad_ok = {n: (n in grads and bool(torch.isfinite(grads[n]).all()) and float(grads[n].abs().sum()) > 0)
               for n in params}
    evidence = {
        "module_class": f"{Mamba2.__module__}.{Mamba2.__name__}", "native_modules": sorted(found),
        "constructor": ctor, "NATIVE_MAMBA2_MODULE_COUNT": len(found),
        "NATIVE_MAMBA2_PARAMETER_COUNT": sum(p.numel() for p in params.values()),
        "devices": sorted({str(p.device) for p in params.values()}),
        "params_without_finite_nonzero_grad": [n for n, ok in grad_ok.items() if not ok],
    }
    print("NATIVE_MAMBA2_MODULE_COUNT:", evidence["NATIVE_MAMBA2_MODULE_COUNT"], flush=True)
    print("NATIVE_MAMBA2_PARAMETER_COUNT:", evidence["NATIVE_MAMBA2_PARAMETER_COUNT"], flush=True)
    want = {"d_model": 128, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 64}
    ok = (Mamba2.__module__ == "mamba_ssm.modules.mamba2" and set(found) == expected
          and all(c == want for c in ctor.values())
          and (DEVICE.type != "cuda" or evidence["devices"] == [str(next(iter(params.values())).device)])
          and not evidence["params_without_finite_nonzero_grad"])
    print("NATIVE_MAMBA2_GRADIENT_GATE:", "PASS" if not evidence["params_without_finite_nonzero_grad"] else "FAIL")
    report.record("NATIVE_MAMBA2_GATE", ok, evidence)


def independent_multipositive_loss(z_img, z_txt, image_ids, scale):
    """Written independently of SymmetricMultiPositiveInfoNCELoss (explicit logsumexp loops)."""
    unique = list(dict.fromkeys(image_ids))
    rows = [image_ids.index(u) for u in unique]
    sim = (z_img[rows].double() @ z_txt.double().T) * float(scale)
    i2t, t2i = [], []
    for u, iid in enumerate(unique):
        pos = [c for c, cid in enumerate(image_ids) if cid == iid]
        lse = torch.logsumexp(sim[u], 0)
        i2t.append(-torch.stack([sim[u, c] - lse for c in pos]).mean())
    for c, cid in enumerate(image_ids):
        t2i.append(-(sim[unique.index(cid), c] - torch.logsumexp(sim[:, c], 0)))
    return 0.5 * (torch.stack(i2t).mean() + torch.stack(t2i).mean()), sim.shape


def geometry_and_loss_gates(report, first_ids):
    counts = defaultdict(int)
    for iid in first_ids:
        counts[iid] += 1
    unique = list(dict.fromkeys(first_ids))
    pos = torch.tensor([[u == c for c in first_ids] for u in unique], dtype=torch.float32)
    evidence = {"captions": len(first_ids), "unique_images": len(unique), "similarity_shape": list(pos.shape),
                "positives_per_image": pos.sum(1).int().tolist(), "positives_per_caption": sorted(set(pos.sum(0).int().tolist()))}
    ok = (len(first_ids) == 40 and len(unique) == 8 and list(pos.shape) == [8, 40]
          and evidence["positives_per_image"] == [5] * 8 and evidence["positives_per_caption"] == [1])
    report.record("MULTIPOSITIVE_GEOMETRY_GATE", ok, evidence)

    g = torch.Generator().manual_seed(0)
    zi = F.normalize(torch.randn(40, 128, generator=g), dim=-1).to(DEVICE)
    zt = F.normalize(torch.randn(40, 128, generator=g), dim=-1).to(DEVICE)
    scale = torch.tensor(14.3, device=DEVICE)
    was = MONITOR.active
    MONITOR.active = False
    ours = float(loss_fn(zi, zt, first_ids, scale))
    aligned = F.normalize(torch.randn(8, 128, generator=g), dim=-1).to(DEVICE)
    owner = [unique.index(c) for c in first_ids]
    zi_aligned = aligned[owner]
    ours_aligned = float(loss_fn(zi_aligned, aligned[owner], first_ids, torch.tensor(100.0, device=DEVICE)))
    MONITOR.active = was
    ref, shape = independent_multipositive_loss(zi.cpu(), zt.cpu(), first_ids, 14.3)
    evidence = {"runner_loss": ours, "independent_loss": float(ref), "abs_diff": abs(ours - float(ref)),
                "independent_similarity_shape": list(shape), "perfect_alignment_loss": ours_aligned,
                "expected_floor_half_ln5": 0.5 * math.log(5), "finite": math.isfinite(ours)}
    ok = (evidence["abs_diff"] < 1e-4 and list(shape) == [8, 40] and evidence["finite"]
          and abs(ours_aligned - 0.5 * math.log(5)) < 1e-2)
    report.record("LOSS_GATE", ok, evidence)


def retrieval_evaluator_gate(report):
    """Handcrafted 2-image / 10-caption case with analytically known ranks."""
    e0, e1 = np.zeros(128, np.float32), np.zeros(128, np.float32)
    e0[0], e1[1] = 1.0, 1.0
    coeffs = [(0.5, 0.1), (0.4, 0.2), (0.3, 0.25), (0.2, 0.4), (0.1, 0.5),
              (0.9, 0.0), (0.8, 0.05), (0.05, 0.9), (0.02, 0.8), (0.01, 0.7)]
    extracted = {"image_embeddings": np.stack([e0, e1]),
                 "text_embeddings": np.stack([a * e0 + b * e1 for a, b in coeffs]),
                 "image_ids": ["img0", "img1"], "caption_image_ids": ["img0"] * 5 + ["img1"] * 5}
    # i2t best-positive ranks (0-based): img0 -> 2 (two img1 captions score higher), img1 -> 0
    # t2i ranks: captions 3, 4, 5, 6 rank the wrong image first -> rank 1; the others rank 0
    expected = {"i2t_r1": 50.0, "i2t_r5": 100.0, "i2t_r10": 100.0, "i2t_medr": 2.0, "i2t_meanr": 2.0,
                "t2i_r1": 60.0, "t2i_r5": 100.0, "t2i_r10": 100.0, "t2i_medr": 1.0, "t2i_meanr": 1.4,
                "mean_recall": 85.0}
    got = compute_retrieval_metrics(extracted)
    diffs = {k: abs(got[k] - v) for k, v in expected.items()}
    source = inspect.getsource(extract_embeddings)
    evidence = {"expected": expected, "got": got, "max_abs_diff": max(diffs.values()),
                "full_split_shape_asserts_present": "(1000, 128)" in source and "(5000, 128)" in source,
                "i2t_rule": "rank of best-ranked of the 5 ground-truth captions",
                "evaluator_uses_posterior_mean_by_default": "sample_posterior=False" in source}
    ok = evidence["max_abs_diff"] < 1e-9 and evidence["full_split_shape_asserts_present"]
    report.record("RETRIEVAL_EVALUATOR_GATE", ok, evidence)


def hedo_gate(report, model, grads):
    if not model.use_hedo:
        report.record("HEDO_GATE", True, {"note": "configuration without HEDO"})
        return
    evidence = {}
    for side in ("img", "txt"):
        mod = getattr(model, f"hedo_{side}")
        gamma = torch.sigmoid(mod.gamma.detach())
        names = [n for n, _ in model.named_parameters() if n.startswith(f"hedo_{side}.")]
        evidence[side] = {
            "HEDO_FINITE_GATE": all(bool(torch.isfinite(p).all()) for p in mod.parameters()),
            "HEDO_PARAMETER_RANGE_GATE": {"gamma_min": float(gamma.min()), "gamma_max": float(gamma.max()),
                                          "in_open_unit_interval": bool((gamma > 0).all() and (gamma < 1).all())},
            "HEDO_GRADIENT_GATE": [n for n in names if not (n in grads and bool(torch.isfinite(grads[n]).all())
                                                            and float(grads[n].abs().sum()) > 0)],
            "HEDO_ENERGY_STATISTICS": {k: float(v) for k, v in mod.last_stats.items()},
        }
    ok = all(e["HEDO_FINITE_GATE"] and e["HEDO_PARAMETER_RANGE_GATE"]["in_open_unit_interval"]
             and not e["HEDO_GRADIENT_GATE"] for e in evidence.values())
    evidence["claim_note"] = ("Descriptive energy statistics only. The update is not symplectic and no "
                              "monotonic-energy property is claimed; frac_tokens_energy_increase may be > 0.")
    report.record("HEDO_GATE", ok, evidence)


def hvsc_gate(report, model, image_features, text_features, mask, grads):
    if not model.use_hvsc:
        report.record("HVSC_GATE", True, {"note": "configuration without HVSC"})
        return
    h = model.hvsc
    with torch.no_grad():
        was = model.training
        model.eval()
        _, bimg, bmask_img = model.mamba2_img(model.hedo_img(model.img_proj(image_features)) if model.use_hedo
                                               else model.img_proj(image_features), return_boundary_states=True)
        txt_in = model.hedo_txt(model.txt_proj(text_features)) if model.use_hedo else model.txt_proj(text_features)
        _, btxt, bmask_txt = model.mamba2_txt(txt_in, mask=mask, return_boundary_states=True)
        pooled = h.pool_boundary_states(bimg, bmask_img)
        mu, logvar = h._posterior(h.img_mu, h.img_logvar, pooled)
        std = torch.exp(0.5 * logvar).clamp(STD_MIN, STD_MAX)
        torch.manual_seed(1234)
        draws = torch.stack([h._sample(mu, logvar, True) for _ in range(20)])
        eps_hat = ((draws - mu) / std).flatten()
        mean_path = h._sample(mu, logvar, False)
        _, _, kl = model(image_features, text_features, text_mask=mask, sample_posterior=False)
        model.train(was)
    names = [n for n, _ in model.named_parameters() if n.startswith("hvsc")]
    evidence = {
        "HVSC_MU_FINITE": bool(torch.isfinite(mu).all()), "HVSC_LOGVAR_FINITE": bool(torch.isfinite(logvar).all()),
        "logvar_within_bounds": bool((logvar >= LOGVAR_MIN).all() and (logvar <= LOGVAR_MAX).all()),
        "HVSC_STD_FINITE": bool(torch.isfinite(std).all()),
        "HVSC_KL_FINITE": bool(torch.isfinite(kl)), "HVSC_KL_NONNEGATIVE": float(kl) >= -KL_NEG_TOL, "kl_raw": float(kl),
        "HVSC_REPARAMETERIZATION_GATE": {"eps_mean": float(eps_hat.mean()), "eps_std": float(eps_hat.std()),
                                          "n": int(eps_hat.numel()),
                                          "mean_path_equals_mu": bool(torch.equal(mean_path, mu))},
        "params_without_finite_nonzero_grad": [n for n in names if not (n in grads and bool(torch.isfinite(grads[n]).all())
                                                                         and float(grads[n].abs().sum()) > 0)],
        "posterior": "diagonal Gaussian per modality from mask-weighted mean of chunk boundary states",
        "retrieval_embedding": "normalize(head(LayerNorm(h_pool + 0.1 W_out z))); z = mu at inference and, "
                               "by default (V6.1), during training",
    }
    rp = evidence["HVSC_REPARAMETERIZATION_GATE"]
    ok = (evidence["HVSC_MU_FINITE"] and evidence["HVSC_LOGVAR_FINITE"] and evidence["logvar_within_bounds"]
          and evidence["HVSC_STD_FINITE"] and evidence["HVSC_KL_FINITE"] and evidence["HVSC_KL_NONNEGATIVE"]
          and abs(rp["eps_mean"]) < 0.1 and abs(rp["eps_std"] - 1.0) < 0.1 and rp["mean_path_equals_mu"]
          and not evidence["params_without_finite_nonzero_grad"])
    report.record("HVSC_GATE", ok, evidence)


def seed_reproducibility_gate(report, seed, config, image_features, text_features, mask, train_dataset):
    def build():
        set_all_seeds(seed)
        return HEDO_HVSC_Model(**config).to(DEVICE)

    m1, m2 = build(), build()
    s1, s2 = m1.state_dict(), m2.state_dict()
    init_equal = s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1)
    with torch.no_grad():
        m1.eval(), m2.eval()
        a = m1(image_features, text_features, text_mask=mask, sample_posterior=False)[0]
        b = m2(image_features, text_features, text_mask=mask, sample_posterior=False)[0]

    def first_batches():
        set_all_seeds(seed)
        sampler = AtomicGroupedBatchSampler(train_dataset.df, batch_size=BATCH_SIZE,
                                            captions_per_image=CAPTIONS_PER_IMAGE, shuffle=True)
        it = iter(sampler)
        return [next(it) for _ in range(3)]

    evidence = {"init_state_bitwise_equal": init_equal, "forward_max_abs_diff": float((a - b).abs().max()),
                "batch_order_equal": first_batches() == first_batches(),
                "seeds_set": ["python random", "numpy", "torch cpu", "torch cuda (all devices)"],
                "cudnn": {"deterministic": torch.backends.cudnn.deterministic, "benchmark": torch.backends.cudnn.benchmark},
                "note": "Mamba-2 Triton backward kernels are not bit-deterministic; training runs are "
                        "seed-controlled, not bit-reproducible. Variation is reported as seed variance."}
    del m1, m2
    ok = evidence["init_state_bitwise_equal"] and evidence["forward_max_abs_diff"] == 0.0 and evidence["batch_order_equal"]
    report.record("SEED_REPRODUCIBILITY_GATE", ok, evidence)


def ablation_gate(report, seed):
    shapes = {}
    for name, cfg in CONFIGS.items():
        set_all_seeds(seed)
        m = HEDO_HVSC_Model(**cfg)
        shapes[name] = {n: tuple(p.shape) for n, p in m.named_parameters()}
        del m
    base = shapes["baseline"]
    evidence = {"shared_hyperparameters_sha256": sha256_json(shared_hyperparameters()), "configs": CONFIGS}
    ok = True
    for name, s in shapes.items():
        extra = sorted(set(s) - set(base))
        missing = sorted(set(base) - set(s))
        changed = sorted(n for n in base if n in s and s[n] != base[n])
        allowed_prefix = tuple(p for p, on in (("hedo_", CONFIGS[name]["use_hedo"]), ("hvsc", CONFIGS[name]["use_hvsc"])) if on)
        illegal = [n for n in extra if not n.startswith(allowed_prefix)] if allowed_prefix else extra
        evidence[name] = {"extra_params": len(extra), "missing_vs_baseline": missing, "shape_changes": changed,
                          "illegal_extra": illegal[:5],
                          "params": sum(int(np.prod(v)) for v in s.values())}
        ok &= not missing and not changed and not illegal
    report.record("ABLATION_GATE", ok, evidence)


def module_manifest(model, vision, language, image_features, text_features, mask, param_groups):
    rows, hooks = {}, []
    names = ["img_proj", "txt_proj", "hedo_img", "hedo_txt", "mamba2_img", "mamba2_img.mamba", "mamba2_txt",
             "mamba2_txt.mamba", "hvsc", "hvsc_out_img", "hvsc_out_txt", "hvsc_norm_img", "hvsc_norm_txt",
             "head_img", "head_txt"]
    modules = dict(model.named_modules())

    def describe(t):
        if torch.is_tensor(t):
            return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
        if isinstance(t, (tuple, list)):
            return [describe(x) for x in t]
        return str(type(t).__name__)

    for name in names:
        if name in modules:
            def hook(mod, inp, out, name=name):
                rows[name] = {"class": f"{type(mod).__module__}.{type(mod).__name__}", "input": describe(inp),
                              "output": describe(out)}
            hooks.append(modules[name].register_forward_hook(hook))
    with torch.no_grad():
        model(image_features, text_features, text_mask=mask, sample_posterior=False)
    for h in hooks:
        h.remove()
    group_of = {n: g["name"] for g in param_groups for n in g["param_names"]}
    for name in rows:
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(name + ".")]
        rows[name].update(parameters=sum(p.numel() for _, p in params),
                          trainable=all(p.requires_grad for _, p in params) if params else None,
                          optimizer_groups=sorted({group_of.get(n) for n, _ in params}))
    rows["vision_backbone"] = {"class": "torchvision.models.vision_transformer.VisionTransformer (IMAGENET1K_V1)",
                               "parameters": sum(p.numel() for p in vision.parameters()), "trainable": False,
                               "output": "[B, 196, 768] float32 (FP16 autocast)"}
    rows["text_backbone"] = {"class": f"{type(language.roberta).__module__}.{type(language.roberta).__name__}",
                             "parameters": sum(p.numel() for p in language.parameters()), "trainable": False,
                             "output": "[B, 64, 768] float32 (FP16 autocast)"}
    return rows


def efficiency_profile(model, vision, language, batch, epoch_seconds, peak_train_mb):
    image_features, text_features, mask = get_features(batch, vision, language)
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)

    def timed(fn, iters=20, warmup=5):
        for _ in range(warmup):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        return (time.perf_counter() - t0) * 1000 / iters

    model.eval()
    with torch.no_grad():
        head_ms = timed(lambda: model(image_features, text_features, text_mask=mask, sample_posterior=False))
        e2e_ms = timed(lambda: model(*get_features(batch, vision, language)[:2], text_mask=mask, sample_posterior=False),
                       iters=5, warmup=2)
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(image_features[:1], text_features[:1], text_mask=mask[:1], sample_posterior=False)
        flops = int(counter.get_total_flops())
    except Exception as exc:  # recorded, not hidden
        flops = f"unavailable: {exc!r}"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in vision.parameters()) + sum(p.numel() for p in language.parameters())
    n = image_features.shape[0]
    return {"trainable_params": trainable, "frozen_backbone_params": frozen, "total_params": trainable + frozen,
            "peak_train_vram_mb": peak_train_mb, "train_seconds_per_epoch": epoch_seconds,
            "train_seconds_total": float(sum(epoch_seconds)),
            "inference_head_ms_per_40_pairs": head_ms, "inference_e2e_ms_per_40_pairs": e2e_ms,
            "throughput_pairs_per_s_head": n / (head_ms / 1000), "throughput_pairs_per_s_e2e": n / (e2e_ms / 1000),
            "flops_head_bs1": flops,
            "flops_note": "torch.utils.flop_counter; custom Mamba-2 Triton/CUDA kernels are not counted (lower bound)",
            "device": str(DEVICE)}


def save_checkpoint(path, model, optimizer, scheduler, epoch, seed, config, meta, val_metrics, train_metrics):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "seed": seed, "config": config,
                "methodology_hash": meta["methodology_hash"], "runner_sha256": meta["runner_sha256"],
                "dataset_fingerprint": meta["dataset_fingerprint_sha256"], "validation_metrics": val_metrics,
                "training_metrics": train_metrics, "git_commit": meta.get("git_commit")}, path)


def load_checkpoint_into(path, model):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def prepare_run_dir(seed, config_name, mhash):
    run_dir = RESULTS_ROOT / f"seed{seed}" / config_name
    status = run_dir / "run_status.json"
    if status.is_file():
        s = json.load(open(status, encoding="utf-8"))
        if s.get("status") == "COMPLETED" and s.get("methodology_hash") == mhash:
            return run_dir, True
    if run_dir.exists() and any(run_dir.iterdir()):
        archived = run_dir.with_name(f"{config_name}__superseded_{time.strftime('%Y%m%d-%H%M%S')}")
        run_dir.rename(archived)   # never overwrite or delete a previous attempt
        print(f"previous attempt preserved at {archived}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, False


def git_commit():
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ==============================================================================
# RUN
# ==============================================================================

def main():
    mode = os.environ.get("HEDO_MODE", "train")
    seed = int(os.environ.get("HEDO_SEED", "42"))
    config_name = os.environ.get("HEDO_CONFIG", "full_hedo_hvsc")
    if config_name not in CONFIGS:
        raise SystemExit(f"unknown HEDO_CONFIG {config_name}; choose from {sorted(CONFIGS)}")
    config = CONFIGS[config_name]
    mhash = methodology_hash()
    meta = {"mode": mode, "seed": seed, "config_name": config_name, "config": config, "methodology_hash": mhash,
            "runner_sha256": sha256_path(Path(__file__).resolve()), "git_commit": git_commit(),
            "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu (debug only, not evidence)",
            "mamba_ssm": package_version("mamba_ssm"), "causal_conv1d": package_version("causal_conv1d"),
            "transformers": package_version("transformers"), "torchvision": package_version("torchvision"),
            "shared_hyperparameters": shared_hyperparameters()}
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if mode == "gates":
        run_dir, done = RESULTS_ROOT / "gates" / mhash[:12], False
        run_dir.mkdir(parents=True, exist_ok=True)
    elif mode == "train":
        go = RESULTS_ROOT / "GO_NO_GO.json"
        decision = json.load(open(go, encoding="utf-8")) if go.is_file() else {}
        if decision.get("decision") != "GO" or decision.get("methodology_hash") != mhash:
            print("NO-GO: run HEDO_MODE=gates for this exact methodology first "
                  f"(found {decision.get('decision')} for {str(decision.get('methodology_hash'))[:12]}, "
                  f"need GO for {mhash[:12]})", flush=True)
            sys.exit(3)
        run_dir, done = prepare_run_dir(seed, config_name, mhash)
        if done:
            print(f"SKIP: {run_dir} already COMPLETED for methodology {mhash[:12]}", flush=True)
            return
    else:
        raise SystemExit(f"unknown HEDO_MODE {mode}")

    print("=" * 80, flush=True)
    print(f"MODE={mode} CONFIG={config_name} SEED={seed} METHODOLOGY={mhash[:16]}", flush=True)
    print("=" * 80, flush=True)
    for k, v in meta.items():
        if k != "shared_hyperparameters":
            print(f"  {k:<18}: {v}", flush=True)
    write_json(run_dir / "config.json", meta)
    report = GateReport()

    try:
        RUN_STATE.update(stage="setup")
        compile(Path(__file__).resolve().read_text(encoding="utf-8"), str(Path(__file__).resolve()), "exec")
        set_all_seeds(seed)
        image_dir, tables = load_flickr8k()
        manifest, split_ids, fingerprint = dataset_gate(report, tables)
        meta["dataset_fingerprint_sha256"] = fingerprint["fingerprint_sha256"]
        write_json(run_dir / "dataset_fingerprint.json", fingerprint)
        write_json(run_dir / "dataset_manifest.json", {k: manifest[k] for k in manifest})

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        vision = FrozenVisionBackbone().to(DEVICE)
        language = FrozenLanguageBackbone().to(DEVICE)

        train_dataset = FlickrDataset(tables["train"], image_dir, tokenizer)
        val_dataset = FlickrDataset(tables["val"], image_dir, tokenizer)
        test_dataset = FlickrDataset(tables["test"], image_dir, tokenizer)
        workers = int(os.environ.get("HEDO_NUM_WORKERS", "2"))
        set_all_seeds(seed)
        train_loader = DataLoader(train_dataset, batch_sampler=AtomicGroupedBatchSampler(
            train_dataset.df, batch_size=BATCH_SIZE, captions_per_image=CAPTIONS_PER_IMAGE, shuffle=True),
            num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers,
                                 pin_memory=True)
        first_batch = next(iter(train_loader))
        first_ids = [str(x) for x in first_batch["image_id"]]

        RUN_STATE.update(stage="gates")
        if mode == "gates" or os.environ.get("HEDO_BACKBONE_GATES_IN_TRAIN", "1") == "1":
            vision_fidelity_gate(report, vision, first_batch["image"][:4])
            roberta_fidelity_gate(report, language, first_batch["input_ids"][:8], first_batch["attention_mask"][:8])
        geometry_and_loss_gates(report, first_ids)
        retrieval_evaluator_gate(report)

        set_all_seeds(seed)
        model = HEDO_HVSC_Model(**config).to(DEVICE)
        initial_state = copy.deepcopy(model.state_dict())

        MONITOR.active = True
        smoke = copy.deepcopy(model)
        smoke_opt, _ = build_optimizer(smoke)
        smoke.train()
        _, grads = training_step(smoke, first_batch, vision, language, smoke_opt, context="real-batch gate",
                                 verbose=True, keep_grads=True)
        MONITOR.active = False
        feats = get_features(first_batch, vision, language)
        native_mamba2_gate(report, smoke, grads)
        hedo_gate(report, smoke, grads)
        hvsc_gate(report, smoke, *feats, grads)
        seed_reproducibility_gate(report, seed, config, *feats, train_dataset)
        ablation_gate(report, seed)
        del smoke, smoke_opt, grads

        optimizer, param_groups = build_optimizer(model, verbose=True)
        write_json(run_dir / "methodology_manifest.json", {
            **meta, "modules": module_manifest(model, vision, language, *feats, param_groups),
            "optimizer_groups": {g["name"]: {"params": sum(p.numel() for p in g["params"]), "lr": LEARNING_RATE,
                                             "weight_decay": WEIGHT_DECAY} for g in param_groups},
            "data_split": {"train": "official Flickr_8k.trainImages.txt (gradient updates only)",
                           "val": "official Flickr_8k.devImages.txt (checkpoint selection only, no_grad)",
                           "test": "official Flickr_8k.testImages.txt (evaluated once, best checkpoint)"},
            "checkpoint_policy": BEST_CHECKPOINT_SELECTION_CRITERION + "; last checkpoint retained",
            "inference_policy": shared_hyperparameters()["inference_policy"]})

        if mode == "gates":
            RUN_STATE.update(stage="stability_epoch")
            MONITOR.active = True
            stab = HEDO_HVSC_Model(**config).to(DEVICE)
            stab.load_state_dict(initial_state)
            stab_opt, _ = build_optimizer(stab)
            stab_sched = torch.optim.lr_scheduler.CosineAnnealingLR(stab_opt, T_max=max(1, len(train_loader) * EPOCHS))
            stab_train = train_single_epoch(stab, train_loader, vision, language, stab_opt, stab_sched, epoch=1)
            MONITOR.active = False
            _, stab_val = evaluate_split(stab, val_loader, vision, language, split_ids["val"], "stability validation")
            report.record("NUMERICAL_STABILITY_GATE", True,
                          {"EPOCH_1_FINITE": "PASS", "train": stab_train, "val_mean_recall": stab_val["mean_recall"]})
            print("EPOCH_1_FINITE: PASS", flush=True)

            RUN_STATE.update(stage="train_inference_consistency")
            consistency = train_inference_consistency(stab, val_loader, vision, language, split_ids["val"], stab_val)
            report.record("TRAIN_INFERENCE_CONSISTENCY_GATE", consistency["pass"], consistency)

            RUN_STATE.update(stage="checkpoint_gate")
            ckpt_path = run_dir / "gate_checkpoint.pt"
            save_checkpoint(ckpt_path, stab, stab_opt, stab_sched, 1, seed, config, meta, stab_val, stab_train)
            reloaded = HEDO_HVSC_Model(**config).to(DEVICE)
            ckpt = load_checkpoint_into(ckpt_path, reloaded)
            _, reload_val = evaluate_split(reloaded, val_loader, vision, language, split_ids["val"], "checkpoint reload")
            required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "seed", "config",
                        "methodology_hash", "dataset_fingerprint", "validation_metrics", "training_metrics"}
            ev = {"missing_keys": sorted(required - set(ckpt)),
                  "max_abs_metric_diff_after_reload": max(abs(reload_val[k] - stab_val[k]) for k in stab_val)}
            report.record("CHECKPOINT_GATE", not ev["missing_keys"] and ev["max_abs_metric_diff_after_reload"] < 1e-9, ev)

            decision = {"decision": "GO", "methodology_hash": mhash, "gates": report.gates, "meta": meta,
                        "note": "GO means the pipeline is valid to run; it is not a result."}
            write_json(run_dir / "gate_report.json", decision)
            write_json(RESULTS_ROOT / "GO_NO_GO.json", decision)
            (RESULTS_ROOT / "final_methodology_hash.txt").write_text(mhash + "\n", encoding="utf-8")
            print("ALL_CRITICAL_GATES: PASS -> GO", flush=True)
            return

        # --------------------------------------------------------------- training
        write_json(run_dir / "gate_report.json", {"gates": report.gates, "methodology_hash": mhash})
        RUN_STATE.update(stage="training")
        set_all_seeds(seed)
        model.load_state_dict(initial_state)
        optimizer, param_groups = build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(train_loader) * EPOCHS))
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        MONITOR.active = True
        history, epoch_seconds, best_mr, best_epoch = [], [], -1.0, 0
        with open(run_dir / "training_log.jsonl", "w", encoding="utf-8") as log:
            for epoch in range(1, EPOCHS + 1):
                t0 = time.time()
                lr_start = optimizer.param_groups[0]["lr"]
                tr = train_single_epoch(model, train_loader, vision, language, optimizer, scheduler, epoch, log)
                RUN_STATE.update(batch=None)
                MONITOR.active = False
                _, val = evaluate_split(model, val_loader, vision, language, split_ids["val"], f"epoch {epoch} validation")
                MONITOR.active = True
                epoch_seconds.append(time.time() - t0)
                row = {"epoch": epoch, "lr_start": lr_start, "train_total_loss": tr["loss"], "infonce": tr["infonce"],
                       "kl_raw": tr["kl_raw"], "kl_weighted": tr["kl_weighted"], "grad_norm_mean": tr["grad_norm"],
                       "grad_norm_max": tr["grad_norm_max"], "logit_scale": tr["logit_scale_end"],
                       "lr_end": tr["lr_end"], **{f"val_{k}": v for k, v in val.items()}, "seconds": epoch_seconds[-1],
                       **{k: v for k, v in tr.items() if k.startswith(("hvsc_", "hedo_"))}}
                history.append(row)
                pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
                print("=" * 60, flush=True)
                print(f"EPOCH {epoch:02d}/{EPOCHS}", flush=True)
                print("=" * 60, flush=True)
                for key, fmt in (("train_total_loss", ".5f"), ("infonce", ".5f"), ("kl_raw", ".5f"),
                                 ("kl_weighted", ".6f"), ("grad_norm_mean", ".4f"), ("logit_scale", ".4f"),
                                 ("lr_start", ".3e"), ("val_i2t_r1", ".2f"), ("val_t2i_r1", ".2f"),
                                 ("val_mean_recall", ".3f")):
                    print(f"{key}={row[key]:{fmt}}", flush=True)
                save_checkpoint(run_dir / "checkpoint_last.pt", model, optimizer, scheduler, epoch, seed, config, meta,
                                val, tr)
                if val["mean_recall"] > best_mr:
                    best_mr, best_epoch = val["mean_recall"], epoch
                    save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, scheduler, epoch, seed, config,
                                    meta, val, tr)
                    print(f"  checkpoint_best.pt <- epoch {epoch} (validation mean recall {best_mr:.3f})", flush=True)
        MONITOR.active = False
        peak_mb = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else None

        RUN_STATE.update(stage="checkpoint_and_consistency", epoch=None)
        ckpt = load_checkpoint_into(run_dir / "checkpoint_best.pt", model)
        _, reval = evaluate_split(model, val_loader, vision, language, split_ids["val"], "best checkpoint reload")
        if max(abs(reval[k] - ckpt["validation_metrics"][k]) for k in reval) >= 1e-9:
            raise GateFailure("CHECKPOINT_GATE", "reloaded best checkpoint does not reproduce its validation metrics")
        consistency = train_inference_consistency(model, val_loader, vision, language, split_ids["val"], reval)
        if not consistency["pass"]:
            raise GateFailure("TRAIN_INFERENCE_CONSISTENCY_GATE", json.dumps(consistency))

        RUN_STATE.update(stage="test")
        test_emb, test = evaluate_split(model, test_loader, vision, language, split_ids["test"], "test")
        np.save(run_dir / "test_image_embeddings.npy", test_emb["image_embeddings"])
        np.save(run_dir / "test_text_embeddings.npy", test_emb["text_embeddings"])
        write_json(run_dir / "test_ids.json", {"image_ids": test_emb["image_ids"],
                                               "caption_image_ids": test_emb["caption_image_ids"]})
        efficiency = efficiency_profile(model, vision, language, first_batch, epoch_seconds, peak_mb)
        write_json(run_dir / "efficiency.json", efficiency)
        write_json(run_dir / "test_results.json", {**test, "TEST_EVALUATED_ON": str(run_dir / "checkpoint_best.pt"),
                                                   "best_epoch": best_epoch, "best_val_mean_recall": best_mr,
                                                   "BEST_CHECKPOINT_SELECTION_CRITERION": BEST_CHECKPOINT_SELECTION_CRITERION,
                                                   "train_inference_consistency": consistency})
        write_json(run_dir / "run_status.json", {"status": "COMPLETED", "methodology_hash": mhash, "seed": seed,
                                                 "config": config_name, "best_epoch": best_epoch,
                                                 "TEST_EVALUATED_ON": "checkpoint_best.pt",
                                                 "test_mean_recall": test["mean_recall"], **meta})
        print(f"BEST_CHECKPOINT_SELECTION_CRITERION: {BEST_CHECKPOINT_SELECTION_CRITERION}", flush=True)
        print(f"TEST_EVALUATED_ON: {run_dir / 'checkpoint_best.pt'} (epoch {best_epoch})", flush=True)
        for k, v in test.items():
            print(f"test_{k}={v:.4f}", flush=True)
        print("NATIVE_TRAINING_RUN_COMPLETE: PASS", flush=True)

    except BaseException as exc:  # noqa: BLE001 - every failure must be loud and recorded
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        MONITOR.active = False
        if isinstance(exc, NonFiniteError):
            failure_type, status = "NON_FINITE_TENSOR", FAILED_STATUS
        elif isinstance(exc, GateFailure):
            failure_type, status = f"GATE_FAILED:{exc.gate}", "FAILED_GATE"
        elif isinstance(exc, AssertionError):
            failure_type, status = "ASSERTION", "FAILED_ASSERTION"
        else:
            failure_type, status = type(exc).__name__, "FAILED_ERROR"
        message = str(exc)
        first_bad = message.split(": ", 1)[1] if isinstance(exc, NonFiniteError) and ": " in message else None
        bad_name = getattr(exc, "tensor_name", first_bad) or ""
        param_name = bad_name.split(" ", 1)[1] if bad_name.startswith(("grad ", "param ")) else None
        status_obj = {"status": status, "failure_type": failure_type, "first_bad_tensor": first_bad,
                      "tensor_statistics": getattr(exc, "tensor_stats", None),
                      "learning_rate": RUN_STATE.get("lr"), "last_finite_grad_norm": RUN_STATE.get("last_grad_norm"),
                      "parameter_group": PARAM_GROUP_OF.get(param_name) if param_name else None,
                      "stage": RUN_STATE["stage"], "epoch": RUN_STATE["epoch"], "batch": RUN_STATE["batch"],
                      "message": message, "traceback": traceback.format_exc(), "configuration": config_name,
                      "seed": seed, "methodology_hash": mhash, "gates_so_far": report.gates}
        write_json(run_dir / "run_status.json", status_obj)
        if mode == "gates":
            write_json(RESULTS_ROOT / "GO_NO_GO.json", {"decision": "NO-GO", "methodology_hash": mhash,
                                                        "failure": status_obj})
        print("=" * 80, flush=True)
        print(f"RUN FAILED: {status} / {failure_type} at stage={RUN_STATE['stage']} epoch={RUN_STATE['epoch']} "
              f"batch={RUN_STATE['batch']}", flush=True)
        print(message[:2000], flush=True)
        if status == FAILED_STATUS:
            print(f"TRAINING_RUN_STATUS = {FAILED_STATUS}", flush=True)
            print("NATIVE_TRAINING_FAILED_NUMERICAL_STABILITY: PASS", flush=True)
        print("=" * 80, flush=True)
        sys.exit(2 if status == FAILED_STATUS else 4)


def train_inference_consistency(model, val_loader, vision, language, val_ids, mean_metrics):
    """Training uses normalize(head(z)), z = mu + eps*std; retrieval uses normalize(head(mu)).
    Quantified on the full validation split with a fixed noise seed."""
    if not model.use_hvsc:
        return {"pass": True, "note": "no stochastic latent in this configuration"}
    mean_emb, _ = evaluate_split(model, val_loader, vision, language, val_ids, "consistency (inference)")
    train_emb, train_metrics = evaluate_split(model, val_loader, vision, language, val_ids,
                                              "consistency (training policy)",
                                              sample_posterior=TRAIN_SAMPLE_POSTERIOR, noise_seed=2024)
    sampled_emb, sampled_metrics = evaluate_split(model, val_loader, vision, language, val_ids,
                                                  "diagnostic (sampled latent)", sample_posterior=True,
                                                  noise_seed=2024)

    def cos(a, b, key):
        return float(np.mean(np.sum(a[key] * b[key], axis=1)))

    cos_img = cos(mean_emb, train_emb, "image_embeddings")
    cos_txt = cos(mean_emb, train_emb, "text_embeddings")
    result = {"training_posterior_policy": "sampled" if TRAIN_SAMPLE_POSTERIOR else "posterior mean",
              "mean_cosine_training_vs_inference_image": cos_img, "mean_cosine_training_vs_inference_text": cos_txt,
              "val_mean_recall_inference": mean_metrics["mean_recall"],
              "val_mean_recall_training_policy": train_metrics["mean_recall"],
              "diagnostic_cosine_sampled_vs_mean_image": cos(mean_emb, sampled_emb, "image_embeddings"),
              "diagnostic_cosine_sampled_vs_mean_text": cos(mean_emb, sampled_emb, "text_embeddings"),
              "diagnostic_val_mean_recall_sampled": sampled_metrics["mean_recall"],
              "threshold_min_cosine": CONSISTENCY_MIN_COSINE}
    result["pass"] = min(cos_img, cos_txt) >= CONSISTENCY_MIN_COSINE
    return result


if __name__ == "__main__":

    main()
