
import os
import sys
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

if not torch.cuda.is_available():

    raise RuntimeError(
        "CRITICAL: CUDA is unavailable in the native Python environment."
    )

DEVICE = torch.device("cuda")


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
print("GPU          :", torch.cuda.get_device_name(0))

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
# HEDO
# ==============================================================================

class HEDO(nn.Module):

    def __init__(
        self,
        d_model=128,
    ):

        super().__init__()

        self.q_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.p_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.gamma = nn.Parameter(
            torch.zeros(d_model)
        )

        self.output_proj = nn.Linear(
            d_model,
            d_model,
        )

    def forward(self, x):

        q = self.q_proj(x)

        p = self.p_proj(x)

        gamma = torch.sigmoid(
            self.gamma
        ).view(
            1,
            1,
            -1,
        )

        q_new = q + p

        p_new = (
            p
            - gamma * q_new
        )

        return self.output_proj(
            q_new + p_new
        )


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
    ):

        super().__init__()

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

        y = self.mamba(
            x_in
        )

        y = self.norm(
            y
        )

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

        boundary_states = torch.gather(
            y,
            dim=1,
            index=gather_index,
        )

        return (
            y,
            boundary_states,
            boundary_mask,
        )


# ==============================================================================
# HVSC
# ==============================================================================

class ChunkWiseHVSC(nn.Module):

    def __init__(
        self,
        d_state=128,
        d_latent=128,
        logvar_min=-10.0,
        logvar_max=10.0,
    ):

        super().__init__()

        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

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

        mu = mu_layer(h)

        logvar = logvar_layer(
            h
        ).clamp(
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

        var1 = logvar1.exp()
        var2 = logvar2.exp()

        kl = 0.5 * (
            logvar2
            - logvar1
            +
            (
                var1
                +
                (mu1 - mu2).pow(2)
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

        return (
            mu
            +
            eps
            *
            torch.exp(
                0.5 * logvar
            )
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

        symmetric_kl = 0.5 * (
            kl_i2t
            +
            kl_t2i
        )

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
                d_model=embed_dim
            )

            self.hedo_txt = HEDO(
                d_model=embed_dim
            )

        self.mamba2_img = (
            NativeMamba2SequenceBlock(
                d_model=embed_dim,
                d_state=d_state,
                chunk_size=chunk_size,
                d_conv=4,
                expand=2,
                headdim=64,
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
            )
        )

        if use_hvsc:

            self.hvsc = ChunkWiseHVSC(
                d_state=embed_dim,
                d_latent=embed_dim,
            )

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

        if self.use_hvsc:

            z_img, z_txt, symmetric_kl = (
                self.hvsc(
                    boundary_img,
                    boundary_txt,
                    boundary_mask_img,
                    boundary_mask_txt,
                    sample_posterior=sample_posterior,
                )
            )

        else:

            if image_mask is None:

                pooled_img = y_img.mean(
                    dim=1
                )

            else:

                w_img = (
                    image_mask
                    .to(dtype=y_img.dtype)
                    .unsqueeze(-1)
                )

                pooled_img = (
                    y_img * w_img
                ).sum(
                    dim=1
                ) / w_img.sum(
                    dim=1
                ).clamp(
                    min=1.0
                )

            if text_mask is None:

                pooled_txt = y_txt.mean(
                    dim=1
                )

            else:

                w_txt = (
                    text_mask
                    .to(dtype=y_txt.dtype)
                    .unsqueeze(-1)
                )

                pooled_txt = (
                    y_txt * w_txt
                ).sum(
                    dim=1
                ) / w_txt.sum(
                    dim=1
                ).clamp(
                    min=1.0
                )

            z_img = pooled_img
            z_txt = pooled_txt

            symmetric_kl = torch.zeros(
                (),
                device=image_features.device,
            )

        z_img = F.normalize(
            self.head_img(z_img),
            dim=-1,
        )

        z_txt = F.normalize(
            self.head_txt(z_txt),
            dim=-1,
        )

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

        sim = (
            unique_z_img
            @ z_txt.T
        ) * logit_scale.clamp(
            max=100.0
        )

        pos_i2t = torch.tensor(
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

        pos_i2t = (
            pos_i2t
            /
            pos_i2t.sum(
                dim=1,
                keepdim=True,
            ).clamp(
                min=1.0
            )
        )

        loss_i2t = -(
            F.log_softmax(
                sim,
                dim=1,
            )
            *
            pos_i2t
        ).sum(
            dim=1
        ).mean()

        pos_t2i = pos_i2t.T

        pos_t2i = (
            pos_t2i
            /
            pos_t2i.sum(
                dim=1,
                keepdim=True,
            ).clamp(
                min=1.0
            )
        )

        loss_t2i = -(
            F.log_softmax(
                sim.T,
                dim=1,
            )
            *
            pos_t2i
        ).sum(
            dim=1
        ).mean()

        return 0.5 * (
            loss_i2t
            +
            loss_t2i
        )


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

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
KL_WEIGHT = 1e-4
GRAD_CLIP = 1.0

BENCHMARK_VERSION = (
    "HEDO_HVSC_NATIVE_MAMBA2_V6"
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

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


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

class FrozenVisionBackbone(
    nn.Module
):

    def __init__(self):

        super().__init__()

        vit = tv_models.vit_b_16(
            weights=(
                tv_models
                .ViT_B_16_Weights
                .DEFAULT
            )
        )

        self.conv_proj = (
            vit.conv_proj
        )

        self.encoder = (
            vit.encoder
        )

        self.class_token = (
            vit.class_token
        )

        for p in self.parameters():

            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x):

        n = x.shape[0]

        x = self.conv_proj(
            x
        )

        x = x.reshape(
            n,
            768,
            -1,
        ).permute(
            0,
            2,
            1,
        )

        cls = (
            self.class_token
            .expand(
                n,
                -1,
                -1,
            )
        )

        x = torch.cat(
            [
                cls,
                x,
            ],
            dim=1,
        )

        x = self.encoder(
            x
        )

        return x[:, 1:, :]


# ==============================================================================
# FROZEN TEXT BACKBONE
# ==============================================================================

class FrozenLanguageBackbone(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.roberta = (
            AutoModel
            .from_pretrained(
                "roberta-base"
            )
        )

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
):

    model.eval()

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
            sample_posterior=False,
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
# TRAINING
# ==============================================================================

def train_single_epoch(
    model,
    loader,
    vision,
    language,
    optimizer,
    scaler,
    scheduler,
    loss_fn,
):

    model.train()

    running_loss = 0.0
    running_info = 0.0
    running_kl = 0.0

    for batch in loader:

        optimizer.zero_grad(
            set_to_none=True
        )

        image_features, text_features, mask = (
            get_features(
                batch,
                vision,
                language,
            )
        )

        image_ids = [
            str(x)
            for x
            in batch["image_id"]
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            z_img, z_txt, kl = model(
                image_features,
                text_features,
                text_mask=mask,
                sample_posterior=True,
            )

            logit_scale = (
                model.logit_scale.exp()
                .clamp(max=100.0)
            )

            info_loss = loss_fn(
                z_img,
                z_txt,
                image_ids,
                logit_scale,
            )

            total_loss = (
                info_loss
                +
                KL_WEIGHT * kl
            )

        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        scheduler.step()

        running_loss += float(
            total_loss.detach()
            .cpu()
        )

        running_info += float(
            info_loss.detach()
            .cpu()
        )

        running_kl += float(
            kl.detach()
            .cpu()
        )

    n = max(
        1,
        len(loader),
    )

    return {

        "loss":
            running_loss / n,

        "infonce_loss":
            running_info / n,

        "kl_loss":
            running_kl / n,

        "logit_scale":
            float(
                model.logit_scale
                .exp()
                .clamp(max=100.0)
                .detach()
                .cpu()
            ),
    }


# ==============================================================================
# RUN
# ==============================================================================

def main():

    set_all_seeds(
        int(
            os.environ.get(
                "HEDO_SEED",
                "42",
            )
        )
    )

    print()
    print("=" * 80)
    print("LOADING FLICKR8K")
    print("=" * 80)

    image_dir, tables = (
        load_flickr8k()
    )

    print(
        "Train captions:",
        len(tables["train"]),
    )

    print(
        "Val captions  :",
        len(tables["val"]),
    )

    print(
        "Test captions :",
        len(tables["test"]),
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            "roberta-base"
        )
    )

    vision = (
        FrozenVisionBackbone()
        .to(DEVICE)
    )

    language = (
        FrozenLanguageBackbone()
        .to(DEVICE)
    )

    train_dataset = FlickrDataset(
        tables["train"],
        image_dir,
        tokenizer,
    )

    val_dataset = FlickrDataset(
        tables["val"],
        image_dir,
        tokenizer,
    )

    test_dataset = FlickrDataset(
        tables["test"],
        image_dir,
        tokenizer,
    )

    train_sampler = (
        AtomicGroupedBatchSampler(
            train_dataset.df,
            batch_size=BATCH_SIZE,
            captions_per_image=CAPTIONS_PER_IMAGE,
            shuffle=True,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # HARD 8x40 TRAINING GEOMETRY CHECK
    # --------------------------------------------------------------------------

    first_batch = next(
        iter(train_loader)
    )

    first_ids = [
        str(x)
        for x
        in first_batch["image_id"]
    ]

    assert len(first_ids) == 40

    assert (
        len(set(first_ids))
        == 8
    )

    counts = defaultdict(int)

    for iid in first_ids:

        counts[iid] += 1

    assert all(
        value == 5
        for value
        in counts.values()
    )

    print()
    print("=" * 80)
    print("TRAINING GEOMETRY")
    print("=" * 80)

    print(
        "Captions in batch :",
        len(first_ids),
    )

    print(
        "Unique images     :",
        len(set(first_ids)),
    )

    print(
        "Captions/image    :",
        sorted(set(counts.values())),
    )

    print(
        "Similarity geometry: 8 x 40"
    )

    print(
        "MULTIPOSITIVE_BATCH_GATE: PASS"
    )

    # --------------------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------------------

    use_hedo = (
        os.environ.get(
            "HEDO_USE_HEDO",
            "1",
        )
        == "1"
    )

    use_hvsc = (
        os.environ.get(
            "HEDO_USE_HVSC",
            "1",
        )
        == "1"
    )

    model = HEDO_HVSC_Model(
        use_hedo=use_hedo,
        use_hvsc=use_hvsc,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = max(
        1,
        len(train_loader)
        * EPOCHS,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    history = []

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        start = time.time()

        train_metrics = (
            train_single_epoch(
                model,
                train_loader,
                vision,
                language,
                optimizer,
                scaler,
                scheduler,
                loss_fn,
            )
        )

        val_embeddings = (
            extract_embeddings(
                model,
                val_loader,
                vision,
                language,
            )
        )

        val_metrics = (
            compute_retrieval_metrics(
                val_embeddings
            )
        )

        row = {

            "epoch": epoch,

            **train_metrics,

            "val_i2t_r1":
                val_metrics["i2t_r1"],

            "val_i2t_r5":
                val_metrics["i2t_r5"],

            "val_i2t_r10":
                val_metrics["i2t_r10"],

            "val_t2i_r1":
                val_metrics["t2i_r1"],

            "val_t2i_r5":
                val_metrics["t2i_r5"],

            "val_t2i_r10":
                val_metrics["t2i_r10"],

            "val_mean_recall":
                val_metrics["mean_recall"],

            "seconds":
                float(
                    time.time()
                    - start
                ),
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:02d}/{EPOCHS} "
            f"loss={row['loss']:.5f} "
            f"InfoNCE={row['infonce_loss']:.5f} "
            f"KL={row['kl_loss']:.5f} "
            f"ValMR={row['val_mean_recall']:.3f}"
        )

    # --------------------------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------------------------

    test_embeddings = (
        extract_embeddings(
            model,
            test_loader,
            vision,
            language,
        )
    )

    test_metrics = (
        compute_retrieval_metrics(
            test_embeddings
        )
    )

    print()
    print("=" * 80)
    print("FINAL TEST")
    print("=" * 80)

    for key, value in (
        test_metrics.items()
    ):

        print(
            f"{key:15s}: {value:.6f}"
        )

    # --------------------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        BENCHMARK_ROOT
        /
        "training_history.csv",
        index=False,
    )

    with open(
        BENCHMARK_ROOT
        /
        "test_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=2,
        )

    np.save(
        BENCHMARK_ROOT
        /
        "test_image_embeddings.npy",
        test_embeddings[
            "image_embeddings"
        ],
    )

    np.save(
        BENCHMARK_ROOT
        /
        "test_text_embeddings.npy",
        test_embeddings[
            "text_embeddings"
        ],
    )

    torch.save(
        model.state_dict(),
        BENCHMARK_ROOT
        /
        "model_final.pt",
    )

    print()
    print("=" * 80)
    print("NATIVE TRAINING RUN COMPLETE")
    print("=" * 80)

    print(
        "Benchmark root:",
        BENCHMARK_ROOT,
    )

    print(
        "NATIVE_TRAINING: PASS"
    )


if __name__ == "__main__":

    main()
