"""
Standalone CLI Benchmark Runner for HEDO-HVSC 12-Run Factorial Experiment (V5 Clean).
Executes the identical clean pipeline as HEDO_HVSC_Research_Master.ipynb with 12/12 completion gating.
"""
import os
import sys
import math
import time
import json
import random
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import torchvision.models as tv_models
from scipy.stats import linregress

# Paths
WORKSPACE_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKSPACE_DIR, "data", "flickr8k")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "research_outputs", "hedohvsc_v3_final_clean")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")

for d in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, TABLE_DIR, ARTIFACT_DIR, SPLIT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"RUNNING HEDO-HVSC MASTER FACTORIAL BENCHMARK (V5)")
print(f"   Device Selected : {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU Model       : {torch.cuda.get_device_name(0)}")
print("=" * 70)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_all_seeds(42)

# Dataset Discovery
CANDIDATE_PATHS = [
    DATA_DIR,
    os.path.join(WORKSPACE_DIR, "flickr8k"),
    os.path.join(WORKSPACE_DIR, "data"),
    os.path.join(WORKSPACE_DIR, "Flickr8k_Dataset"),
    "/kaggle/input/flickr8k",
    "/kaggle/input/flickr8k-dataset",
    "/kaggle/input/flickr-image-dataset"
]

def locate_flickr8k():
    for base in CANDIDATE_PATHS:
        img_dir = os.path.join(base, "Images") if os.path.isdir(os.path.join(base, "Images")) else os.path.join(base, "Flicker8k_Dataset")
        if not os.path.isdir(img_dir):
            img_dir = base
        
        token_file = os.path.join(base, "Flickr8k.token.txt")
        if not os.path.isfile(token_file):
            token_file = os.path.join(base, "captions.txt")
            
        if os.path.isdir(img_dir) and (os.path.isfile(token_file) or len([f for f in os.listdir(img_dir) if f.endswith('.jpg')]) > 500):
            return img_dir, token_file, base
    return None, None, None

IMG_DIR, TOKEN_FILE, FLICKR_BASE = locate_flickr8k()
if IMG_DIR is None or not os.path.isdir(IMG_DIR):
    raise FileNotFoundError("CRITICAL: Flickr8k dataset was not found in candidate paths.")

# Parse captions
records = []
if os.path.isfile(TOKEN_FILE):
    with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if '\t' in line:
                cap_id, caption = line.split('\t', 1)
            elif ',' in line and not line.startswith('image'):
                cap_id, caption = line.split(',', 1)
            else:
                continue
            img_id = cap_id.split('#')[0] if '#' in cap_id else cap_id
            records.append({"image_id": img_id, "caption_id": cap_id, "caption": caption})
else:
    captions_csv = os.path.join(FLICKR_BASE, "captions.txt")
    if os.path.isfile(captions_csv):
        df_raw = pd.read_csv(captions_csv)
        for idx, row in df_raw.iterrows():
            img_col = "image" if "image" in row else ("image_id" if "image_id" in row else df_raw.columns[0])
            cap_col = "caption" if "caption" in row else df_raw.columns[1]
            img_id = str(row[img_col])
            records.append({"image_id": img_id, "caption_id": f"{img_id}#{idx%5}", "caption": str(row[cap_col])})

DF_PAIRS = pd.DataFrame(records)

# Split checking & downloading
def find_split_file(filename_candidates):
    for base in CANDIDATE_PATHS:
        for fname in filename_candidates:
            p = os.path.join(base, fname)
            if os.path.isfile(p):
                return p
    return None

train_file = find_split_file(["Flickr_8k.trainImages.txt", "Flickr8k.trainImages.txt"])
val_file = find_split_file(["Flickr_8k.devImages.txt", "Flickr8k.devImages.txt"])
test_file = find_split_file(["Flickr_8k.testImages.txt", "Flickr8k.testImages.txt"])

SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/Flickr8k_text/Flickr_8k.trainImages.txt",
    "val": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/Flickr8k_text/Flickr_8k.devImages.txt",
    "test": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/Flickr8k_text/Flickr_8k.testImages.txt"
}

if not (train_file and val_file and test_file):
    print("Fetching canonical Flickr8k split manifest text files...")
    try:
        if not train_file:
            train_file = os.path.join(DATA_DIR, "Flickr_8k.trainImages.txt")
            urllib.request.urlretrieve(SPLIT_URLS["train"], train_file)
        if not val_file:
            val_file = os.path.join(DATA_DIR, "Flickr_8k.devImages.txt")
            urllib.request.urlretrieve(SPLIT_URLS["val"], val_file)
        if not test_file:
            test_file = os.path.join(DATA_DIR, "Flickr_8k.testImages.txt")
            urllib.request.urlretrieve(SPLIT_URLS["test"], test_file)
    except Exception as e:
        print(f"Warning: Online download of split manifests failed ({e}).")

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

split_hashes = {}
for name, path in [("train", train_file), ("val", val_file), ("test", test_file)]:
    dest_path = os.path.join(SPLIT_DIR, f"Flickr_8k.{name}Images.txt")
    shutil.copy2(path, dest_path)
    split_hashes[name] = get_file_sha256(dest_path)

with open(os.path.join(SPLIT_DIR, "split_sha256_provenance.json"), 'w') as f:
    json.dump(split_hashes, f, indent=2)

with open(train_file, 'r', encoding='utf-8') as f:
    train_imgs = set(line.strip() for line in f if line.strip())
with open(val_file, 'r', encoding='utf-8') as f:
    val_imgs = set(line.strip() for line in f if line.strip())
with open(test_file, 'r', encoding='utf-8') as f:
    test_imgs = set(line.strip() for line in f if line.strip())

assert len(train_imgs) == 6000, f"Expected 6,000 train images, got {len(train_imgs)}"
assert len(val_imgs) == 1000, f"Expected 1,000 val images, got {len(val_imgs)}"
assert len(test_imgs) == 1000, f"Expected 1,000 test images, got {len(test_imgs)}"
assert train_imgs.isdisjoint(val_imgs) and train_imgs.isdisjoint(test_imgs) and val_imgs.isdisjoint(test_imgs)

def assign_split(img_id):
    if img_id in train_imgs: return "train"
    if img_id in val_imgs: return "val"
    if img_id in test_imgs: return "test"
    return "unassigned"

DF_PAIRS["split"] = DF_PAIRS["image_id"].apply(assign_split)
train_df = DF_PAIRS[DF_PAIRS["split"] == "train"].reset_index(drop=True)
val_df = DF_PAIRS[DF_PAIRS["split"] == "val"].reset_index(drop=True)
test_df = DF_PAIRS[DF_PAIRS["split"] == "test"].reset_index(drop=True)

# Tokenizer & Canonical ViT Pretrained Transform
TOKENIZER_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
VIT_WEIGHTS = tv_models.ViT_B_16_Weights.DEFAULT
image_transform = VIT_WEIGHTS.transforms()
print(f"Loaded canonical ViT-B/16 preprocessing: {image_transform}")

class Flickr8kDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, max_seq_len=64):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.records = df.to_dict('records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        img_name = item["image_id"]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"CRITICAL: Flickr8k image not found at path: {img_path}")
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        caption = str(item["caption"])
        tok = tokenizer(caption, padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        return {
            "image": image_tensor,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "image_id": img_name,
            "caption_id": item.get("caption_id", f"{img_name}#{idx}"),
            "caption_text": caption
        }

train_dataset = Flickr8kDataset(train_df, IMG_DIR, transform=image_transform)
val_dataset = Flickr8kDataset(val_df, IMG_DIR, transform=image_transform)
test_dataset = Flickr8kDataset(test_df, IMG_DIR, transform=image_transform)

class AtomicGroupedBatchSampler(Sampler):
    def __init__(self, df, batch_size=32, captions_per_image=2, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.k = captions_per_image
        self.num_unique_images_per_batch = batch_size // captions_per_image
        self.shuffle = shuffle
        self.img_to_indices = defaultdict(list)
        for idx, row in self.df.iterrows():
            self.img_to_indices[row["image_id"]].append(idx)
        self.unique_images = list(self.img_to_indices.keys())
        self.num_batches = len(self.unique_images) // self.num_unique_images_per_batch

    def __iter__(self):
        imgs = list(self.unique_images)
        if self.shuffle: random.shuffle(imgs)
        batches, cur_batch = [], []
        for img in imgs:
            idxs = self.img_to_indices[img]
            sampled = random.sample(idxs, self.k) if (self.shuffle and len(idxs) >= self.k) else (idxs * ((self.k // len(idxs)) + 1))[:self.k]
            cur_batch.extend(sampled)
            if len(cur_batch) >= self.batch_size:
                batches.append(cur_batch[:self.batch_size])
                cur_batch = []
        return iter(batches)

    def __len__(self):
        return self.num_batches

BATCH_SIZE = 32
train_sampler = AtomicGroupedBatchSampler(train_df, batch_size=BATCH_SIZE, captions_per_image=2, shuffle=True)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Backbones
class FrozenVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
        for p in self.vit.parameters(): p.requires_grad = False
        self.vit.eval()

    @torch.no_grad()
    def forward(self, x):
        x_prep = self.vit._process_input(x)
        n = x_prep.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x_seq = torch.cat([batch_class_token, x_prep], dim=1)
        x_enc = self.vit.encoder(x_seq)
        return x_enc[:, 1:, :]

class FrozenLanguageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(TOKENIZER_NAME)
        for p in self.roberta.parameters(): p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        return self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

vision_backbone = FrozenVisionBackbone().to(DEVICE)
language_backbone = FrozenLanguageBackbone().to(DEVICE)

# Architecture
class HEDO(nn.Module):
    def __init__(self, d_model=128, K_steps=3, dt=0.1, beta=0.05, gamma=0.1):
        super().__init__()
        self.K_steps, self.dt, self.beta, self.gamma = K_steps, dt, beta, gamma
        self.W_p = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_diagnostics=False):
        q = x
        p = torch.tanh(self.W_p(q))
        energies = []
        if return_diagnostics:
            energies.append((0.5 * (p.pow(2).sum(dim=-1) + q.pow(2).sum(dim=-1)).mean()).item())
        for _ in range(self.K_steps):
            grad_V = torch.tanh(self.W_q(q))
            p = (1.0 - self.beta * self.dt) * p - self.dt * grad_V
            q = q + self.gamma * self.dt * p
            if return_diagnostics:
                energies.append((0.5 * (p.pow(2).sum(dim=-1) + q.pow(2).sum(dim=-1)).mean()).item())
        out = x + self.gamma * self.norm(q)
        if return_diagnostics:
            return out, {"energies": energies, "cosine_fidelity": F.cosine_similarity(x, out, dim=-1).mean().item()}
        return out

class StateContinuousSSD(nn.Module):
    def __init__(self, d_model=128, d_state=64, chunk_size=16):
        super().__init__()
        self.d_model, self.d_state, self.chunk_size = d_model, d_state, chunk_size
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_boundary_states=False):
        B, L, D = x.shape
        x_in, gate = self.in_proj(x).chunk(2, dim=-1)
        x_in = F.silu(x_in)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        A_decay = torch.exp(-torch.exp(self.A_log))
        outputs, boundary_states = [], []
        for t in range(L):
            xt = x_in[:, t, :]
            Bt = self.B_proj(xt)
            if mask is not None:
                mt = mask[:, t:t+1]
                h = mt * (A_decay * h + Bt) + (1.0 - mt) * h
            else:
                h = A_decay * h + Bt
            outputs.append(self.C_proj(h) + self.D * xt)
            if (t + 1) % self.chunk_size == 0:
                boundary_states.append(h)
        y = self.norm(self.out_proj(torch.stack(outputs, dim=1) * F.silu(gate)))
        if return_boundary_states:
            if len(boundary_states) == 0: boundary_states = [h]
            b_states = torch.stack(boundary_states, dim=1)
            if mask is not None:
                chunk_masks = [(mask[:, k*self.chunk_size:(k+1)*self.chunk_size].sum(dim=-1) > 0).float() for k in range(b_states.shape[1])]
                b_mask = torch.stack(chunk_masks, dim=1)
            else:
                b_mask = torch.ones(B, b_states.shape[1], device=x.device)
            return y, b_states, b_mask
        return y

class ChunkWiseHVSC(nn.Module):
    def __init__(self, d_state=64, d_latent=128):
        super().__init__()
        self.img_mu = nn.Linear(d_state, d_latent)
        self.img_logvar = nn.Linear(d_state, d_latent)
        self.txt_mu = nn.Linear(d_state, d_latent)
        self.txt_logvar = nn.Linear(d_state, d_latent)

    def forward(self, h_img_bound, h_txt_bound, mask_txt_chunks=None, sample_latent=True):
        h_img_pooled = h_img_bound.mean(dim=1)
        if mask_txt_chunks is not None:
            w = mask_txt_chunks.unsqueeze(-1)
            h_txt_pooled = (h_txt_bound * w).sum(dim=1) / (w.sum(dim=1).clamp(min=1.0))
        else:
            h_txt_pooled = h_txt_bound.mean(dim=1)
        mu_img, logvar_img = self.img_mu(h_img_pooled), self.img_logvar(h_img_pooled).clamp(-10.0, 10.0)
        mu_txt, logvar_txt = self.txt_mu(h_txt_pooled), self.txt_logvar(h_txt_pooled).clamp(-10.0, 10.0)
        if sample_latent and self.training:
            z_img = mu_img + torch.exp(0.5 * logvar_img) * torch.randn_like(logvar_img)
            z_txt = mu_txt + torch.exp(0.5 * logvar_txt) * torch.randn_like(logvar_txt)
        else:
            z_img, z_txt = mu_img, mu_txt
        var_img, var_txt = torch.exp(logvar_img.float()), torch.exp(logvar_txt.float())
        kl_i = 0.5 * torch.sum(logvar_txt.float() - logvar_img.float() + (var_img + (mu_img.float() - mu_txt.float()).pow(2)) / var_txt - 1.0, dim=-1)
        kl_t = 0.5 * torch.sum(logvar_img.float() - logvar_txt.float() + (var_txt + (mu_txt.float() - mu_img.float()).pow(2)) / var_img - 1.0, dim=-1)
        return z_img, z_txt, 0.5 * (kl_i + kl_t).mean()

class HEDO_HVSC_Model(nn.Module):
    def __init__(self, embed_dim=128, d_state=64, chunk_size=16, use_hedo=True, use_hvsc=True, alpha=0.5):
        super().__init__()
        self.use_hedo, self.use_hvsc, self.embed_dim, self.alpha = use_hedo, use_hvsc, embed_dim, alpha
        self.img_proj = nn.Linear(768, embed_dim)
        self.txt_proj = nn.Linear(768, embed_dim)
        if use_hedo:
            self.hedo_img, self.hedo_txt = HEDO(embed_dim), HEDO(embed_dim)
        self.ssd_img = StateContinuousSSD(embed_dim, d_state, chunk_size)
        self.ssd_txt = StateContinuousSSD(embed_dim, d_state, chunk_size)
        if use_hvsc:
            self.hvsc = ChunkWiseHVSC(d_state, embed_dim)
        self.head_img, self.head_txt = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)
        self.norm_img, self.norm_txt = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)

    def encode_image(self, feats_img):
        x = self.hedo_img(self.img_proj(feats_img)) if self.use_hedo else self.img_proj(feats_img)
        seq_out, bounds, _ = self.ssd_img(x, return_boundary_states=True)
        h_pool = seq_out.mean(dim=1)
        emb = self.norm_img(h_pool + self.alpha * self.head_img(self.hvsc.img_mu(bounds.mean(dim=1)))) if self.use_hvsc else self.norm_img(self.head_img(h_pool))
        return F.normalize(emb, p=2, dim=-1)

    def encode_text(self, feats_txt, mask_txt):
        x = self.hedo_txt(self.txt_proj(feats_txt)) if self.use_hedo else self.txt_proj(feats_txt)
        seq_out, bounds, b_mask = self.ssd_txt(x, mask=mask_txt, return_boundary_states=True)
        w = mask_txt.unsqueeze(-1)
        h_pool = (seq_out * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        if self.use_hvsc:
            w_b = b_mask.unsqueeze(-1)
            h_bound = (bounds * w_b).sum(dim=1) / w_b.sum(dim=1).clamp(min=1.0)
            emb = self.norm_txt(h_pool + self.alpha * self.head_txt(self.hvsc.txt_mu(h_bound)))
        else:
            emb = self.norm_txt(self.head_txt(h_pool))
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, feats_img, feats_txt, mask_txt):
        x_img = self.hedo_img(self.img_proj(feats_img)) if self.use_hedo else self.img_proj(feats_img)
        x_txt = self.hedo_txt(self.txt_proj(feats_txt)) if self.use_hedo else self.txt_proj(feats_txt)
        seq_img, bounds_img, _ = self.ssd_img(x_img, return_boundary_states=True)
        seq_txt, bounds_txt, mask_txt_chunks = self.ssd_txt(x_txt, mask=mask_txt, return_boundary_states=True)
        h_pool_img = seq_img.mean(dim=1)
        w_txt = mask_txt.unsqueeze(-1)
        h_pool_txt = (seq_txt * w_txt).sum(dim=1) / w_txt.sum(dim=1).clamp(min=1.0)
        kl_loss = torch.tensor(0.0, device=feats_img.device)
        if self.use_hvsc:
            z_img, z_txt, kl_loss = self.hvsc(bounds_img, bounds_txt, mask_txt_chunks=mask_txt_chunks, sample_latent=True)
            emb_img = self.norm_img(h_pool_img + self.alpha * self.head_img(z_img))
            emb_txt = self.norm_txt(h_pool_txt + self.alpha * self.head_txt(z_txt))
        else:
            emb_img = self.norm_img(self.head_img(h_pool_img))
            emb_txt = self.norm_txt(self.head_txt(h_pool_txt))
        return F.normalize(emb_img, p=2, dim=-1), F.normalize(emb_txt, p=2, dim=-1), kl_loss

class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_img, z_txt, image_ids):
        sim = torch.matmul(z_img, z_txt.T) / self.temperature
        pos_mask = torch.tensor([[id_i == id_j for id_j in image_ids] for id_i in image_ids], device=z_img.device, dtype=torch.float32)
        pos_mask = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        loss_i2t = -(F.log_softmax(sim, dim=1) * pos_mask).sum(dim=1).mean()
        loss_t2i = -(F.log_softmax(sim.T, dim=1) * pos_mask.T).sum(dim=1).mean()
        return 0.5 * (loss_i2t + loss_t2i)

infonce_criterion = MultiPositiveInfoNCELoss(temperature=0.07)

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device=DEVICE, return_raw_artifacts=False):
    model.eval()
    img_embs, txt_embs, img_ids_list, cap_ids_list = [], [], [], []
    for batch in dataloader:
        imgs, input_ids, attention_mask = batch["image"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device)
        feats_img = vision_backbone(imgs)
        feats_txt = language_backbone(input_ids, attention_mask)
        z_img = model.encode_image(feats_img)
        z_txt = model.encode_text(feats_txt, attention_mask)
        img_embs.append(z_img.cpu())
        txt_embs.append(z_txt.cpu())
        img_ids_list.extend(batch["image_id"])
        cap_ids_list.extend(batch["caption_id"])
    img_embs = torch.cat(img_embs, dim=0).numpy()
    txt_embs = torch.cat(txt_embs, dim=0).numpy()
    unique_img_ids, unique_img_embs, seen = [], [], set()
    for idx, iid in enumerate(img_ids_list):
        if iid not in seen:
            seen.add(iid)
            unique_img_ids.append(iid)
            unique_img_embs.append(img_embs[idx])
    unique_img_embs = np.array(unique_img_embs)
    sim_matrix = np.dot(unique_img_embs, txt_embs.T)
    img_to_cap_indices = defaultdict(list)
    for cap_idx, iid in enumerate(img_ids_list): img_to_cap_indices[iid].append(cap_idx)
    i2t_ranks = [next(r for r, c_idx in enumerate(np.argsort(-sim_matrix[i])) if c_idx in set(img_to_cap_indices[iid])) for i, iid in enumerate(unique_img_ids)]
    t2i_ranks = [np.where(np.argsort(-sim_matrix[:, c_idx]) == unique_img_ids.index(iid))[0][0] for c_idx, iid in enumerate(img_ids_list)]
    i2t_r1 = float(np.mean(np.array(i2t_ranks) < 1) * 100)
    i2t_r5 = float(np.mean(np.array(i2t_ranks) < 5) * 100)
    i2t_r10 = float(np.mean(np.array(i2t_ranks) < 10) * 100)
    t2i_r1 = float(np.mean(np.array(t2i_ranks) < 1) * 100)
    t2i_r5 = float(np.mean(np.array(t2i_ranks) < 5) * 100)
    t2i_r10 = float(np.mean(np.array(t2i_ranks) < 10) * 100)
    mean_recall = float((i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6.0)
    metrics = {"i2t_r1": i2t_r1, "i2t_r5": i2t_r5, "i2t_r10": i2t_r10, "t2i_r1": t2i_r1, "t2i_r5": t2i_r5, "t2i_r10": t2i_r10, "mean_recall": mean_recall}
    if return_raw_artifacts:
        return metrics, {"unique_img_embs": unique_img_embs, "txt_embs": txt_embs, "unique_img_ids": unique_img_ids, "cap_ids_list": cap_ids_list, "sim_matrix": sim_matrix}
    return metrics

def train_single_epoch(model, dataloader, optimizer, scaler, scheduler, kl_weight=1e-4, grad_clip=1.0, device=DEVICE):
    model.train()
    tot_loss, tot_info, tot_kl = 0.0, 0.0, 0.0
    for batch in dataloader:
        imgs, input_ids, attention_mask = batch["image"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device)
        image_ids = batch["image_id"]
        optimizer.zero_grad()
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            f_img = vision_backbone(imgs)
            f_txt = language_backbone(input_ids, attention_mask)
            z_img, z_txt, kl_loss = model(f_img, f_txt, attention_mask)
            infonce_loss = infonce_criterion(z_img, z_txt, image_ids)
            loss = infonce_loss + kl_weight * kl_loss
        if scaler and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        tot_loss += loss.item()
        tot_info += infonce_loss.item()
        tot_kl += kl_loss.item()
    n = len(dataloader)
    return {"loss": tot_loss / n, "infonce_loss": tot_info / n, "kl_loss": tot_kl / n}

if __name__ == "__main__":
    BENCHMARK_CONFIGS = [
        {"name": "SSD Baseline",           "tag": "baseline",      "use_hedo": False, "use_hvsc": False},
        {"name": "HEDO-HVSC w/o HEDO",     "tag": "wo_hedo",       "use_hedo": False, "use_hvsc": True},
        {"name": "HEDO-HVSC w/o HVSC",     "tag": "wo_hvsc",       "use_hedo": True,  "use_hvsc": False},
        {"name": "Full HEDO-HVSC (Ours)", "tag": "full_hedo_hvsc","use_hedo": True,  "use_hvsc": True},
    ]
    BENCHMARK_SEEDS = [42, 43, 44]
    MAX_EPOCHS, PATIENCE, MIN_DELTA = 8, 3, 0.05
    BASE_LR, MIN_LR, GRAD_CLIP_NORM, KL_WEIGHT = 2e-4, 1e-6, 1.0, 1e-4

    FORCE_RERUN = False
    MAX_RUNTIME_HOURS = float(os.environ.get("MAX_RUNTIME_HOURS", 10.0))
    BENCHMARK_START_TIME = time.time()
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "training.log")

    def log_event(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

    def check_runtime_guard():
        elapsed_hours = (time.time() - BENCHMARK_START_TIME) / 3600.0
        if elapsed_hours >= MAX_RUNTIME_HOURS:
            log_event(f"RUNTIME GUARD TRIGGERED: Elapsed {elapsed_hours:.2f}h >= {MAX_RUNTIME_HOURS:.2f}h. Gracefully yielding.")
            return False
        return True

    master_results_json = os.path.join(OUTPUT_DIR, "master_results.json")
    master_results_csv = os.path.join(OUTPUT_DIR, "master_results.csv")
    if os.path.isfile(master_results_json) and not FORCE_RERUN:
        with open(master_results_json, 'r') as f:
            master_records = json.load(f)
    else:
        master_records = []

    log_event("=" * 70)
    log_event(f"STARTING 12-RUN FACTORIAL BENCHMARK ({len(BENCHMARK_CONFIGS)} CONFIGS x {len(BENCHMARK_SEEDS)} SEEDS)")
    log_event(f"Output directory: {OUTPUT_DIR}")
    log_event(f"Max runtime: {MAX_RUNTIME_HOURS}h | Force rerun: {FORCE_RERUN}")
    log_event("=" * 70)

    runtime_budget_exhausted = False
    completed_count, resumed_count, skipped_count = 0, 0, 0

    for cfg in BENCHMARK_CONFIGS:
        if runtime_budget_exhausted:
            break
        for seed in BENCHMARK_SEEDS:
            if not check_runtime_guard():
                runtime_budget_exhausted = True
                break

            run_tag = f"{cfg['tag']}_seed_{seed}"
            run_ckpt_dir = os.path.join(CHECKPOINT_DIR, run_tag)
            os.makedirs(run_ckpt_dir, exist_ok=True)
            best_ckpt_path = os.path.join(run_ckpt_dir, "best_val.pt")
            latest_ckpt_path = os.path.join(run_ckpt_dir, "latest.pt")
            history_path = os.path.join(run_ckpt_dir, "train_history.json")
            config_path = os.path.join(run_ckpt_dir, "run_config.json")
            state_path = os.path.join(run_ckpt_dir, "run_state.json")
            test_metrics_path = os.path.join(run_ckpt_dir, "test_metrics.json")

            set_all_seeds(seed)
            model = HEDO_HVSC_Model(embed_dim=128, d_state=64, chunk_size=16, use_hedo=cfg["use_hedo"], use_hvsc=cfg["use_hvsc"]).to(DEVICE)
            actual_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            run_config = {
                "name": cfg["name"],
                "tag": cfg["tag"],
                "seed": seed,
                "use_hedo": cfg["use_hedo"],
                "use_hvsc": cfg["use_hvsc"],
                "embed_dim": 128,
                "d_state": 64,
                "chunk_size": 16,
                "trainable_params": actual_trainable_params,
                "base_lr": BASE_LR,
                "kl_weight": KL_WEIGHT,
                "max_epochs": MAX_EPOCHS
            }
            with open(config_path, 'w') as f: json.dump(run_config, f, indent=2)

            if os.path.isfile(best_ckpt_path) and os.path.isfile(test_metrics_path) and not FORCE_RERUN:
                with open(test_metrics_path, 'r') as f:
                    cached_metrics = json.load(f)
                record = {
                    "name": cfg["name"],
                    "tag": cfg["tag"],
                    "seed": seed,
                    "use_hedo": cfg["use_hedo"],
                    "use_hvsc": cfg["use_hvsc"],
                    "train_time_sec": cached_metrics.get("train_time_sec", 0.0),
                    "trainable_params": actual_trainable_params,
                    **cached_metrics
                }
                master_records = [r for r in master_records if not (r.get("tag") == cfg["tag"] and r.get("seed") == seed)]
                master_records.append(record)
                with open(state_path, 'w') as f:
                    json.dump({"status": "COMPLETED", "tag": run_tag, "mean_recall": cached_metrics.get("mean_recall", 0.0)}, f, indent=2)
                log_event(f"Run [{cfg['name']}] (Seed {seed}) already completed & cached. Mean Recall: {cached_metrics.get('mean_recall', 0.0):.2f}%")
                skipped_count += 1
                continue

            log_event(f"\n▶ Training Model: {cfg['name']} (Params: {actual_trainable_params:,}) | Seed: {seed}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * len(train_loader), eta_min=MIN_LR)
            scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

            start_epoch = 1
            best_val_recall = -1.0
            patience_counter = 0
            train_history = []

            if os.path.isfile(latest_ckpt_path) and not FORCE_RERUN:
                ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
                if scaler and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
                    scaler.load_state_dict(ckpt["scaler_state"])
                start_epoch = ckpt["epoch"] + 1
                best_val_recall = ckpt.get("best_val_recall", -1.0)
                train_history = ckpt.get("train_history", [])
                log_event(f"   Resuming from Epoch {start_epoch} (Best Val Recall so far: {best_val_recall:.2f}%)")
                resumed_count += 1

            with open(state_path, 'w') as f:
                json.dump({"status": "RUNNING", "tag": run_tag, "current_epoch": start_epoch, "best_val_recall": best_val_recall}, f, indent=2)

            train_start = time.time()
            interrupted = False

            for epoch in range(start_epoch, MAX_EPOCHS + 1):
                if not check_runtime_guard():
                    log_event(f"   Runtime limit reached during training of [{run_tag}]. Checkpointing.")
                    interrupted = True
                    break

                ep_stats = train_single_epoch(model, train_loader, optimizer, scaler, scheduler, kl_weight=KL_WEIGHT, grad_clip=GRAD_CLIP_NORM)
                val_metrics = evaluate_retrieval(model, val_loader)
                train_history.append({"epoch": epoch, "loss": ep_stats["loss"], "val_mean_recall": val_metrics["mean_recall"]})
                log_event(f"   [Epoch {epoch:02d}/{MAX_EPOCHS:02d}] Loss: {ep_stats['loss']:.4f} | Val MR: {val_metrics['mean_recall']:.2f}%")
                
                if val_metrics["mean_recall"] > best_val_recall + MIN_DELTA:
                    best_val_recall = val_metrics["mean_recall"]
                    patience_counter = 0
                    torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_metrics": val_metrics}, best_ckpt_path)
                else:
                    patience_counter += 1
                    
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler else None,
                    "best_val_recall": best_val_recall,
                    "train_history": train_history,
                    "seed": seed
                }, latest_ckpt_path)
                
                if patience_counter >= PATIENCE:
                    log_event(f"   Early stopping triggered at epoch {epoch}.")
                    break

            if interrupted:
                with open(state_path, 'w') as f:
                    json.dump({"status": "INTERRUPTED", "tag": run_tag, "epoch_reached": epoch}, f, indent=2)
                runtime_budget_exhausted = True
                break

            train_time_sec = time.time() - train_start
            with open(history_path, 'w') as f: json.dump(train_history, f, indent=2)

            checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state"])
            test_metrics, test_raw = evaluate_retrieval(model, test_loader, return_raw_artifacts=True)
            test_metrics["train_time_sec"] = train_time_sec

            np.save(os.path.join(run_ckpt_dir, "test_image_embeddings.npy"), test_raw["unique_img_embs"])
            np.save(os.path.join(run_ckpt_dir, "test_text_embeddings.npy"), test_raw["txt_embs"])
            np.save(os.path.join(run_ckpt_dir, "test_similarity_matrix.npy"), test_raw["sim_matrix"])
            with open(os.path.join(run_ckpt_dir, "test_image_ids.json"), 'w') as f: json.dump(test_raw["unique_img_ids"], f)
            with open(os.path.join(run_ckpt_dir, "test_caption_ids.json"), 'w') as f: json.dump(test_raw["cap_ids_list"], f)
            with open(test_metrics_path, 'w') as f: json.dump(test_metrics, f, indent=2)
            with open(state_path, 'w') as f:
                json.dump({"status": "COMPLETED", "tag": run_tag, "mean_recall": test_metrics["mean_recall"]}, f, indent=2)

            record = {
                "name": cfg["name"],
                "tag": cfg["tag"],
                "seed": seed,
                "use_hedo": cfg["use_hedo"],
                "use_hvsc": cfg["use_hvsc"],
                "train_time_sec": train_time_sec,
                "trainable_params": actual_trainable_params,
                **test_metrics
            }
            master_records = [r for r in master_records if not (r.get("tag") == cfg["tag"] and r.get("seed") == seed)]
            master_records.append(record)
            completed_count += 1
            with open(master_results_json, 'w') as f: json.dump(master_records, f, indent=2)
            pd.DataFrame(master_records).to_csv(master_results_csv, index=False)
            log_event(f"   ✓ Completed: Mean Recall = {test_metrics['mean_recall']:.2f}%")

    EXPECTED_TAGS = {"baseline", "wo_hedo", "wo_hvsc", "full_hedo_hvsc"}
    EXPECTED_SEEDS = {42, 43, 44}
    assert len(master_records) == 12, f"CRITICAL: Expected 12 runs, found {len(master_records)}"
    for tag in EXPECTED_TAGS:
        seeds = {r["seed"] for r in master_records if r["tag"] == tag}
        assert seeds == EXPECTED_SEEDS

    print("\n" + "=" * 70)
    print("12/12 FACTORIAL BENCHMARK FINISHED SUCCESSFULLY!")
    print("=" * 70)
