"""Shared evaluation: cached splits, dual embeddings, exchange re-ranking, run loading."""

import numpy as np
import torch

from common import FEATURE_DIR, read_json
from metrics import CAPTIONS_PER_IMAGE, retrieval_metrics, topk_candidates
from models import ModelConfig, RetrievalModel


class NumericalFailure(RuntimeError):
    pass


class CachedSplit:
    """Cached features for one split. Paths can be overridden (robustness corruptions)."""

    def __init__(self, split, in_memory=True, img_path=None, txt_path=None, mask_path=None):
        mode = None if in_memory else "r"
        self.img = np.load(img_path or FEATURE_DIR / f"img_{split}.npy", mmap_mode=mode)
        self.txt = np.load(txt_path or FEATURE_DIR / f"txt_{split}.npy", mmap_mode=mode)
        self.mask = np.load(mask_path or FEATURE_DIR / f"mask_{split}.npy")
        self.image_ids = read_json(FEATURE_DIR / "feature_manifest.json")["splits"][split]["image_ids_order"]
        assert self.txt.shape[0] == CAPTIONS_PER_IMAGE * self.img.shape[0] == self.mask.shape[0]

    def img_tensor(self, rows, device):
        return torch.from_numpy(np.asarray(self.img[rows], dtype=np.float32)).to(device, non_blocking=True)

    def txt_tensor(self, rows, device):
        txt = torch.from_numpy(np.asarray(self.txt[rows], dtype=np.float32)).to(device, non_blocking=True)
        mask = torch.from_numpy(np.asarray(self.mask[rows]).astype(bool)).to(device, non_blocking=True)
        return txt, mask

    def batch(self, image_rows, device):
        image_rows = np.sort(np.asarray(image_rows))
        cap_rows = (image_rows[:, None] * CAPTIONS_PER_IMAGE + np.arange(CAPTIONS_PER_IMAGE)).reshape(-1)
        txt, mask = self.txt_tensor(cap_rows, device)
        pair = torch.arange(len(image_rows), device=device).repeat_interleave(CAPTIONS_PER_IMAGE)
        return self.img_tensor(image_rows, device), txt, mask, pair, cap_rows


def _slices(n, size):
    return [np.arange(s, min(n, s + size)) for s in range(0, n, size)]


@torch.no_grad()
def pass1_all(model, data, device, chunk=250, img_transform=None):
    """Dual embeddings and the pass-1 state needed for exchange (kept on device)."""
    model.eval()
    keep = ("x", "states", "cmask", "z", "z_pooled")
    img_emb, txt_emb, img_aux, txt_aux = [], [], {k: [] for k in keep}, {k: [] for k in keep}
    masks = []
    for rows in _slices(data.img.shape[0], chunk):
        feats = data.img_tensor(rows, device)
        if img_transform is not None:
            feats = img_transform(feats, rows)
        z, aux, _ = model.pass1("img", feats, None, sample=False)
        img_emb.append(z)
        for k in keep:
            if k in aux:
                img_aux[k].append(aux[k])
    for rows in _slices(data.txt.shape[0], 2 * chunk):
        feats, mask = data.txt_tensor(rows, device)
        z, aux, _ = model.pass1("txt", feats, mask, sample=False)
        txt_emb.append(z)
        masks.append(mask)
        for k in keep:
            if k in aux:
                txt_aux[k].append(aux[k])
    img = torch.cat(img_emb).float()
    txt = torch.cat(txt_emb).float()
    if not (torch.isfinite(img).all() and torch.isfinite(txt).all()):
        raise NumericalFailure("non-finite embeddings during evaluation")
    cache = None
    if model.cfg.exchanges:
        cache = {"img": {k: torch.cat(v) for k, v in img_aux.items() if v},
                 "txt": {k: torch.cat(v) for k, v in txt_aux.items() if v},
                 "txt_mask": torch.cat(masks)}
    return img.cpu().numpy(), txt.cpu().numpy(), cache


@torch.no_grad()
def exchange_scores(model, cache, img_rows, txt_rows, device, batch=512, message_scale=(1.0, 1.0)):
    out = []
    for s in range(0, len(img_rows), batch):
        ir = torch.as_tensor(img_rows[s:s + batch], device=device)
        tr = torch.as_tensor(txt_rows[s:s + batch], device=device)
        ai = {k: v[ir] for k, v in cache["img"].items()}
        at = {k: v[tr] for k, v in cache["txt"].items()}
        out.append(model.pair_similarity(ai, at, cache["txt_mask"][tr], message_scale).float())
    scores = torch.cat(out).cpu().numpy()
    if not np.isfinite(scores).all():
        raise NumericalFailure("non-finite exchange scores")
    return scores


@torch.no_grad()
def rerank(model, img, txt, cache, device, k):
    i2t_idx, t2i_idx = topk_candidates(img, txt, k)
    n_img, n_txt = img.shape[0], txt.shape[0]
    pair_keys = np.unique(np.concatenate([
        (np.arange(n_img)[:, None] * n_txt + i2t_idx).reshape(-1),
        (t2i_idx * n_txt + np.arange(n_txt)[:, None]).reshape(-1)]))
    scores = exchange_scores(model, cache, pair_keys // n_txt, pair_keys % n_txt, device)
    lookup = dict(zip(pair_keys.tolist(), scores.tolist()))
    i2t_score = np.array([[lookup[i * n_txt + t] for t in row] for i, row in enumerate(i2t_idx)], dtype=np.float64)
    t2i_score = np.array([[lookup[i * n_txt + c] for i in row] for c, row in enumerate(t2i_idx)], dtype=np.float64)
    return {"i2t_idx": i2t_idx, "i2t_score": i2t_score, "t2i_idx": t2i_idx, "t2i_score": t2i_score}


def evaluate(model, data, device, rerank_k=16, img_transform=None):
    img, txt, cache = pass1_all(model, data, device, img_transform=img_transform)
    rr = rerank(model, img, txt, cache, device, rerank_k) if cache is not None and rerank_k > 0 else None
    return retrieval_metrics(img, txt, rr), img, txt, rr, cache


def load_run_model(run_dir, device, checkpoint="best_model.pt"):
    ckpt = torch.load(run_dir / checkpoint, map_location=device, weights_only=True)
    model = RetrievalModel(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()
