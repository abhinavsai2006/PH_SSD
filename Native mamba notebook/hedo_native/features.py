"""Cache frozen backbone features once (FP32 compute, FP16 storage).

Writes to FEATURE_DIR, per split s in {train, val, test}:
  img_{s}.npy    [N_images, 196, 768] float16   (image i = row i of manifest order)
  txt_{s}.npy    [N_captions, 64, 768] float16  (caption 5i+k belongs to image i)
  mask_{s}.npy   [N_captions, 64] uint8
  ids_{s}.npy    [N_captions, 64] int32         (token ids, for end-to-end profiling)
  saliency_{s}.npy [N_images, 196] float32      (last-block CLS->patch attention; H1 probe only)
  feature_manifest.json  shapes, file SHA256, backbone identities, preprocessing, FP16 range check
  roberta_loading_info.json

Because both backbones are frozen and in eval mode (no augmentation), caching is
exactly equivalent to recomputing them every step.

Usage: python features.py [--batch-size 64]
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backbones import (MAX_TEXT_LEN, ROBERTA_NAME, ROBERTA_REVISION, VIT_WEIGHTS, FrozenRoBERTa, FrozenViT,
                       load_tokenizer, tokenize, vit_transform)
from common import (DATA_DIR, FEATURE_DIR, banner, feature_code_sha256, provenance, read_json, sha256_file,
                    write_json)

FP16_MAX = 65504.0


class ImageFiles(Dataset):
    def __init__(self, image_dir, ids):
        self.image_dir, self.ids, self.tf = image_dir, ids, vit_transform()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(f"{self.image_dir}/{self.ids[i]}") as im:
            return self.tf(im.convert("RGB"))


def store(arr_path, shape, batches):
    out = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float16, shape=shape)
    pos, absmax = 0, 0.0
    for feats in batches:
        feats = feats.float()
        if not torch.isfinite(feats).all():
            raise FloatingPointError(f"non-finite backbone features while writing {arr_path.name}")
        absmax = max(absmax, float(feats.abs().max()))
        if absmax >= FP16_MAX:
            raise OverflowError(f"{arr_path.name}: |x|={absmax} does not fit float16")
        out[pos:pos + feats.shape[0]] = feats.cpu().numpy().astype(np.float16)
        pos += feats.shape[0]
    assert pos == shape[0], (pos, shape)
    out.flush()
    del out
    return absmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    banner("FROZEN BACKBONE FEATURE CACHE")
    device = torch.device("cuda")
    manifest = read_json(DATA_DIR / "manifest.json")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    vit = FrozenViT().to(device)
    roberta = FrozenRoBERTa().to(device)
    tokenizer = load_tokenizer()
    write_json(FEATURE_DIR / "roberta_loading_info.json",
               {"loading_info": roberta.loading_info, "commit_hash": roberta.commit_hash})
    print("RoBERTa commit:", roberta.commit_hash)
    print("RoBERTa unexpected keys (expected lm_head.* only):", len(roberta.loading_info.get("unexpected_keys", [])))

    record = {"splits": {}, "absmax": {}}
    for split in ("train", "val", "test"):
        table = pd.read_csv(DATA_DIR / f"captions_{split}.csv", keep_default_na=False)
        image_ids = table["image_id"].iloc[::5].tolist()
        t0 = time.time()

        loader = DataLoader(ImageFiles(manifest["image_dir"], image_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
        img_path = FEATURE_DIR / f"img_{split}.npy"
        saliency = []

        def image_batches():
            for i, x in enumerate(loader):
                feats, sal = vit.forward_with_saliency(x.to(device, non_blocking=True))
                if i == 0:  # the saliency path must reproduce the official forward exactly
                    ref = vit(x.to(device))
                    assert torch.allclose(ref, feats, atol=1e-5), float((ref - feats).abs().max())
                saliency.append(sal.float().cpu().numpy())
                yield feats

        record["absmax"][f"img_{split}"] = store(img_path, (len(image_ids), 196, 768), image_batches())
        np.save(FEATURE_DIR / f"saliency_{split}.npy", np.concatenate(saliency).astype(np.float32))

        tok = tokenize(tokenizer, table["caption"].tolist())
        ids = tok["input_ids"].to(torch.int32)
        mask = tok["attention_mask"].to(torch.uint8)
        np.save(FEATURE_DIR / f"ids_{split}.npy", ids.numpy())
        np.save(FEATURE_DIR / f"mask_{split}.npy", mask.numpy())
        truncated = int((tok["attention_mask"].sum(1) == MAX_TEXT_LEN).sum())

        def text_batches():
            for s in range(0, len(table), 256):
                yield roberta(ids[s:s + 256].long().to(device), mask[s:s + 256].long().to(device))

        txt_path = FEATURE_DIR / f"txt_{split}.npy"
        record["absmax"][f"txt_{split}"] = store(txt_path, (len(table), MAX_TEXT_LEN, 768), text_batches())

        record["splits"][split] = {
            "images": len(image_ids),
            "captions": len(table),
            "captions_at_max_len_possibly_truncated": truncated,
            "image_ids_order": image_ids,
            "sha256": {p.name: sha256_file(p) for p in [img_path, txt_path, FEATURE_DIR / f"mask_{split}.npy",
                                                        FEATURE_DIR / f"ids_{split}.npy"]},
        }
        print(f"{split}: {len(image_ids)} images, {len(table)} captions, {time.time() - t0:.0f}s, "
              f"truncated={truncated}")

    record.update({
        "vit_weights": str(VIT_WEIGHTS),
        "vit_transform": repr(vit_transform()),
        "roberta": {"name": ROBERTA_NAME, "requested_revision": ROBERTA_REVISION, "commit_hash": roberta.commit_hash},
        "backbone_compute_dtype": "float32",
        "storage_dtype": "float16",
        "max_text_len": MAX_TEXT_LEN,
        "vit_token_output": "196 patch tokens after encoder LayerNorm (class token dropped)",
        "text_token_output": "last_hidden_state, right padded",
        "dataset_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "feature_code_sha256": feature_code_sha256(),
    })
    record["provenance"] = provenance(include_artifacts=False)
    write_json(FEATURE_DIR / "feature_manifest.json", record)
    print("FEATURE_CACHE: PASS")


if __name__ == "__main__":
    main()
