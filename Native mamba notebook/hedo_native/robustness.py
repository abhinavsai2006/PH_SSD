"""H3: robustness to noisy multimodal inputs on the Flickr8k test split.

  build     recompute frozen features for corrupted test inputs (deterministic, seed 0):
              image: gaussian_noise_{0.08,0.16} (pixel std in [0,1]), blur_{1,3} (Gaussian radius px),
                     jpeg_{30,10} (quality), occlusion_{0.25,0.5} (fraction of 16x16 patches greyed)
              text:  word_dropout_{0.1,0.3}, char_typo_{0.05,0.15} (per-character swap/delete/insert),
                     word_shuffle_{3} (local shuffles within windows of 3 words)
            -> FEATURE_DIR/robust/<name>/{img_test.npy | txt_test.npy, mask_test.npy}
  evaluate  every COMPLETED run x every corruption, same metric code as the clean test
            -> results/robustness.csv with mean_recall, dual_mean_recall and relative_mr = MR_corrupt / MR_clean

Usage: python robustness.py build ; python robustness.py evaluate [--runs-glob "*"]
"""

import argparse
import io
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from backbones import MAX_TEXT_LEN, FrozenRoBERTa, FrozenViT, load_tokenizer, tokenize, vit_transform
from common import DATA_DIR, FEATURE_DIR, RUNS_DIR, banner, provenance, read_json, sha256_file, write_json
from evaluation import CachedSplit, evaluate, load_run_model

ROBUST_DIR = FEATURE_DIR / "robust"
PATCH = 16


def gaussian_noise(std):
    def f(im, rng):
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = np.clip(a + rng.normal(0, std, a.shape), 0, 1)
        return Image.fromarray((a * 255).round().astype(np.uint8))
    return f


def blur(radius):
    return lambda im, rng: im.filter(ImageFilter.GaussianBlur(radius))


def jpeg(quality):
    def f(im, rng):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def occlusion(frac):
    """Applied after the official resize/crop, on the 224x224 grid, so occluded blocks are exact ViT patches."""
    def f(im, rng):
        a = np.array(im)
        n = 14 * 14
        for k in rng.choice(n, int(round(frac * n)), replace=False):
            r, c = divmod(int(k), 14)
            a[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 127
        return Image.fromarray(a)
    return f


def word_dropout(p):
    def f(text, rng):
        words = text.split()
        kept = [w for w in words if rng.random() >= p]
        return " ".join(kept or words[:1])
    return f


def char_typo(p):
    letters = "abcdefghijklmnopqrstuvwxyz"

    def f(text, rng):
        out, chars, i = [], list(text), 0
        while i < len(chars):
            ch = chars[i]
            if ch.isalpha() and rng.random() < p:
                op = rng.randrange(3)
                if op == 0 and i + 1 < len(chars):
                    out += [chars[i + 1], ch]
                    i += 2
                    continue
                if op == 1:
                    i += 1
                    continue
                out += [ch, rng.choice(letters)]
            else:
                out.append(ch)
            i += 1
        return "".join(out)
    return f


def word_shuffle(window):
    def f(text, rng):
        words = text.split()
        for s in range(0, len(words), window):
            seg = words[s:s + window]
            rng.shuffle(seg)
            words[s:s + window] = seg
        return " ".join(words)
    return f


IMAGE_CORRUPTIONS = {"gaussian_noise_0.08": gaussian_noise(0.08), "gaussian_noise_0.16": gaussian_noise(0.16),
                     "blur_1": blur(1), "blur_3": blur(3), "jpeg_30": jpeg(30), "jpeg_10": jpeg(10),
                     "occlusion_0.25": occlusion(0.25), "occlusion_0.5": occlusion(0.5)}
TEXT_CORRUPTIONS = {"word_dropout_0.1": word_dropout(0.1), "word_dropout_0.3": word_dropout(0.3),
                    "char_typo_0.05": char_typo(0.05), "char_typo_0.15": char_typo(0.15),
                    "word_shuffle_3": word_shuffle(3)}


@torch.no_grad()
def build(device):
    manifest = read_json(DATA_DIR / "manifest.json")
    table = pd.read_csv(DATA_DIR / "captions_test.csv", keep_default_na=False)
    image_ids = table["image_id"].iloc[::5].tolist()
    tf = vit_transform()
    vit = FrozenViT().to(device)
    record = {}
    for name, fn in IMAGE_CORRUPTIONS.items():
        out_dir = ROBUST_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        feats = []
        for s in range(0, len(image_ids), 64):
            batch = []
            for iid in image_ids[s:s + 64]:
                with Image.open(f"{manifest['image_dir']}/{iid}") as im:
                    im = im.convert("RGB")
                if name.startswith("occlusion"):
                    # official resize + centre crop, occlude on the 224 grid, then normalise
                    im = Image.fromarray((_crop(im, tf)))
                    im = fn(im, rng)
                    batch.append(_normalise(im, tf))
                else:
                    batch.append(tf(fn(im, rng)))
            feats.append(vit(torch.stack(batch).to(device)).float().cpu().numpy().astype(np.float16))
        np.save(out_dir / "img_test.npy", np.concatenate(feats))
        record[name] = {"modality": "image", "sha256": sha256_file(out_dir / "img_test.npy")}
        print("built", name, flush=True)
    del vit

    roberta, tokenizer = FrozenRoBERTa().to(device), load_tokenizer()
    for name, fn in TEXT_CORRUPTIONS.items():
        out_dir = ROBUST_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(0)
        captions = [fn(c, rng) for c in table["caption"].tolist()]
        tok = tokenize(tokenizer, captions)
        feats = []
        for s in range(0, len(captions), 256):
            feats.append(roberta(tok["input_ids"][s:s + 256].to(device), tok["attention_mask"][s:s + 256].to(device))
                         .float().cpu().numpy().astype(np.float16))
        np.save(out_dir / "txt_test.npy", np.concatenate(feats))
        np.save(out_dir / "mask_test.npy", tok["attention_mask"].numpy().astype(np.uint8))
        pd.DataFrame({"clean": table["caption"], "corrupted": captions}).head(50).to_csv(out_dir / "examples.csv",
                                                                                         index=False)
        record[name] = {"modality": "text", "sha256": sha256_file(out_dir / "txt_test.npy"),
                        "max_text_len": MAX_TEXT_LEN}
        print("built", name, flush=True)
    write_json(ROBUST_DIR / "robust_manifest.json", {"corruptions": record, "provenance": provenance()})
    print("ROBUSTNESS_BUILD: PASS")


def _crop(im, tf):
    """Resize + centre crop of the official transform, returned as uint8 HxWx3."""
    from torchvision.transforms import functional as TF
    im = TF.resize(im, tf.resize_size, interpolation=tf.interpolation, antialias=tf.antialias)
    return np.array(TF.center_crop(im, tf.crop_size))


def _normalise(im, tf):
    from torchvision.transforms import functional as TF
    return TF.normalize(TF.to_tensor(im), mean=tf.mean, std=tf.std)


def evaluate_all(device, runs_glob):
    splits = {"clean": CachedSplit("test")}
    for name in IMAGE_CORRUPTIONS:
        splits[name] = CachedSplit("test", img_path=ROBUST_DIR / name / "img_test.npy")
    for name in TEXT_CORRUPTIONS:
        splits[name] = CachedSplit("test", txt_path=ROBUST_DIR / name / "txt_test.npy",
                                   mask_path=ROBUST_DIR / name / "mask_test.npy")
    rows = []
    for summary_path in sorted(RUNS_DIR.glob(f"{runs_glob}/run_summary.json")):
        s = read_json(summary_path)
        if s.get("status") != "COMPLETED":
            continue
        run_dir = summary_path.parent
        cfg = read_json(run_dir / "config.json")
        model = load_run_model(run_dir, device)
        rerank_k = cfg.get("rerank_k", 16) if model.cfg.exchanges else 0
        clean = None
        for name, data in splits.items():
            m = evaluate(model, data, device, rerank_k)[0]
            clean = clean or m
            rows.append({"run": run_dir.name, "variant": s["variant"], "seed": s["seed"], "kl_weight": s["kl_weight"],
                         "hvsc_chunk_size": s["hvsc_chunk_size"], "corruption": name,
                         "modality": "none" if name == "clean" else ("image" if name in IMAGE_CORRUPTIONS else "text"),
                         "mean_recall": m["mean_recall"], "dual_mean_recall": m["dual_mean_recall"],
                         "relative_mr": m["mean_recall"] / max(clean["mean_recall"], 1e-9)})
        print(run_dir.name, "done", flush=True)
    out = RUNS_DIR.parent / "results" / "robustness.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")
    print("ROBUSTNESS_EVAL: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "evaluate"])
    ap.add_argument("--runs-glob", default="*")
    args = ap.parse_args()
    device = torch.device(os.environ.get("HEDO_DEVICE", "cuda"))
    banner(f"ROBUSTNESS {args.action.upper()}")
    if args.action == "build":
        build(device)
    else:
        evaluate_all(device, args.runs_glob)


if __name__ == "__main__":
    main()
