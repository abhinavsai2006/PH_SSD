"""Flickr8k preparation: download, checksum, official splits, leakage checks.

Writes to DATA_DIR:
  manifest.json            canonical paths + all checksums (single key schema)
  captions_{train,val,test}.csv   image_id, cap_idx, caption; sorted by (image_id, cap_idx)
  leakage_report.json      exact/near duplicate images and duplicate captions across splits

Canonical ordering: within a split, image i owns captions 5i..5i+4. Every
downstream script (features, training, evaluation) relies on this order.

Usage: python data.py [--expected-checksums checksums.json]
"""

import argparse
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import DATA_DIR, banner, provenance, read_json, sha256_file, sha256_json, write_json

# Third-party mirror of the original Flickr8k release. Checksums are recorded on
# first download; pin them with --expected-checksums for every later run.
ARCHIVES = {
    "Flickr8k_Dataset.zip": "https://github.com/Avaneesh40585/Flickr8k-Dataset/releases/download/v1.0/Flickr8k_Dataset.zip",
    "Flickr8k_text.zip": "https://github.com/Avaneesh40585/Flickr8k-Dataset/releases/download/v1.0/Flickr8k_text.zip",
}
SPLIT_FILES = {"train": "Flickr_8k.trainImages.txt", "val": "Flickr_8k.devImages.txt", "test": "Flickr_8k.testImages.txt"}
EXPECTED = {"train": 6000, "val": 1000, "test": 1000}
CAPTIONS_PER_IMAGE = 5
NEAR_DUP_HAMMING = 2


def download(url, path):
    if path.is_file() and path.stat().st_size > 0:
        print(f"reuse {path.name} ({path.stat().st_size / 2**20:.1f} MB)")
        return
    print(f"download {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(path)


def find_unique(root, name):
    hits = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == name.lower() and "__MACOSX" not in p.parts]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} under {root}, found {len(hits)}: {hits[:5]}")
    return hits[0]


def find_image_dir(root):
    best, best_n = None, 0
    for d in [root, *[p for p in root.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts]]:
        n = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
        if n > best_n:
            best, best_n = d, n
    if best_n < 8000:
        raise FileNotFoundError(f"no directory with >=8000 jpg images under {root} (best: {best_n})")
    return best


def read_ids(path):
    return sorted({line.strip() for line in open(path, encoding="utf-8") if line.strip()})


def parse_captions(token_file):
    rows = []
    for line in open(token_file, encoding="utf-8", errors="strict"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        key, caption = line.split("\t", 1)
        image_id, idx = key.split("#", 1)
        rows.append((image_id.strip(), int(idx), caption.strip()))
    return pd.DataFrame(rows, columns=["image_id", "cap_idx", "caption"])


def dhash(path, size=8):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def popcount64(x):
    x = x.astype(np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def leakage_report(image_dir, ids, tables):
    report = {}
    # 1. exact duplicate image bytes across splits
    file_hash = {s: {i: sha256_file(image_dir / i) for i in ids[s]} for s in ids}
    owner = defaultdict(list)
    for s in ids:
        for i, h in file_hash[s].items():
            owner[h].append((s, i))
    exact = [v for v in owner.values() if len({s for s, _ in v}) > 1]
    report["exact_duplicate_images_across_splits"] = exact

    # 2. near-duplicate images (dHash, Hamming <= NEAR_DUP_HAMMING) train vs val/test
    hashes = {s: np.array([dhash(image_dir / i) for i in ids[s]], dtype=np.uint64) for s in ids}
    near = []
    for other in ("val", "test"):
        for j, h in enumerate(hashes[other]):
            dist = popcount64(np.bitwise_xor(hashes["train"], h))
            for k in np.nonzero(dist <= NEAR_DUP_HAMMING)[0]:
                near.append({"train": ids["train"][k], other: ids[other][j], "hamming": int(dist[k])})
    report["near_duplicate_images_train_vs_eval"] = near

    # 3. identical normalised captions shared between train and evaluation splits
    norm = {s: set(tables[s]["caption"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" "))
            for s in tables}
    report["duplicate_caption_strings"] = {
        "train_val": len(norm["train"] & norm["val"]),
        "train_test": len(norm["train"] & norm["test"]),
        "note": "Generic short captions recur naturally; reported, not treated as leakage.",
    }
    report["image_file_sha256"] = file_hash
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-checksums", default=None,
                    help="JSON {archive_name: sha256}. When given, any mismatch aborts.")
    ap.add_argument("--allow-image-leakage", action="store_true",
                    help="Do not abort when exact duplicate images exist across splits.")
    args = ap.parse_args()

    banner("FLICKR8K PREPARATION")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_sha = {}
    for name, url in ARCHIVES.items():
        path = DATA_DIR / name
        download(url, path)
        archive_sha[name] = sha256_file(path)
        print(f"sha256 {name}: {archive_sha[name]}")
    if args.expected_checksums:
        expected = read_json(args.expected_checksums)
        for name, digest in expected.items():
            if archive_sha.get(name) != digest:
                raise RuntimeError(f"checksum mismatch for {name}: {archive_sha.get(name)} != {digest}")
        print("ARCHIVE_CHECKSUMS: PASS (pinned)")
    else:
        print("ARCHIVE_CHECKSUMS: RECORDED (not pinned; pass --expected-checksums to enforce)")

    extract_root = DATA_DIR / "extracted"
    marker = extract_root / ".complete"
    if not marker.is_file():
        if extract_root.exists():
            shutil.rmtree(extract_root)
        for name in ARCHIVES:
            with zipfile.ZipFile(DATA_DIR / name) as zf:
                zf.extractall(extract_root)
        marker.write_text("ok")

    image_dir = find_image_dir(extract_root)
    token_file = find_unique(extract_root, "Flickr8k.token.txt")
    split_paths = {s: find_unique(extract_root, f) for s, f in SPLIT_FILES.items()}

    ids = {s: read_ids(p) for s, p in split_paths.items()}
    for s, n in EXPECTED.items():
        assert len(ids[s]) == n, f"{s}: expected {n} images, got {len(ids[s])}"
    assert set(ids["train"]).isdisjoint(ids["val"])
    assert set(ids["train"]).isdisjoint(ids["test"])
    assert set(ids["val"]).isdisjoint(ids["test"])
    missing = [i for s in ids for i in ids[s] if not (image_dir / i).is_file()]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:5]}"
    print("SPLITS: 6000/1000/1000 disjoint, all image files present")

    captions = parse_captions(token_file)
    tables = {}
    for s in ids:
        t = captions[captions["image_id"].isin(set(ids[s]))].sort_values(["image_id", "cap_idx"]).reset_index(drop=True)
        counts = t.groupby("image_id").size()
        assert len(counts) == len(ids[s]) and (counts == CAPTIONS_PER_IMAGE).all(), f"{s}: not 5 captions per image"
        assert t["image_id"].iloc[::CAPTIONS_PER_IMAGE].tolist() == ids[s], f"{s}: canonical order broken"
        assert (t["caption"].str.len() > 0).all()
        tables[s] = t
        t.to_csv(DATA_DIR / f"captions_{s}.csv", index=False)
    print("CAPTIONS: 30000/5000/5000, exactly 5 per image, canonical order verified")

    banner("LEAKAGE CHECKS")
    report = leakage_report(image_dir, ids, tables)
    n_exact = len(report["exact_duplicate_images_across_splits"])
    n_near = len(report["near_duplicate_images_train_vs_eval"])
    print(f"exact duplicate images across splits : {n_exact}")
    print(f"near-duplicate images (dHash<={NEAR_DUP_HAMMING})  : {n_near}")
    print(f"shared caption strings train/val/test: {report['duplicate_caption_strings']}")
    write_json(DATA_DIR / "leakage_report.json", report)
    if n_exact and not args.allow_image_leakage:
        raise RuntimeError("exact duplicate images across splits; inspect leakage_report.json")

    manifest = {
        "dataset": "Flickr8k",
        "source_urls": ARCHIVES,
        "archive_sha256": archive_sha,
        "archive_checksums_pinned": bool(args.expected_checksums),
        "image_dir": str(image_dir),
        "caption_file": str(token_file),
        "caption_file_sha256": sha256_file(token_file),
        "split_files": {s: str(p) for s, p in split_paths.items()},
        "split_file_sha256": {s: sha256_file(p) for s, p in split_paths.items()},
        "image_list_sha256": {s: sha256_json(ids[s]) for s in ids},
        "caption_table_sha256": {s: sha256_file(DATA_DIR / f"captions_{s}.csv") for s in ids},
        "counts": {s: {"images": len(ids[s]), "captions": len(tables[s])} for s in ids},
        "captions_per_image": CAPTIONS_PER_IMAGE,
        "leakage": {"exact_duplicate_images": n_exact, "near_duplicate_images": n_near,
                    "report_sha256": sha256_file(DATA_DIR / "leakage_report.json")},
        "synthetic_data": False,
    }
    manifest["provenance"] = provenance(include_artifacts=False)
    write_json(DATA_DIR / "manifest.json", manifest)
    print(f"manifest: {DATA_DIR / 'manifest.json'}")
    print("DATASET_READY: PASS")


if __name__ == "__main__":
    main()
