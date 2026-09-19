"""Independent retrieval evaluator.

Shares no code with metrics.py or the model. Reads only the saved embeddings, the
saved image-id order, the optional re-ranking file, and the dataset caption
table, then recomputes all metrics by *counting strictly higher scores* (instead
of sorting) and compares them with test_results.json.

Re-ranked rank of gallery item g for a query with top-K set T:
  g in T     -> #{t in T : s2(t) > s2(g)}
  g not in T -> K + #{h not in T : s1(h) > s1(g)}

Exit 0 iff all metrics match within --tol, the id order is consistent, and there are no ties.
Usage: python evaluate_independent.py --run <run_dir> [--tol 1e-6]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def count_rank(s1_row, positive_cols, topk=None, s2=None):
    """Best (minimum) 0-based rank over positive columns; also returns the number of exact ties."""
    ties = 0
    if topk is None:
        best = s1_row[positive_cols].max()
        ties += int((s1_row == best).sum()) - 1
        return int((s1_row > best).sum()), ties
    in_top = np.zeros(s1_row.shape[0], dtype=bool)
    in_top[topk] = True
    s2_full = np.full(s1_row.shape[0], np.nan)
    s2_full[topk] = s2
    ranks = []
    for g in positive_cols:
        if in_top[g]:
            ranks.append(int((s2 > s2_full[g]).sum()))
            ties += int((s2 == s2_full[g]).sum()) - 1
        else:
            outside = s1_row[~in_top]
            ranks.append(len(topk) + int((outside > s1_row[g]).sum()))
            ties += int((outside == s1_row[g]).sum()) - 1
    return min(ranks), ties


def compute(sim, owner, rerank):
    n_img, n_txt = sim.shape
    i2t, t2i, ties = [], [], 0
    for i in range(n_img):
        pos = np.nonzero(owner == i)[0]
        r, t = count_rank(sim[i], pos, *((rerank["i2t_idx"][i], rerank["i2t_score"][i]) if rerank else (None, None)))
        i2t.append(r)
        ties += t
    for c in range(n_txt):
        r, t = count_rank(sim[:, c], np.array([owner[c]]),
                          *((rerank["t2i_idx"][c], rerank["t2i_score"][c]) if rerank else (None, None)))
        t2i.append(r)
        ties += t
    i2t, t2i = np.array(i2t), np.array(t2i)
    m = {}
    for name, r in (("i2t", i2t), ("t2i", t2i)):
        for k in (1, 5, 10):
            m[f"{name}_r{k}"] = 100.0 * np.count_nonzero(r < k) / len(r)
        m[f"{name}_medr"] = float(np.median(r + 1))
        m[f"{name}_meanr"] = float(np.mean(r + 1))
    m["mean_recall"] = sum(m[f"{d}_r{k}"] for d in ("i2t", "t2i") for k in (1, 5, 10)) / 6.0
    return m, ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("HEDO_WORK", "/content/hedo_work"),
                                                       "data", "flickr8k"))
    args = ap.parse_args()
    run = Path(args.run)

    img = np.load(run / "test_image_embeddings.npy").astype(np.float64)
    txt = np.load(run / "test_text_embeddings.npy").astype(np.float64)
    saved_ids = json.load(open(run / "test_image_ids.json", encoding="utf-8"))
    stored = json.load(open(run / "test_results.json", encoding="utf-8"))
    rerank = dict(np.load(run / "test_rerank.npz")) if (run / "test_rerank.npz").is_file() else None

    with open(Path(args.data_dir) / "captions_test.csv", newline="", encoding="utf-8") as f:
        caption_image = [row["image_id"] for row in csv.DictReader(f)]
    image_order = list(dict.fromkeys(caption_image))
    problems = []
    if image_order != saved_ids:
        problems.append("image id order differs from dataset caption table")
    if img.shape[0] != len(image_order) or txt.shape[0] != len(caption_image):
        problems.append(f"shape mismatch img={img.shape} txt={txt.shape}")
    index_of = {iid: i for i, iid in enumerate(image_order)}
    owner = np.array([index_of[c] for c in caption_image])
    sim = img @ txt.T

    dual, ties_dual = compute(sim, owner, None)
    recomputed = {f"dual_{k}": v for k, v in dual.items()}
    ties = ties_dual
    if rerank is not None:
        primary, ties_rr = compute(sim, owner, rerank)
        ties += ties_rr
    else:
        primary = dual
    recomputed.update(primary)

    diffs = {k: abs(recomputed[k] - stored[k]) for k in recomputed}
    worst = max(diffs.values())
    if worst > args.tol:
        problems.append(f"metric mismatch (max |diff|={worst:.3g})")
    if ties:
        problems.append(f"{ties} exact score ties; ranks are tie-order dependent")

    report = {"recomputed": recomputed, "abs_diff": diffs, "max_abs_diff": worst, "tolerance": args.tol,
              "reranked": rerank is not None, "score_ties": ties, "problems": problems, "pass": not problems}
    with open(run / "independent_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"INDEPENDENT_EVALUATOR: {'PASS' if not problems else 'FAIL'} max|diff|={worst:.3g} "
          f"MR={recomputed['mean_recall']:.4f} reranked={rerank is not None} {problems if problems else ''}")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
