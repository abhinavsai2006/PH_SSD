"""Retrieval metrics under the canonical order: image i owns captions 5i..5i+4.

Ranks are 0-based; MedR / MeanR are reported 1-based. i2t uses the best-ranked
of the 5 ground-truth captions (standard Flickr protocol).

Re-ranking (exchange models): for each query the top-K gallery items by dual
(pass-1) similarity are re-ordered by the exchange score; items outside the
top-K keep their dual order after them.
"""

import numpy as np

CAPTIONS_PER_IMAGE = 5
RECALL_KS = (1, 5, 10)


def caption_owner(n_img, k=CAPTIONS_PER_IMAGE):
    return np.repeat(np.arange(n_img), k)


def _ranks_from_orders(order_i2t, order_t2i, owner):
    n_img = order_i2t.shape[0]
    i2t = (owner[order_i2t] == np.arange(n_img)[:, None]).argmax(axis=1)
    t2i = (order_t2i == owner[:, None]).argmax(axis=1)
    return i2t, t2i


def retrieval_ranks(img, txt, k=CAPTIONS_PER_IMAGE):
    n_img = img.shape[0]
    assert txt.shape[0] == k * n_img, (img.shape, txt.shape)
    sim = img.astype(np.float64) @ txt.astype(np.float64).T
    return _ranks_from_orders(np.argsort(-sim, axis=1, kind="stable"), np.argsort(-sim.T, axis=1, kind="stable"),
                              caption_owner(n_img, k))


def topk_candidates(img, txt, k):
    sim = img.astype(np.float64) @ txt.astype(np.float64).T
    i2t = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    t2i = np.argsort(-sim.T, axis=1, kind="stable")[:, :k]
    return i2t, t2i


def _rerank_orders(sim, topk_idx, topk_scores):
    orders = np.argsort(-sim, axis=1, kind="stable")
    out = np.empty_like(orders)
    k = topk_idx.shape[1]
    for r in range(sim.shape[0]):
        head = topk_idx[r][np.argsort(-topk_scores[r], kind="stable")]
        rest = orders[r][~np.isin(orders[r], head)]
        out[r, :k], out[r, k:] = head, rest
    return out


def reranked_ranks(img, txt, rerank):
    n_img = img.shape[0]
    sim = img.astype(np.float64) @ txt.astype(np.float64).T
    order_i2t = _rerank_orders(sim, rerank["i2t_idx"], rerank["i2t_score"])
    order_t2i = _rerank_orders(sim.T, rerank["t2i_idx"], rerank["t2i_score"])
    return _ranks_from_orders(order_i2t, order_t2i, caption_owner(n_img))


def metrics_from_ranks(i2t, t2i):
    m = {}
    for name, r in (("i2t", i2t), ("t2i", t2i)):
        for k in RECALL_KS:
            m[f"{name}_r{k}"] = float(np.mean(r < k) * 100.0)
        m[f"{name}_medr"] = float(np.median(r + 1))
        m[f"{name}_meanr"] = float(np.mean(r + 1))
    m["mean_recall"] = float(np.mean([m[f"{d}_r{k}"] for d in ("i2t", "t2i") for k in RECALL_KS]))
    return m


def retrieval_metrics(img, txt, rerank=None):
    """Primary metrics (re-ranked when `rerank` is given) plus the dual metrics under "dual_*"."""
    dual = metrics_from_ranks(*retrieval_ranks(img, txt))
    if rerank is None:
        return {**dual, **{f"dual_{k}": v for k, v in dual.items()}}
    primary = metrics_from_ranks(*reranked_ranks(img, txt, rerank))
    return {**primary, **{f"dual_{k}": v for k, v in dual.items()}}
