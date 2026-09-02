"""
Image-Text Retrieval Evaluation Metrics (R@1, R@5, R@10, Mean Rank, Median Rank, mR).
Supports Flickr8k Multi-Caption Multiplicity Evaluation.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional


def compute_retrieval_recalls(
    similarity_matrix: torch.Tensor,
    img_to_txt_map: Optional[Dict[int, List[int]]] = None,
    txt_to_img_map: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """
    Compute Image-to-Text and Text-to-Image Recall@K, Mean Rank, Median Rank metrics.

    Args:
        similarity_matrix (torch.Tensor): Cosine similarity matrix (N_img, N_txt)
        img_to_txt_map (Optional[Dict[int, List[int]]]): Map of img_idx -> list of ground truth caption indices.
        txt_to_img_map (Optional[Dict[int, int]]): Map of txt_idx -> ground truth image index.

    Returns:
        Dict[str, float]: Dictionary containing R@1, R@5, R@10 for I2T & T2I, Mean Ranks, and mR.
    """
    if torch.is_tensor(similarity_matrix):
        sim_mat = similarity_matrix.detach().cpu().numpy()
    else:
        sim_mat = np.array(similarity_matrix)

    N_img, N_txt = sim_mat.shape

    # 1. Image-to-Text (I2T) Evaluation
    ranks_i2t = []
    for i in range(N_img):
        score_row = sim_mat[i]
        sorted_indices = np.argsort(-score_row)  # Descending rank order

        if img_to_txt_map and i in img_to_txt_map:
            target_txt_ids = set(img_to_txt_map[i])
            # Find the minimum rank among all ground-truth captions for this image
            found_ranks = [np.where(sorted_indices == txt_id)[0][0] for txt_id in target_txt_ids if txt_id < N_txt]
            min_rank = min(found_ranks) if found_ranks else N_txt
        else:
            # Default 1-to-1 matching assumption if map not provided
            target_id = i % N_txt
            min_rank = np.where(sorted_indices == target_id)[0][0]

        ranks_i2t.append(min_rank)

    ranks_i2t_arr = np.array(ranks_i2t)

    i2t_r1 = (ranks_i2t_arr < 1).mean() * 100.0
    i2t_r5 = (ranks_i2t_arr < 5).mean() * 100.0
    i2t_r10 = (ranks_i2t_arr < 10).mean() * 100.0
    i2t_mean_rank = float(np.mean(ranks_i2t_arr) + 1.0)
    i2t_median_rank = float(np.median(ranks_i2t_arr) + 1.0)

    # 2. Text-to-Image (T2I) Evaluation
    ranks_t2i = []
    for j in range(N_txt):
        score_col = sim_mat[:, j]
        sorted_indices = np.argsort(-score_col)  # Descending rank order

        if txt_to_img_map and j in txt_to_img_map:
            target_img_id = txt_to_img_map[j]
            rank = np.where(sorted_indices == target_img_id)[0][0] if target_img_id < N_img else N_img
        else:
            target_id = j % N_img
            rank = np.where(sorted_indices == target_id)[0][0]

        ranks_t2i.append(rank)

    ranks_t2i_arr = np.array(ranks_t2i)

    t2i_r1 = (ranks_t2i_arr < 1).mean() * 100.0
    t2i_r5 = (ranks_t2i_arr < 5).mean() * 100.0
    t2i_r10 = (ranks_t2i_arr < 10).mean() * 100.0
    t2i_mean_rank = float(np.mean(ranks_t2i_arr) + 1.0)
    t2i_median_rank = float(np.median(ranks_t2i_arr) + 1.0)

    mean_recall = (i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6.0

    return {
        "retrieval/i2t_r1": float(i2t_r1),
        "retrieval/i2t_r5": float(i2t_r5),
        "retrieval/i2t_r10": float(i2t_r10),
        "retrieval/i2t_mean_rank": float(i2t_mean_rank),
        "retrieval/i2t_median_rank": float(i2t_median_rank),
        "retrieval/t2i_r1": float(t2i_r1),
        "retrieval/t2i_r5": float(t2i_r5),
        "retrieval/t2i_r10": float(t2i_r10),
        "retrieval/t2i_mean_rank": float(t2i_mean_rank),
        "retrieval/t2i_median_rank": float(t2i_median_rank),
        "retrieval/mean_recall": float(mean_recall),
    }
