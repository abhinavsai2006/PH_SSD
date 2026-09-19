# Methodology reconstruction — native Mamba-2 HEDO-HVSC (stable runner)

Source of truth: `native_train_runner.py` in this folder. Line numbers refer to the file whose SHA256 is recorded in
every run's `config.json` / `methodology_manifest.json` (`runner_sha256`). The runtime manifest additionally
records the observed input/output shape, dtype and device of every module on a real Flickr8k batch.
Provenance chain: the runner of `HEDO_HVSC_NATIVE_MAMBA2_READY_TO_TRAIN (1).ipynb` (whose run diverged, kept as
`native_train_runner_ORIGINAL_FAILED_NaN.py`) → v1 numerical repairs → v2 audit repairs. All edits are asserted
exact replacements (`patch_runner.py`, `patch_v2.py`).

## 1. Data
| Item | Value | Location |
|---|---|---|
| Dataset | Flickr8k, official `Flickr_8k.{train,dev,test}Images.txt`, `Flickr8k.token.txt` | `load_flickr8k` (1413) |
| Counts | 6000 / 1000 / 1000 images; 30000 / 5000 / 5000 captions; 5 captions per image | asserted in `load_flickr8k`; recomputed from raw files in `dataset_gate` (2537) |
| Leakage | pairwise split overlap = 0, no duplicate ids, every image file exists, loader tables equal the raw caption records | `dataset_gate` |
| Fingerprint | SHA256 of caption file, split files, sorted split ids, caption tables | `dataset_fingerprint.json` |
| Train batches | 8 unique images × 5 captions = 40 (`AtomicGroupedBatchSampler`, python `random`, seeded) | 1675 |
| Validation | `val_loader`, no shuffle; used only inside `extract_embeddings` under `torch.no_grad` (asserted) for checkpoint selection | 1983 |
| Test | `test_loader`; evaluated once, after reloading `checkpoint_best.pt` | `main` |

## 2. Model (per modality; image L=196, text L=64 right-padded)
| Component | Class / function | In → out | Trainable | Notes |
|---|---|---|---|---|
| Image preprocessing | `ViT_B_16_Weights.IMAGENET1K_V1.transforms()` (1579) | PIL → [3,224,224] | – | resize 256 bilinear, centre crop 224, ImageNet mean/std |
| Vision backbone | `FrozenVisionBackbone` (1825) | [B,3,224,224] → [B,196,768] | frozen, eval | `_process_input` → class token → `encoder` (adds `pos_embedding`, dropout, 12 blocks, LayerNorm) → drop class token; `forward_logits` must equal `vit(x)` |
| Text backbone | `FrozenLanguageBackbone` (1865) | ids/mask [B,64] → [B,64,768] | frozen, eval | `RobertaModel` from `roberta-base`, `add_pooling_layer=False`, `last_hidden_state`; missing keys abort |
| Backbone precision | `get_features` (1927) | FP16 autocast → `.float()` | – | |
| Projection | `img_proj`, `txt_proj` `nn.Linear(768,128)` | [B,L,768] → [B,L,128] | yes | FP32 |
| HEDO | `HEDO` (186) | [B,L,128] → [B,L,128] | yes | q=W_q x, p=W_p x, γ=sigmoid(g)∈(0,1)^128, q'=q+p, p'=p−γ⊙q', out=W_o(q'+p') |
| Native Mamba-2 | `NativeMamba2SequenceBlock` (272) → `mamba_ssm.modules.mamba2.Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=64)` + `LayerNorm(128)` | [B,L,128] → [B,L,128] | yes | text input multiplied by mask before the causal scan |
| Boundary states | same block | → [B,K,128], K=⌈L/16⌉ (image 13, text 4) + chunk mask [B,K] | – | state at the last valid token of each chunk |
| HVSC | `ChunkWiseHVSC` (502) | boundary states → z [B,128], symmetric KL scalar | yes | mask-weighted mean of boundary states; μ=W_μ h, logvar=clamp(W_lv h, −6, 4); std=clamp(exp(½logvar),1e-3,20); training z=μ+ε·std, inference z=μ; KL(p‖q) summed over 128 dims, batch mean, symmetric average, computed in FP32 with variance floor 1e-6 and (Δμ)²≤1e4 |
| Without HVSC | `HEDO_HVSC_Model.forward` (871) | mask-weighted mean of Mamba-2 outputs | – | |
| Head | `head_img`, `head_txt` `nn.Linear(128,128)` → L2 normalise | [B,128] → [B,128] | yes | |
| Temperature | `logit_scale` (init log(1/0.07)) | scalar | yes | used as exp(clamp(·, log 1, log 100)); parameter clamped after each step |

Important property (audited in the TRAIN_INFERENCE_CONSISTENCY gate): with HVSC the **retrieval embedding is built
from the sampled latent during training and from the posterior mean at inference**. The runner quantifies this on the
full validation split (mean cosine between sampled- and mean-based embeddings, and both mean recalls) and requires
cosine ≥ 0.90 (pre-specified).

## 3. Objective
`SymmetricMultiPositiveInfoNCELoss` (1009): similarity S = z_img[unique 8] · z_txt[40]ᵀ · scale → [8, 40].
L_i2t = −mean_u mean_{c∈P(u)} log softmax_row(S)[u,c] (|P(u)|=5); L_t2i = −mean_c log softmax_col(S)[owner(c),c].
L = ½(L_i2t + L_t2i) + λ_KL · KL_sym, λ_KL = 1e-3. An independent logsumexp implementation must match to 1e-4 (LOSS_GATE).

## 4. Optimisation
| Item | Value |
|---|---|
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 (all groups), betas default |
| Parameter groups | hedo / hvsc / mamba2 / projection_head_temperature (same hyperparameters; membership asserted equal to all trainable params) |
| Scheduler | CosineAnnealingLR stepped per batch, T_max = 750 × epochs |
| Gradient clipping | global norm 1.0 (pre-clip norm logged) |
| Epochs | 10 (`HEDO_EPOCHS`) |
| Precision | FP32 trainable stack, no autocast, no GradScaler |
| Finite checks | every batch: features, HEDO intermediates, Mamba outputs, boundary states, HVSC μ/logvar/variance/std/KL/z, embeddings, logit scale, similarity, log-probabilities, InfoNCE, total loss, every gradient, every parameter, every optimizer-state tensor |
| Seeds | 42, 43, 44; python `random`, NumPy, torch CPU, torch CUDA; cudnn deterministic, benchmark off. Mamba-2 Triton backward is not bit-deterministic: runs are seed-controlled, not bit-reproducible |

## 5. Selection, evaluation, artifacts
- Best checkpoint: max validation mean recall (earlier epoch on ties). Last checkpoint retained.
- Test: evaluated exactly once on the reloaded best checkpoint, after the checkpoint-reload and consistency gates.
- Retrieval (`compute_retrieval_metrics`, 2107): I2T 1000 images × 5000 captions, rank = best-ranked of the 5 positives;
  T2I 5000 captions × 1000 images; R@1/5/10, MedR, MeanR (1-based), mean recall = mean of the six recalls.
  `extract_embeddings` asserts exactly (1000,128) and (5000,128), the evaluated ids equal the split ids.
- Run layout: `results/seed{S}/{config}/` with `config.json`, `dataset_fingerprint.json`, `dataset_manifest.json`,
  `methodology_manifest.json`, `gate_report.json`, `training_log.jsonl`, `epoch_metrics.csv`, `checkpoint_best.pt`,
  `checkpoint_last.pt`, `test_results.json`, `efficiency.json`, test embeddings, `run_status.json`.
  Existing attempts are renamed `__superseded_<time>`, never overwritten.
- Methodology hash: SHA256 of (runner SHA256, shared hyperparameters) — identical for all seeds and ablation configs;
  `results/final_methodology_hash.txt`. Training refuses to start without `GO_NO_GO.json` = GO for that hash.

## 6. Ablations
| Config | use_hedo | use_hvsc |
|---|---|---|
| baseline | no | no |
| mamba2_hedo | yes | no |
| mamba2_hvsc | no | yes |
| full_hedo_hvsc | yes | yes |
Only these flags differ (ABLATION_GATE checks that shared parameter names/shapes are identical and extra parameters
belong only to `hedo_*` / `hvsc.*`).
