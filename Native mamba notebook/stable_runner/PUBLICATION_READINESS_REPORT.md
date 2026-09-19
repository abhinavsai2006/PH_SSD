# Publication readiness report — native Mamba-2 HEDO-HVSC (Flickr8k)

Runner: `native_train_runner.py` (SHA256 `63ca95316c3281ae…`), notebook `HEDO_HVSC_NATIVE_MAMBA2_STABLE_FP32.ipynb`.
Evidence levels used below:
- **source** = verified by reading the code;
- **CPU test** = executed in this session on CPU (synthetic data, stand-in Mamba2 — tests code paths only, never results);
- **T4** = requires the real run (not executed; there is no GPU in this environment).
No status is PASS merely because a print statement says PASS.

**Overall decision: NO-GO for the benchmark and NOT READY for submission.** No valid native-Mamba-2 Flickr8k result
exists yet, and the CPU dry run shows a train/inference inconsistency in HVSC that is expected to block GO on the T4
(section K).

| Category | Status | Evidence |
|---|---|---|
| A. Architecture correctness | WARNING | source: HEDO → Mamba2 → HVSC → head per modality, as specified. CPU test: forward/backward on all 4 configs. The HVSC design uses the sampled latent as the training embedding (see K). |
| B. Native Mamba-2 correctness | PASS (source), pending T4 | `native_mamba2_gate` checks `isinstance(..., mamba_ssm.modules.mamba2.Mamba2)` on `mamba2_img.mamba` / `mamba2_txt.mamba`, constructor attributes (128/64/4/2/64), CUDA device, and finite non-zero gradients on every Mamba parameter. CPU test with a stand-in exercised the logic only. Real kernel execution needs the T4. |
| C. Dataset correctness | PASS (CPU test on raw files), pending T4 | `dataset_gate` recomputes from raw split/caption files: 6000/1000/1000, 30000/5000/5000, 5 per image, zero overlaps, no duplicate ids, files exist, loader tables equal raw records; fingerprint saved. Found and fixed: the original notebook never wrote the `manifest.json` its runner reads. |
| D. Loss correctness | PASS (CPU test) | `LOSS_GATE`: runner loss equals an independent logsumexp implementation to <1e-4 on an 8×40 matrix; aligned floor ½ln5 reproduced. `MULTIPOSITIVE_GEOMETRY_GATE`: 5 positives per image, 1 per caption. |
| E. Numerical stability | PASS (CPU test), pending T4 | FP32 trainable stack, no GradScaler; finite checks on every stage, gradient, parameter and optimizer-state tensor; injected NaN/Inf stops at the first bad tensor with statistics, `run_status.json` (failure_type, first_bad_tensor, epoch, batch, traceback), exit 2. The original divergence mechanism (FP16 + GradScaler + logvar [−10, 10]) is removed but has not yet been re-run on the T4. |
| F. Evaluation correctness | PASS (CPU test) | `RETRIEVAL_EVALUATOR_GATE`: handcrafted 2-image / 10-caption case with analytically known ranks reproduced exactly. Full-split shape asserts (1000×128, 5000×128); evaluated ids must equal the split ids; posterior-mean inference; test evaluated once on the reloaded validation-best checkpoint. |
| G. Experimental fairness | PASS (CPU test) | `ABLATION_GATE`: shared parameter names/shapes identical across the 4 configs; extra parameters only `hedo_*` / `hvsc.*`; one shared hyperparameter set in the methodology hash. |
| H. Statistical reproducibility | WARNING | `SEED_REPRODUCIBILITY_GATE`: bitwise-equal initial weights, identical eval forward, identical batch order for the same seed (CPU test). Mamba-2 Triton backward is not bit-deterministic, so the runs are seed-controlled rather than bit-reproducible. Three seeds give a t-interval with df = 2: wide, no significance claims. |
| I. Efficiency measurement | WARNING | Per run: trainable/frozen/total params, peak VRAM, time per epoch, head and end-to-end latency, throughput, FLOPs via `torch.utils.flop_counter` (custom Mamba-2 kernels not counted → lower bound). Measured on the run GPU only; no speed comparison against another architecture exists, so no "efficient" claim. |
| J. Ablation completeness | FAIL (not run) | The 4 configs × 3 seeds are orchestrated (cell 9) but have not been executed on the T4. |
| K. Claim/method consistency | FAIL | CPU test (deterministic synthetic features): sampled-vs-mean embedding cosine 0.56 (image) / 0.55 (text); validation mean recall 3.3 (sampled) vs 20.2 (mean); both HVSC configs about 63–66 MR points below the non-HVSC ones. On the T4 the gate threshold is 0.90, so NO-GO is expected unless real features behave differently. This is consistent with the chance-level ValMR (0.647) of the diverged run. Terminology is fixed: "Hamiltonian-inspired discrete dissipative coordinate-momentum transformation"; "HVSC aligns modality-specific boundary-state posterior distributions using symmetric KL regularization". No symplectic, conservation, information-bottleneck, hierarchical or cross-modal-feedback claims. |
| L. Artifact reproducibility | PASS (CPU test), pending T4 | Per run: config, dataset fingerprint/manifest, methodology manifest (module shapes/dtype/device/params/optimizer group), gate report, training_log.jsonl, epoch_metrics.csv, best/last checkpoints (model, optimizer, scheduler, seed, config, methodology hash, dataset fingerprint, validation and training metrics), test results, efficiency, embeddings, run_status. Results live under `results/seed{S}/{config}`; old attempts are renamed, never overwritten; completed runs are skipped. Aggregation, paired deltas, curves and the claim→evidence map are generated from these artifacts. |

## Gate status (rule 25)
- PASS on CPU test: DATASET, MULTIPOSITIVE_GEOMETRY, LOSS, RETRIEVAL_EVALUATOR, HEDO, HVSC, SEED_REPRODUCIBILITY, ABLATION, NUMERICAL_STABILITY, CHECKPOINT (reload reproduces validation metrics exactly).
- PASS on CPU test against a randomly initialised torchvision ViT-B/16: VISION_BACKBONE_FIDELITY (extraction-path logits equal `vit(x)`, diff 0.0; official IMAGENET1K_V1 transform).
- T4 only: ROBERTA_BACKBONE_FIDELITY (the real checkpoint is compared against the default RobertaModel), NATIVE_MAMBA2 with real kernels.
- **FAIL on CPU test: TRAIN_INFERENCE_CONSISTENCY** (0.55–0.56 < 0.90).

## Invalid results (never to be used)
See `invalid_results_registry.json`:
- the diverged FP16 run;
- the v13 custom-SSD benchmark (48.28 / 48.25) and its robustness/scaling numbers;
- the PH-SSD tables (83.2 / 64.8 / 87.47);
- the subset-evaluated `results/native_mamba_results.json`;
- all CPU harness runs;
- the unexecuted E-HEDO/X-HVSC pipeline.
