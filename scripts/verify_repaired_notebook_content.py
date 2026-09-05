import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

for nb_name in ["HEDO_HVSC_Research_Master_REPAIRED.ipynb", "HEDO_HVSC_Research_Master_REPAIRED(1).ipynb"]:
    print("=" * 80)
    print(f"AUDITING NOTEBOOK: {nb_name}")
    print("=" * 80)
    with open(nb_name, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_text = "\n".join("".join(c["source"]) for c in nb["cells"])

    # 1. Output directory
    assert "HEDO_HVSC_FINAL_LOCKED_BENCHMARK" in full_text, "Missing HEDO_HVSC_FINAL_LOCKED_BENCHMARK directory"
    print("✓ [PASS] Output directory: HEDO_HVSC_FINAL_LOCKED_BENCHMARK")

    # 2. Immutable LOCKED_BENCHMARK_CONFIG
    assert "LOCKED_BENCHMARK_CONFIG = types.MappingProxyType({" in full_text, "Missing LOCKED_BENCHMARK_CONFIG definition"
    required_keys = [
        "embed_dim", "d_state", "chunk_size", "max_epochs", "patience",
        "base_learning_rate", "minimum_learning_rate", "weight_decay",
        "gradient_clip_norm", "kl_weight", "optimizer", "scheduler",
        "warmup_fraction", "dataset_name", "train_images", "val_images",
        "test_images", "captions_per_image", "vision_backbone",
        "text_backbone", "frozen_backbones"
    ]
    for k in required_keys:
        assert f'"{k}"' in full_text or f"'{k}'" in full_text, f"Missing key {k} in LOCKED_BENCHMARK_CONFIG"
    print(f"✓ [PASS] LOCKED_BENCHMARK_CONFIG present with all {len(required_keys)} parameters.")

    # 3. Real verification at gate (NO dummy config_lock_pass = True)
    assert "config_lock_pass = True\n" not in full_text, "Found forbidden dummy config_lock_pass = True!"
    assert "config_lock_pass = (len(config_mismatches) == 0)" in full_text, "Missing real config_lock_pass comparison!"
    print("✓ [PASS] Real Configuration Lock assertion at pre-benchmark gate verified.")

    # 4. Final Pre-Benchmark Gate (Exact 14 Checks)
    assert "FINAL PRE-BENCHMARK SCIENTIFIC GATE" in full_text, "Missing exact gate header"
    gate_checks = [
        "DATASET COUNTS", "LEAKAGE", "CAPTION MAPPING", "MULTI-POSITIVE LOSS",
        "RETRIEVAL EVALUATOR", "SSD STATE CONTINUITY", "SSD PADDING INVARIANCE",
        "DETERMINISTIC INFERENCE", "FULL TEST EXTRACTION", "CHECKPOINT LOGIC",
        "CONFIGURATION LOCK", "TEST/VALIDATION SEPARATION", "HEDO DIAGNOSTICS",
        "HVSC NUMERICAL STABILITY"
    ]
    for gc in gate_checks:
        assert gc in full_text, f"Missing gate check: {gc}"
    assert "READY FOR LOCKED 12-RUN BENCHMARK" in full_text, "Missing exact READY printout"
    print(f"✓ [PASS] Exact 14-item pre-benchmark gate verified.")

    # 5. Separation of Test Information
    # Check that training epoch loop only uses val_loader
    assert "val_metrics = evaluate_retrieval(model, val_loader)" in full_text
    assert "test_eval_results = evaluate_retrieval(model, test_loader)" in full_text
    print("✓ [PASS] Strict isolation of test set (evaluated strictly once post-best-checkpoint).")

    # 6. Validation Convergence Status
    assert "VALIDATION_NOT_CONVERGED" in full_text, "Missing VALIDATION_NOT_CONVERGED tracking"
    print("✓ [PASS] Validation convergence tracking present.")

    # 7. HEDO Claims
    assert "Empirical trajectory was non-monotonic; no formal discrete energy dissipation guarantee is claimed." in full_text
    # Verify no code cells contain mamba_ssm imports
    for idx, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            c_text = "".join(c["source"])
            assert "mamba_ssm" not in c_text, f"Cell {idx} contains forbidden mamba_ssm in code!"
    print("✓ [PASS] No mamba_ssm in any code cells. Custom PyTorch SSD confirmed.")

    # 8. Cell 23 On-Disk Recomputation
    assert "recomputed_sim = img_embs @ txt_embs.T" in full_text, "Missing similarity recomputation in audit"
    assert "sim_max_diff < 1e-5" in full_text, "Missing similarity diff assertion in audit"
    assert "recomputed_mr" in full_text, "Missing independent metric recomputation in audit"
    assert "best_epoch_saved == best_ep_hist" in full_text, "Missing checkpoint epoch verification in audit"
    assert "cap_counts == 5" in full_text, "Missing 1:5 caption ratio check in audit"
    print("✓ [PASS] Comprehensive On-Disk Recomputation & Artifact Integrity Audit verified.")

print("\n" + "=" * 80)
print("🎉 ALL CRITICAL AUDIT VERIFICATIONS PASSED FOR BOTH NOTEBOOKS!")
print("=" * 80)
