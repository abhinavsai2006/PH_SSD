"""Run experiment suites and rebuild results/all_runs.csv from run_summary.json files.

A run is skipped only if its run_summary.json says COMPLETED *and* its
code_sha256, config_sha256 and feature_manifest_sha256 match what this code
would produce now. Anything else is re-run from scratch (--overwrite).
The CSV is regenerated from disk every time; it is never appended to.

Suites:
  proposal            baseline, hedo_energy, hvsc_x, full_x  (proposed 2x2)
  proposal_ablations  full_x_affine, full_x_constdamp, full_x_noexchange, full_x_noprior,
                      baseline_param_matched_x, transformer_param_matched, transformer_x, no_mixer_meanpool
  proposal_chunks     full_x with hvsc_chunk_size in {8, 32}
  core       baseline, hedo, hvsc, full   (legacy)
  ablations  full_linear_operator, full_hvsc_deterministic, baseline_param_matched,
             no_mixer_meanpool, transformer_param_matched
  klsweep    full with kl_weight in {0, 1e-3, 1e-2}   (1e-4 is the core run)
  chunks     full with hvsc_chunk_size in {8, 32}     (16 is the core run)

Usage: python run_matrix.py --suite proposal --seeds 42 43 44 45 46 [--epochs 10]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import FEATURE_DIR, PACKAGE_DIR, RUNS_DIR, banner, code_sha256, read_json, sha256_file
from models import VARIANTS

DEFAULT_KL = 1e-4
DEFAULT_CHUNK = 16

SUITES = {
    # proposed model: 2x2 factorial of EnergyHEDO x ExchangeHVSC
    "proposal": [(v, DEFAULT_KL, DEFAULT_CHUNK) for v in ("baseline", "hedo_energy", "hvsc_x", "full_x")],
    "proposal_ablations": [(v, DEFAULT_KL, DEFAULT_CHUNK) for v in (
        "full_x_affine", "full_x_constdamp", "full_x_noexchange", "full_x_noprior", "baseline_param_matched_x",
        "transformer_param_matched", "transformer_x", "no_mixer_meanpool")],
    "proposal_chunks": [("full_x", DEFAULT_KL, c) for c in (8, 32)],
    # legacy (affine HEDO / index-KL HVSC)
    "core": [(v, DEFAULT_KL, DEFAULT_CHUNK) for v in ("baseline", "hedo", "hvsc", "full")],
    "ablations": [(v, DEFAULT_KL, DEFAULT_CHUNK) for v in (
        "full_linear_operator", "full_hvsc_deterministic", "baseline_param_matched",
        "no_mixer_meanpool", "transformer_param_matched")],
    "klsweep": [("full", kl, DEFAULT_CHUNK) for kl in (0.0, 1e-3, 1e-2)],
    "chunks": [("full", DEFAULT_KL, c) for c in (8, 32)],
}


def run_name(variant, kl, chunk, seed):
    return f"{variant}__kl{kl:g}__chunk{chunk}__seed{seed}"


def is_current(run_dir, epochs):
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return False, "no summary"
    s = read_json(summary_path)
    if s.get("status") != "COMPLETED":
        return False, s.get("status")
    if s.get("code_sha256") != code_sha256():
        return False, "code changed"
    if s.get("feature_manifest_sha256") != sha256_file(FEATURE_DIR / "feature_manifest.json"):
        return False, "features changed"
    cfg = read_json(run_dir / "config.json")
    if cfg.get("epochs") != epochs:
        return False, "epoch count differs"
    results = read_json(run_dir / "test_results.json")
    if results["provenance"]["config_sha256"] != s.get("config_sha256"):
        return False, "config hash mismatch"
    if sha256_file(run_dir / "test_results.json") != s.get("test_results_sha256"):
        return False, "test_results.json modified after run"
    return True, "current"


def collect(root):
    rows = []
    for summary_path in sorted(root.glob("*/run_summary.json")):
        run_dir = summary_path.parent
        s = read_json(summary_path)
        row = {"run": run_dir.name, "variant": s.get("variant"), "seed": s.get("seed"),
               "kl_weight": s.get("kl_weight"), "hvsc_chunk_size": s.get("hvsc_chunk_size"),
               "status": s.get("status"), "trainable_params": s.get("trainable_params")}
        results_path = run_dir / "test_results.json"
        if s.get("status") == "COMPLETED" and results_path.is_file():
            r = read_json(results_path)
            for k in ("i2t_r1", "i2t_r5", "i2t_r10", "i2t_medr", "t2i_r1", "t2i_r5", "t2i_r10", "t2i_medr",
                      "mean_recall", "dual_mean_recall", "reranked", "best_epoch", "best_val_mean_recall",
                      "train_seconds", "peak_train_memory_mb"):
                row[k] = r.get(k)
            row["code_sha256"] = r["provenance"]["code_sha256"]
            row["config_sha256"] = r["provenance"]["config_sha256"]
            row["current_code"] = row["code_sha256"] == code_sha256()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", required=True, choices=sorted(SUITES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--stop-on-failure", action="store_true")
    args = ap.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    plan = [(v, kl, c, s) for suite in args.suite for (v, kl, c) in SUITES[suite] for s in args.seeds]
    assert all(v in VARIANTS for v, _, _, _ in plan)
    banner(f"RUN MATRIX: {len(plan)} runs, code {code_sha256()[:12]}")

    failures = []
    for i, (variant, kl, chunk, seed) in enumerate(plan, 1):
        run_dir = RUNS_DIR / run_name(variant, kl, chunk, seed)
        current, reason = is_current(run_dir, args.epochs)
        if current:
            print(f"[{i}/{len(plan)}] skip {run_dir.name} (verified current)")
            continue
        print(f"[{i}/{len(plan)}] run  {run_dir.name} ({reason})", flush=True)
        cmd = [sys.executable, "-u", str(PACKAGE_DIR / "train.py"), "--variant", variant, "--seed", str(seed),
               "--epochs", str(args.epochs), "--kl-weight", str(kl), "--hvsc-chunk-size", str(chunk),
               "--out", str(run_dir), "--overwrite"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((run_dir.name, rc))
            print(f"FAILED {run_dir.name} exit={rc}", flush=True)
            if args.stop_on_failure:
                break

    df = collect(RUNS_DIR)
    out_csv = RUNS_DIR.parent / "results" / "all_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    banner(f"{len(df)} run directories, {int((df['status'] == 'COMPLETED').sum()) if len(df) else 0} completed "
           f"-> {out_csv}")
    if failures:
        print("failures:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
