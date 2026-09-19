"""Shared provenance, hashing, seeding and IO helpers.

Every artifact written by this package carries the same provenance block
(`provenance()`), so a result can always be traced to the exact code,
configuration, dataset and cached features that produced it.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent

# Work directory for data, feature cache and runs. Never inside the git checkout.
WORK_DIR = Path(os.environ.get("HEDO_WORK", "/content/hedo_work"))
DATA_DIR = WORK_DIR / "data" / "flickr8k"
FEATURE_DIR = WORK_DIR / "features"
RUNS_DIR = WORK_DIR / "runs"
REPORT_DIR = WORK_DIR / "report"

TRACKED_PACKAGES = [
    "torch", "torchvision", "mamba_ssm", "causal_conv1d", "triton", "transformers",
    "tokenizers", "huggingface_hub", "safetensors", "numpy", "pandas", "scipy",
    "matplotlib", "Pillow", "einops",
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


FEATURE_CODE_FILES = ("common.py", "data.py", "backbones.py", "features.py")


def feature_code_sha256():
    """Hash of the code that determines cached features. A mismatch means the cache is stale."""
    h = hashlib.sha256()
    for name in FEATURE_CODE_FILES:
        h.update(name.encode("utf-8"))
        h.update((PACKAGE_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def code_sha256():
    """Hash of every .py file in the package (name + bytes, sorted). Tests excluded."""
    h = hashlib.sha256()
    for p in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE_DIR, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PACKAGE_DIR)], cwd=PACKAGE_DIR,
                               capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return {"commit": None, "dirty": None}
        return {"commit": out.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def package_versions():
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest():
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        info["nvidia_smi"] = None
    try:
        info["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                            text=True, timeout=120).stdout.splitlines()
    except Exception:
        info["pip_freeze"] = None
    return info


def provenance(config=None, extra=None, include_artifacts=True):
    block = {
        "code_sha256": code_sha256(),
        "git": git_commit(),
        "packages": package_versions(),
        "python": sys.version.split()[0],
    }
    if config is not None:
        block["config"] = config
        block["config_sha256"] = sha256_json(config)
    manifest = DATA_DIR / "manifest.json"
    if include_artifacts and manifest.is_file():
        block["dataset_manifest_sha256"] = sha256_file(manifest)
    fmanifest = FEATURE_DIR / "feature_manifest.json"
    if include_artifacts and fmanifest.is_file():
        block["feature_manifest_sha256"] = sha256_file(fmanifest)
    if extra:
        block.update(extra)
    return block


def set_seed(seed, deterministic=True):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        # warn_only: Mamba-2 Triton backward kernels have no deterministic variant.
        # Training is therefore NOT bit-reproducible; evaluation is (checked separately).
        torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def banner(text):
    print("=" * 80, flush=True)
    print(text, flush=True)
    print("=" * 80, flush=True)
