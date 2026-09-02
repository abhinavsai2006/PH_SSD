"""
Project Bundle Exporter for Moving PH-SSD Codebase to Remote GPU Systems.
Author: Lead MLOps Engineer
License: Apache 2.0
"""

import os
import zipfile


def package_repository(output_zip: str = "ph_ssd_project_bundle.zip") -> str:
    """
    Bundle clean repository source code, scripts, configs, and notebooks into a single zip archive.

    Args:
        output_zip (str): Path to destination zip archive.

    Returns:
        str: Absolute path to created zip file.
    """
    included_dirs = ["ph_ssd", "scripts", "configs", "docs", "tables", "figures"]
    included_files = [
        "PH_SSD_Complete_Pipeline.ipynb",
        "train.py",
        "evaluate.py",
        "benchmark.py",
        "infer.py",
        "export_torchscript.py",
        "export_onnx.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        "LICENSE",
        "FINAL_AUDIT.md",
    ]

    print(f"Creating portable project zip archive: '{output_zip}'...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in included_files:
            if os.path.exists(f):
                zipf.write(f, arcname=f)
                print(f" + Added file: {f}")

        for d in included_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        if "__pycache__" in root or file.endswith((".pyc", ".pyo")):
                            continue
                        fp = os.path.join(root, file)
                        arcname = os.path.relpath(fp, start=".")
                        zipf.write(fp, arcname=arcname)
                        print(f" + Added module: {arcname}")

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n[SUCCESS] Project bundle ready: '{os.path.abspath(output_zip)}' ({size_mb:.2f} MB)")
    return os.path.abspath(output_zip)


if __name__ == "__main__":
    package_repository()
