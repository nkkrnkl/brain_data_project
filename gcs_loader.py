from __future__ import annotations

import gzip
from functools import lru_cache

import gcsfs
import nibabel as nib

BUCKET = "results_050626"
PROJECT = "brain-data-project-050626"


@lru_cache(maxsize=1)
def fs() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(token="google_default")


def list_files(prefix: str = "", suffix: str | None = None) -> list[str]:
    paths = fs().ls(f"{BUCKET}/{prefix}", detail=False)
    if suffix:
        paths = [p for p in paths if p.endswith(suffix)]
    return paths


def open_nifti(path: str) -> nib.Nifti1Image:
    """Open a NIfTI file from GCS by streaming its bytes into memory.

    Handles .nii.gz by decompressing the streamed bytes before parsing —
    nib.Nifti1Image.from_bytes does not accept gzipped input.
    """
    full = path if path.startswith(BUCKET) else f"{BUCKET}/{path.lstrip('/')}"
    with fs().open(full, "rb") as f:
        raw = f.read()
    if full.endswith(".gz"):
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def load_nifti(path: str):
    """Open and immediately load the data array."""
    return open_nifti(path).get_fdata()


def gcs_url(path: str) -> str:
    """Return a gs:// URL for tools that accept fsspec URLs (e.g., nilearn-with-fsspec)."""
    full = path if path.startswith(BUCKET) else f"{BUCKET}/{path.lstrip('/')}"
    return f"gs://{full}"


if __name__ == "__main__":
    files = list_files()
    print(f"{len(files)} objects in gs://{BUCKET}/")
    for p in files[:10]:
        print(" ", p)
