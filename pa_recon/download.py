"""Checksum-verified access to the public development subset."""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

SUBSET_URL = "https://zenodo.org/records/21785486/files/example_subset_100_samples.h5?download=1"
SUBSET_MD5 = "f9cab78d59b66eba53cc8a80bf6cc6a5"


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_subset(output: str | Path) -> Path:
    path = Path(output)
    if path.exists():
        if file_digest(path, "md5") != SUBSET_MD5:
            raise ValueError(f"Existing file checksum differs from the published subset: {path}")
        print(f"Verified existing subset: {path}", flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Partial download exists; inspect or remove it before retrying: {temporary}")
    stream = temporary.open("xb")
    try:
        request = urllib.request.Request(SUBSET_URL, headers={"User-Agent": "pa-recon-lab/0.1.0"})
        with stream, urllib.request.urlopen(request, timeout=120) as response:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if file_digest(temporary, "md5") != SUBSET_MD5:
            raise ValueError("Downloaded file does not match the dataset's published MD5")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified: {path}", flush=True)
    return path
