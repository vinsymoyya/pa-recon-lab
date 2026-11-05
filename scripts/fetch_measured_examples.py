"""Fetch matched phantom examples by ZIP byte ranges, without downloading 525 MB."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

ARCHIVES = {"invision": (238064334, "a3ab81d0256feb2ffa8412b3ed40efa6"),
            "tropus": (286740645, "e826db69f96eeeadb7e18a459114b9e9")}


def read_range(url, start, stop, size):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop-1}",
        "User-Agent": "pa-recon-lab/research", "Accept-Encoding": "identity"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{size}":
                    raise ValueError("Source did not honor requested archive byte range")
                data = response.read(stop-start+1)
            if len(data) != stop-start:
                raise ValueError(f"Incomplete or oversized range: {len(data)} != {stop-start}")
            return data
        except (OSError, ValueError):
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phantom", default="P.5.6.2")
    parser.add_argument("--wavelength", type=int, default=750)
    parser.add_argument("--output", type=Path, default=Path("data/measured/openqpai/examples"))
    args = parser.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in args.phantom) or not args.phantom:
        raise ValueError("Unsafe phantom ID")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "ATTRIBUTION.md").write_text(
        "# Measured source data attribution\n\nGröhl et al., OpenQPAIData, version v1. "
        "Source: https://doi.org/10.5281/zenodo.14044853. "
        "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. "
        "Selected raw recordings, supplied reconstructions and labels are unmodified. "
        "The curator is not the original dataset creator. See manifest.json for selection, ZIP CRC32, "
        "member SHA256 and the unverified whole-archive MD5 boundary.\n", encoding="utf-8")
    manifest = {"source": "https://doi.org/10.5281/zenodo.14044853", "license": "CC-BY-4.0",
        "credit": "Grohl et al., OpenQPAIData", "phantom": args.phantom, "wavelength_nm": args.wavelength,
        "selection": "Fixed example ID used in the source loading tutorial; no selection by performance",
        "full_archive_checksum_verified": False,
        "integrity": "HTTPS source, exact Content-Range, ZIP central-directory CRC32, SHA256 of fetched members; whole-archive MD5 not checked",
        "archives": {}, "members": {}}
    for device, (size, md5) in ARCHIVES.items():
        url = f"https://zenodo.org/api/records/14044853/files/{device}.zip/content"
        tail_start = size - 65536
        tail = read_range(url, tail_start, size, size)
        # ZipFile can read this complete central directory with a negative concat
        # offset. Restore its original archive offsets before fetching members.
        with zipfile.ZipFile(io.BytesIO(tail)) as archive:
            infos = archive.infolist()
        expected = [f"{device}/raw/{args.phantom}_{args.wavelength}_ipasc.hdf5",
                    f"{device}/recon/{args.phantom}_{args.wavelength}.npy",
                    f"{device}/labels/{args.phantom}.npy"]
        by_name = {info.filename: info for info in infos}
        if not all(name in by_name for name in expected):
            raise ValueError(f"Missing requested matched example in {device}")
        manifest["archives"][device] = {"bytes": size, "published_md5_unverified": md5,
            "member_count": len(infos), "central_directory_tail_sha256": hashlib.sha256(tail).hexdigest()}
        for name in expected:
            info = by_name[name]
            offset = info.header_offset + tail_start
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Unsafe archive member path")
            if target.exists():
                payload = target.read_bytes()
            else:
                stop = min(size, offset + 30 + len(name.encode()) + 65535 + info.compress_size)
                raw = read_range(url, offset, stop, size)
                header = struct.unpack_from("<4s5H3L2H", raw)
                if header[0] != b"PK\x03\x04" or header[3] != info.compress_type:
                    raise ValueError("Local ZIP header disagrees with central directory")
                start = 30 + header[-2] + header[-1]
                compressed = raw[start:start + info.compress_size]
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    payload = zlib.decompress(compressed, -15)
                elif info.compress_type == zipfile.ZIP_STORED:
                    payload = compressed
                else:
                    raise ValueError("Unsupported compression")
            if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                raise ValueError(f"ZIP member integrity failure: {name}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            manifest["members"][name] = {"bytes": len(payload), "crc32": f"{info.CRC:08x}", "sha256": hashlib.sha256(payload).hexdigest()}
            print(f"Verified {name}: {len(payload)} bytes", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
