"""Download the S3 .ntds corpus to a flat local directory for NtdsChunkDataset.

NtdsChunkDataset/build_ntds_datasets expect NTDS_LOCAL_DIR/<basename(ntds_uri)>
-- a flat directory, not the partitioned interim/ntds/dataset_version=v1/...
layout the files actually live under in S3. `aws s3 sync` on that prefix would
preserve the partitioning and silently produce zero matching files, so this
does per-row downloads to the flattened path instead. Idempotent: rows whose
local file already exists at the expected size are skipped, so a partial run
(interrupted spot instance, etc.) can just be re-run.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    bucket, key = uri.replace("s3://", "", 1).split("/", 1)
    return bucket, key


def _download_row(s3, row: dict, local_dir: str) -> str:
    bucket, key = _parse_s3_uri(row["ntds_uri"])
    dest = os.path.join(local_dir, os.path.basename(key))
    if os.path.exists(dest) and os.path.getsize(dest) == row["n_bytes"]:
        return "skipped"
    s3.download_file(bucket, key, dest)
    return "downloaded"


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=os.getenv("NTDS_MANIFEST", "data/manifests/clips_v1.parquet"))
    p.add_argument("--local-dir", default=os.getenv("NTDS_LOCAL_DIR", "data/ntds"))
    p.add_argument("--max-workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="Only sync the first N rows (smoke test)")
    args = p.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    manifest = pd.read_parquet(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    rows = manifest.to_dict(orient="records")

    s3 = boto3.client("s3", config=Config(max_pool_connections=max(20, args.max_workers * 2)))
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_download_row, s3, row, args.local_dir): row for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"FAILED {row['ntds_uri']}: {exc}")
            if i % 2000 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} processed -- {counts}")

    print(f"done: {counts}")
    if counts["failed"]:
        raise SystemExit(f"{counts['failed']} file(s) failed to download -- re-run to retry (idempotent)")


if __name__ == "__main__":
    main()
