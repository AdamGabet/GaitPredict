"""Migrate `.ntds` pose files from the WIS p10k Google Shared Drive to S3.

Streams each file Google Drive -> memory -> S3 (no local disk), and emits a
Parquet manifest that maps every clip to its participant, activity, front-camera
flag, demographics, and final S3 URI. The manifest — not the object layout — is
the query/split layer for training.

Design notes
------------
* There is exactly ONE Google Shared Drive in play. The service account is not
  a member of it (drives().list() returns empty for it) — it only has specific
  folders shared directly to its email (`sharedWithMe`), all living inside that
  one drive: `WIS_P10K_Data` (raw pose data), `SessionData` (mapping JSONs), and
  `10K_export_metadata` (participant metadata — already scraped into the local
  `meta_table.csv` by a separate step, per Notebooks/Scrape_all_reports.ipynb).
* Because the SA lacks drive membership, unscoped drive-wide search queries
  (`corpora="drive"` + free-text `q`) are not reliable for it. Every traversal
  here is parent-scoped (`'<folder_id>' in parents`), starting from a folder the
  SA actually holds — the one pattern confirmed to work for this SA.
* The actual `.ntds` files live under `WIS_P10K_Data/k4a/`, partitioned as
  `yyyy/mm/dd/<test_id>/...`. Mapping JSONs live under `SessionData/` with the
  same `yyyy/mm/dd/<test_id>/` partitioning. The date/test_id folder names are
  never parsed directly — both trees are discovered by walking. At this tree's
  scale (tens of thousands of leaf folders) a naive one-call-at-a-time recursive
  walk is unusably slow (69 min covered 80k files without finishing); both walks
  use `parallel_walk_files` (bounded thread pool, one Drive service per worker
  thread — see --max-workers).
* `.ntds` filenames do NOT encode activity. Activity / front-camera / test_id come
  only from the SessionData mapping JSONs (the `videos_dict_list` schema). We join
  a mapping-JSON clip to a Drive file on the `.ntds` basename
  (`K4A<cam_serial>_HPECONFIG_<timestamp>.ntds`), which is unique within the drive.
* Demographics come from the already-scraped meta_table.csv (test_id -> participant
  + year_of_birth/gender/height/weight/dominant_hand). We store age_at_session, not
  raw year_of_birth (PII discipline, per CLAUDE.md).
* There is no longer a physical "source drive" batch axis (only one drive exists).
  We record the recording date extracted from the `.ntds` filename timestamp
  (day granularity) as `recording_date` — a plain fact, and the top S3 partition
  for object organization — but it is DELIBERATELY NOT treated as a validated
  batch-effect label anywhere in this script. Two reasons:
    1. Unconfirmed: whether date tracks real room/equipment/staff changes has
       not been verified against the actual recording schedule.
    2. Confounded with clinical signal: for a longitudinal trial (e.g. Solid:
       4 subjects x 4 time points), a participant's recording date also encodes
       which protocol visit this is (baseline / 3-month / 6-month / ...). Date
       and visit-number are the same axis for a given participant. Any future
       batch-effect correction that naively regresses out or suppresses
       variance by date risks erasing the longitudinal treatment-effect signal
       the trial exists to measure, not just correcting a nuisance confound.
       Disentangling "same room, different participant" from "same participant,
       different visit" requires joining against the trial protocol's visit
       schedule — this manifest alone does not carry visit-number and must not
       be used for automatic batch correction without that join.
  Given Romberg (open/closed) + sit-to-stand are ~41% of this dataset and are
  the tests CLAUDE.md flags as most batch-effect-sensitive, this caveat matters
  disproportionately here — flag for whoever picks up PO 008 batch-effect work.
* .mkv is intentionally skipped; only `.ntds` is migrated.

Idempotent & resumable: an object already present in S3 (matching size) is skipped.
Use --limit for a smoke test before a full run.

Usage
-----
    # Config + credentials are read from a local .env file (default: ./.env), e.g.
    #   SERVICE_ACCOUNT_FILE=biopilot-458819-531753a47910.json
    #   META_TABLE=data/meta_table.csv
    #   S3_BUCKET=biopilot-movement
    #   AWS_ACCESS_KEY_ID=...
    #   AWS_SECRET_ACCESS_KEY=...
    #   AWS_DEFAULT_REGION=us-east-1
    python migrate_ntds_to_s3.py \
        --limit 20                         # smoke test: first 20 clips only
        --dry-run                          # plan + manifest, no uploads
    # Any .env value can still be overridden with the matching CLI flag.

Assumptions worth verifying on first run (all logged, none fatal):
    * The SA has folders named `WIS_P10K_Data` and `SessionData` shared to it
      directly (override via --wis-folder / --session-folder).
    * `WIS_P10K_Data` has a child folder named `k4a` holding the pose tree
      (override via --k4a-subfolder).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import pandas as pd

logger = logging.getLogger("migrate_ntds")

# --- Google Drive -----------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
_PAGE_FIELDS = "nextPageToken, files(id, name, mimeType, size, parents)"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _execute_with_retry(request, max_retries: int = 6, base_delay: float = 1.0):
    """Run a googleapiclient request with exponential-backoff retries.

    The Drive API returns occasional transient 5xx/429s under sustained paginated
    traffic (observed in practice on the k4a/ walk, which issues thousands of
    parent-scoped list calls). Google's own guidance is to retry these with
    backoff rather than treat them as fatal.
    """
    from googleapiclient.errors import HttpError

    for attempt in range(max_retries + 1):
        try:
            return request.execute()
        except HttpError as exc:
            status = exc.resp.status if exc.resp is not None else None
            if status not in _RETRYABLE_STATUS or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("  Drive API %s, retrying in %.1fs (attempt %d/%d)",
                           status, delay, attempt + 1, max_retries)
            time.sleep(delay)


def build_drive_service(service_account_file: str):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    # cache_discovery=False silences a noisy warning under newer libs.
    return build("drive", "v3", credentials=creds, cache_discovery=False)


_thread_local = threading.local()


def _thread_drive_service(service_account_file: str):
    """One Drive service (and its own http transport) per worker thread.

    googleapiclient Resource objects wrap an httplib2 transport that isn't safe
    to share across threads; building one per thread avoids relying on any
    undocumented thread-safety guarantee.
    """
    svc = getattr(_thread_local, "service", None)
    if svc is None:
        svc = build_drive_service(service_account_file)
        _thread_local.service = svc
    return svc


def resolve_shared_folder(service, name: str) -> Optional[dict]:
    """Find a folder shared directly with the SA by name (sharedWithMe).

    The SA is not a drive member; it has specific folders shared to its email
    (e.g. WIS_P10K_Data, SessionData), all inside one drive. Returns the folder
    dict {id, name, driveId} or None.
    """
    resp = _execute_with_retry(service.files().list(
        q=(f"name = '{name}' and mimeType = '{FOLDER_MIME}' "
           f"and sharedWithMe = true and trashed = false"),
        corpora="allDrives", includeItemsFromAllDrives=True, supportsAllDrives=True,
        fields="files(id, name, driveId)", pageSize=10,
    ))
    files = resp.get("files", [])
    return files[0] if files else None


def _list_children(service, folder_id: str, drive_id: str) -> Iterator[dict]:
    page_token = None
    while True:
        resp = _execute_with_retry(service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            corpora="drive",
            driveId=drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields=_PAGE_FIELDS,
            pageSize=1000,
            pageToken=page_token,
        ))
        yield from resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def find_child_folder(service, parent_id: str, drive_id: str, name: str) -> Optional[str]:
    for f in _list_children(service, parent_id, drive_id):
        if f["mimeType"] == FOLDER_MIME and f["name"] == name:
            return f["id"]
    return None


def parallel_walk_files(
    service_account_file: str, drive_id: str, root_folder_id: str,
    max_workers: int = 16, heartbeat: int = 1000, label: str = "",
) -> Iterator[dict]:
    """Breadth-first walk of a Drive folder tree, listing sibling folders concurrently.

    A naive one-call-at-a-time recursive walk took 69 minutes to cover 80k files
    without finishing the yyyy/mm/dd/<test_id> tree under k4a/ — unusable at this
    scale, since the cost is dominated by the number of leaf folders, not files.
    This expands one BFS level at a time, farming the listing call for every
    folder in that level out to a thread pool. Each worker gets its own Drive
    service (see _thread_drive_service) since a single service's http transport
    isn't safe to share across threads.
    """
    frontier = [root_folder_id]
    seen = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while frontier:
            def _list(fid: str) -> list[dict]:
                svc = _thread_drive_service(service_account_file)
                return list(_list_children(svc, fid, drive_id))

            next_frontier: list[str] = []
            for children in pool.map(_list, frontier):
                for f in children:
                    if f["mimeType"] == FOLDER_MIME:
                        next_frontier.append(f["id"])
                    else:
                        seen += 1
                        if heartbeat and seen % heartbeat == 0:
                            logger.info("  ...walked %d files under %s (%d parallel workers)",
                                       seen, label, max_workers)
                        yield f
            frontier = next_frontier


def resolve_roots(
    service, wis_folder: str, session_folder: str, k4a_subfolder: str,
) -> tuple[str, str, str]:
    """Resolve the drive id + the two folder roots we actually need to walk.

    Both `wis_folder` and `session_folder` must be shared directly with the SA
    (sharedWithMe) — parent-scoped listing from there works even though the SA
    is not a drive member. Returns (drive_id, k4a_root_id, session_root_id).
    """
    wis = resolve_shared_folder(service, wis_folder)
    if wis is None:
        raise RuntimeError(f"folder '{wis_folder}' not shared with this service account")
    session = resolve_shared_folder(service, session_folder)
    if session is None:
        raise RuntimeError(f"folder '{session_folder}' not shared with this service account")

    drive_id = wis["driveId"]
    k4a_root = find_child_folder(service, wis["id"], drive_id, k4a_subfolder)
    if k4a_root is None:
        raise RuntimeError(f"no '{k4a_subfolder}' child folder under '{wis_folder}'")

    return drive_id, k4a_root, session["id"]


def walk_ntds_files(
    service_account_file: str, k4a_root: str, drive_id: str, max_workers: int = 16,
) -> Iterator[dict]:
    """Walk the k4a/yyyy/mm/dd/<test_id>/ tree in parallel, yielding `.ntds` files.

    Parent-scoped (not a free-text drive search) — the only traversal pattern
    confirmed to work for a SA that only has specific folders shared to it.
    """
    for f in parallel_walk_files(service_account_file, drive_id, k4a_root,
                                  max_workers=max_workers, heartbeat=1000, label="k4a/"):
        if f["name"].lower().endswith(".ntds"):
            yield f


def download_bytes(service, file_id: str, max_retries: int = 6) -> io.BytesIO:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request, chunksize=16 * 1024 * 1024)
    done = False
    attempt = 0
    while not done:
        try:
            _, done = downloader.next_chunk()
        except HttpError as exc:
            status = exc.resp.status if exc.resp is not None else None
            if status not in _RETRYABLE_STATUS or attempt >= max_retries:
                raise
            delay = 1.0 * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("  Drive download %s on %s, retrying in %.1fs", status, file_id, delay)
            time.sleep(delay)
            attempt += 1
    fh.seek(0)
    return fh


# --- Parsing ----------------------------------------------------------------
_TS_RX = re.compile(r"HPECONFIG_(\d{4}-\d{2}-\d{2})T", re.IGNORECASE)


@dataclass
class Clip:
    test_id: str
    activity: str
    speed: float
    camera_serial: str
    ntds_basename: str
    is_front_facing: bool
    session_date: Optional[str] = None
    drive_file_id: Optional[str] = None
    n_bytes: Optional[int] = None


def parse_session_json(raw: bytes, drive_name: str) -> list[Clip]:
    """Parse a SessionData mapping JSON (`videos_dict_list` schema) into Clip rows."""
    doc = json.loads(raw)
    test_id = doc.get("test_id") or ""
    # The demo file prefixes ids with "test_id_"; strip any such wrapper.
    test_id = test_id.replace("test_id_", "")
    clips: list[Clip] = []
    for v in doc.get("videos_dict_list", []):
        ntds_path = v.get("ntds_path")
        if not ntds_path:
            continue
        basename = os.path.basename(ntds_path.replace("\\", "/"))
        cam = _extract_camera_serial(basename)
        sess_date = _extract_date(basename)
        clips.append(
            Clip(
                test_id=test_id,
                activity=v.get("activity_type", "unknown"),
                speed=float(v.get("speed", 0) or 0),
                camera_serial=cam,
                ntds_basename=basename,
                is_front_facing=bool(v.get("front_facing_camera", 0)),
                session_date=sess_date,
            )
        )
    return clips


def _extract_camera_serial(basename: str) -> str:
    m = re.match(r"K4A0*(\d+)_", basename)
    return m.group(1) if m else ""


def _extract_date(basename: str) -> Optional[str]:
    m = _TS_RX.search(basename)
    return m.group(1) if m else None


# --- Manifest / demographics ------------------------------------------------
def load_demographics(meta_table_csv: str) -> pd.DataFrame:
    df = pd.read_csv(meta_table_csv)
    keep = [
        "test_id", "participant_id", "year_of_birth",
        "gender", "height", "weight", "dominant_hand",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df.drop_duplicates(subset="test_id")


def _age_at_session(year_of_birth, session_date: Optional[str]) -> Optional[int]:
    if pd.isna(year_of_birth) or not session_date:
        return None
    try:
        return int(session_date[:4]) - int(year_of_birth)
    except (ValueError, TypeError):
        return None


def s3_key_for(session_date: Optional[str], participant_id: str, activity: str,
               ntds_basename: str, dataset_version: str) -> str:
    # `recording_date` is a plain descriptive fact used for object organization —
    # NOT an asserted batch-effect label. See module docstring: date is confounded
    # with longitudinal visit-number per participant, so naming this "batch" or
    # "source" here would overclaim something nobody has verified.
    date_label = session_date or "unknown-date"
    pid = participant_id or "unknown"
    return (
        f"interim/ntds/dataset_version={dataset_version}/"
        f"recording_date={date_label}/participant_id={pid}/"
        f"activity={activity}/{ntds_basename}"
    )


# --- S3 ---------------------------------------------------------------------
def object_exists(s3, bucket: str, key: str, expected_size: Optional[int]) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    if expected_size is None:
        return True
    return head.get("ContentLength") == expected_size


# --- Migration driver ---------------------------------------------------------
@dataclass
class MigrationResult:
    rows: list[dict] = field(default_factory=list)
    uploaded: int = 0
    skipped: int = 0
    missing_ntds: int = 0
    unmatched_meta: int = 0


def run_migration(
    service_account_file: str, s3, drive_id: str, k4a_root: str, session_root: str,
    demographics: pd.DataFrame, bucket: str, dataset_version: str,
    dry_run: bool, limit: Optional[int], max_workers: int = 16,
) -> MigrationResult:
    result = MigrationResult()

    # 1. Index every .ntds file under k4a/: basename -> (file_id, size).
    ntds_index: dict[str, dict] = {}
    for f in walk_ntds_files(service_account_file, k4a_root, drive_id, max_workers=max_workers):
        ntds_index[f["name"]] = {"id": f["id"], "size": int(f.get("size", 0) or 0)}
    logger.info("indexed %d .ntds files", len(ntds_index))

    # 2. Discover SessionData mapping JSONs, then download+parse them concurrently.
    json_files = [
        f for f in parallel_walk_files(service_account_file, drive_id, session_root,
                                        max_workers=max_workers, heartbeat=500, label="SessionData/")
        if f["name"].lower().endswith(".json")
    ]
    logger.info("found %d SessionData JSONs", len(json_files))

    def _fetch_and_parse(jf: dict) -> list[Clip]:
        svc = _thread_drive_service(service_account_file)
        try:
            raw = download_bytes(svc, jf["id"]).read()
            return parse_session_json(raw, jf["name"])
        except Exception as exc:  # noqa: BLE001 - log and continue, never abort the run
            logger.warning("failed to parse %s: %s", jf["name"], exc)
            return []

    clips: list[Clip] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, parsed in enumerate(pool.map(_fetch_and_parse, json_files), 1):
            clips.extend(parsed)
            if i % 200 == 0:
                logger.info("...parsed %d/%d SessionData JSONs (%d clips so far)",
                           i, len(json_files), len(clips))
    logger.info("parsed %d clips from SessionData", len(clips))

    demo_by_test = demographics.set_index("test_id").to_dict("index")
    if limit is not None:
        clips = clips[:limit]

    # 3. Resolve each clip to a Drive file_id + demographics, download+upload, build its row.
    # Parallelized like the walks above: at ~57k clips / ~129GB, one-at-a-time
    # download+upload would take many hours. Each worker gets its own Drive
    # service; the boto3 S3 client is safe to share across threads.
    def _process_clip(clip: Clip) -> dict:
        entry = ntds_index.get(clip.ntds_basename)
        if entry is None:
            return {"status": "missing_ntds"}
        clip.drive_file_id, clip.n_bytes = entry["id"], entry["size"]

        demo = demo_by_test.get(clip.test_id)
        participant_id = (demo or {}).get("participant_id", "")
        key = s3_key_for(clip.session_date, participant_id, clip.activity,
                         clip.ntds_basename, dataset_version)

        if dry_run:
            status = "skipped"
        elif object_exists(s3, bucket, key, clip.n_bytes):
            status = "skipped"
        else:
            svc = _thread_drive_service(service_account_file)
            fh = download_bytes(svc, clip.drive_file_id)
            s3.upload_fileobj(fh, bucket, key)
            status = "uploaded"

        row = {
            "session_date": clip.session_date,
            "participant_id": participant_id,
            "test_id": clip.test_id,
            "activity": clip.activity,
            "camera_serial": clip.camera_serial,
            "is_front_facing": clip.is_front_facing,
            "speed": clip.speed,
            "age_at_session": _age_at_session((demo or {}).get("year_of_birth"), clip.session_date),
            "gender": (demo or {}).get("gender"),
            "height_cm": (demo or {}).get("height"),
            "weight_kg": (demo or {}).get("weight"),
            "dominant_hand": (demo or {}).get("dominant_hand"),
            "n_bytes": clip.n_bytes,
            "pipeline_version": "ntds_fix_001_filtering",
            "source": "azure_kinect",
            "ntds_uri": f"s3://{bucket}/{key}",
        }
        return {"status": status, "unmatched_meta": demo is None, "row": row}

    n = len(clips)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, outcome in enumerate(pool.map(_process_clip, clips), 1):
            if outcome["status"] == "missing_ntds":
                result.missing_ntds += 1
                continue
            if outcome["status"] == "uploaded":
                result.uploaded += 1
            else:
                result.skipped += 1
            if outcome["unmatched_meta"]:
                result.unmatched_meta += 1
            result.rows.append(outcome["row"])
            if i % 1000 == 0:
                logger.info("...processed %d/%d clips (uploaded=%d skipped=%d missing_ntds=%d)",
                           i, n, result.uploaded, result.skipped, result.missing_ntds)

    logger.info(
        "migration done: uploaded=%d skipped=%d missing_ntds=%d unmatched_meta=%d",
        result.uploaded, result.skipped, result.missing_ntds, result.unmatched_meta,
    )
    return result


# --- Entry point ------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--service-account", default=None,
                   help="Google service account JSON (falls back to SERVICE_ACCOUNT_FILE in .env)")
    p.add_argument("--meta-table", default=None,
                   help="Consolidated meta_table CSV (falls back to META_TABLE in .env)")
    p.add_argument("--s3-bucket", default=None,
                   help="Target S3 bucket (falls back to S3_BUCKET in .env)")
    p.add_argument("--dataset-version", default="v1")
    p.add_argument("--wis-folder", default="WIS_P10K_Data", help="Shared folder holding the pose tree")
    p.add_argument("--k4a-subfolder", default="k4a", help="Child of --wis-folder holding yyyy/mm/dd/<test_id>/*.ntds")
    p.add_argument("--session-folder", default="SessionData", help="Shared folder holding mapping JSONs")
    p.add_argument("--manifest-dir", default="data/manifests", help="Local dir for the Parquet manifest")
    p.add_argument("--env-file", default=".env", help="dotenv file with AWS_* creds for boto3")
    p.add_argument("--limit", type=int, default=None, help="Cap total clips processed (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="Plan + manifest only, no S3 writes")
    p.add_argument("--max-workers", type=int, default=16, help="Concurrent Drive API workers for tree walks")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Load AWS_* creds from the .env file into the environment for boto3 to pick up.
    from dotenv import load_dotenv
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        logger.info("loaded env from %s", args.env_file)
    else:
        logger.warning("env file %s not found; relying on ambient creds", args.env_file)

    service_account_file = args.service_account or os.getenv("SERVICE_ACCOUNT_FILE")
    if not service_account_file:
        p.error("no service account: pass --service-account or set SERVICE_ACCOUNT_FILE in .env")
    if not os.path.exists(service_account_file):
        p.error(f"service account file not found: {service_account_file}")

    meta_table = args.meta_table or os.getenv("META_TABLE")
    if not meta_table:
        p.error("no meta table: pass --meta-table or set META_TABLE in .env")
    if not os.path.exists(meta_table):
        p.error(f"meta table not found: {meta_table}")

    s3_bucket = args.s3_bucket or os.getenv("S3_BUCKET")
    if not s3_bucket:
        p.error("no S3 bucket: pass --s3-bucket or set S3_BUCKET in .env")

    service = build_drive_service(service_account_file)
    demographics = load_demographics(meta_table)
    logger.info("loaded demographics for %d test sessions", len(demographics))

    import boto3
    from botocore.config import Config
    # Default pool (10) is smaller than --max-workers concurrent uploads, which was
    # forcing boto3 to tear down and reopen connections constantly (observed as
    # "Connection pool is full, discarding connection" spam during a real run).
    s3 = boto3.client("s3", config=Config(max_pool_connections=max(20, args.max_workers * 2)))

    drive_id, k4a_root, session_root = resolve_roots(
        service, args.wis_folder, args.session_folder, args.k4a_subfolder,
    )
    logger.info("resolved drive=%s k4a_root=%s session_root=%s", drive_id, k4a_root, session_root)

    os.makedirs(args.manifest_dir, exist_ok=True)
    result = run_migration(
        service_account_file, s3, drive_id, k4a_root, session_root, demographics,
        s3_bucket, args.dataset_version, args.dry_run, args.limit, args.max_workers,
    )

    if result.rows:
        manifest = pd.DataFrame(result.rows)
        out = os.path.join(args.manifest_dir, f"clips_{args.dataset_version}.parquet")
        manifest.to_parquet(out, index=False)
        logger.info("wrote manifest: %s (%d clips)", out, len(manifest))
        if not args.dry_run:
            key = f"manifests/clips/dataset_version={args.dataset_version}/clips.parquet"
            s3.upload_file(out, s3_bucket, key)
            logger.info("uploaded manifest to s3://%s/%s", s3_bucket, key)


if __name__ == "__main__":
    main()
