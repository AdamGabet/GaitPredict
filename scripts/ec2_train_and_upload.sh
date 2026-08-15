#!/usr/bin/env bash
# End-to-end training run on a fresh EC2 instance: install, sync the S3 ntds
# corpus, train, and stream checkpoints to S3 as they're written (so a spot
# interruption doesn't lose the run). See model/training/README.md for the
# background on data sources / configs this wraps.
#
# Usage: scripts/ec2_train_and_upload.sh
#
# Required env (set in .env or the shell -- see README.md section 3):
#   S3_BUCKET              bucket holding the ntds corpus (and where checkpoints land)
# Optional env (defaults shown):
#   TRAIN_CONFIG=long_lowmem_ntds
#   CHECKPOINT_S3_PREFIX=checkpoints
#   SYNC_INTERVAL_SECONDS=300
#   NTDS_LOCAL_DIR=data/ntds
#   NTDS_MANIFEST=data/manifests/clips_v1.parquet
#
# AWS credentials: prefer an IAM instance role attached to the EC2 instance
# over AWS_* keys in .env -- boto3/aws-cli pick it up automatically, nothing
# to configure here, and it avoids putting long-lived credentials on disk.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${S3_BUCKET:?Set S3_BUCKET (env or .env) -- source bucket for the ntds corpus and checkpoint destination}"
TRAIN_CONFIG="${TRAIN_CONFIG:-long_lowmem_ntds}"
CHECKPOINT_S3_PREFIX="${CHECKPOINT_S3_PREFIX:-checkpoints}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
export NTDS_LOCAL_DIR="${NTDS_LOCAL_DIR:-data/ntds}"
export NTDS_MANIFEST="${NTDS_MANIFEST:-data/manifests/clips_v1.parquet}"

RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$HOME/checkpoints/${RUN_TAG}/}"
S3_CHECKPOINT_DEST="s3://${S3_BUCKET}/${CHECKPOINT_S3_PREFIX}/${RUN_TAG}/"

echo "== 1/5: environment =="
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible -- check driver/instance type'; print('CUDA ok:', torch.cuda.get_device_name(0))"

echo "== 2/5: sync manifest =="
mkdir -p "$(dirname "$NTDS_MANIFEST")"
aws s3 sync "s3://${S3_BUCKET}/manifests/clips/dataset_version=v1/" "$(dirname "$NTDS_MANIFEST")/"

echo "== 3/5: sync ntds corpus (resumable -- safe to re-run) =="
python3 scripts/sync_ntds_corpus.py --manifest "$NTDS_MANIFEST" --local-dir "$NTDS_LOCAL_DIR"

echo "== 4/5: train (config=${TRAIN_CONFIG}, checkpoints -> ${S3_CHECKPOINT_DEST}) =="
mkdir -p "$MODEL_SAVE_DIR"

# Stream checkpoints to S3 as they land, in case the instance dies mid-run
# (e.g. spot interruption) -- training itself never touches S3.
(
    while true; do
        sleep "$SYNC_INTERVAL_SECONDS"
        aws s3 sync "$MODEL_SAVE_DIR" "$S3_CHECKPOINT_DEST" --only-show-errors
    done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; echo "== 5/5: final checkpoint sync =="; aws s3 sync "$MODEL_SAVE_DIR" "$S3_CHECKPOINT_DEST" --only-show-errors; echo "checkpoints at ${S3_CHECKPOINT_DEST}"' EXIT

TRAIN_CONFIG="$TRAIN_CONFIG" python3 -m model.training.nonsweep_main
