#!/usr/bin/env bash
#SBATCH --job-name=arkplus-resnet50
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4

set -euo pipefail
cd "$(dirname "$0")" || exit 1

# Examples:
# ./apptainer-run-arkplus-resnet50.sh dataset_dir=$XRAY_DATASET debug=true ark.pretrain_epochs=1
# ./apptainer-run-arkplus-resnet50.sh 4 ark.global_batch_size=200 ark.workers=8

MODEL="resnet50"
GLOBAL_BATCH_SIZE="${ARKPLUS_GLOBAL_BATCH_SIZE:-200}"
OUTPUT_DIR="${ARKPLUS_OUTPUT_DIR:-${SCRATCH:-$PWD}/outputs-Xray-Classification-Intro-v2/${CLUSTER:-local}}"
mkdir -p "$OUTPUT_DIR"

APPTAINER_CMD=(./cuda-apptainer.sh exec)

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="$1"
    shift
else
    NUM_GPUS="${ARKPLUS_NUM_GPUS:-4}"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    FREE_PORT=$("${APPTAINER_CMD[@]}" python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' | tail -n 1)
    "${APPTAINER_CMD[@]}" torchrun --nproc_per_node="$NUM_GPUS" --master_port="$FREE_PORT" src/train_ark_plus.py model="$MODEL" ark.global_batch_size="$GLOBAL_BATCH_SIZE" hydra.run.dir="$OUTPUT_DIR" "$@"
else
    "${APPTAINER_CMD[@]}" python src/train_ark_plus.py model="$MODEL" ark.global_batch_size="$GLOBAL_BATCH_SIZE" hydra.run.dir="$OUTPUT_DIR" "$@"
fi
