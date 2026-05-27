#!/usr/bin/env bash
#SBATCH --job-name=arkplus-finetune
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "$(dirname "$0")" || exit 1

usage() {
    echo "Usage: ./apptainer-run-arkplus-finetune.sh <dataset> <model> <arkplus_checkpoint> [num_gpus] [hydra overrides...]"
    echo ""
    echo "Examples:"
    echo "  ./apptainer-run-arkplus-finetune.sh chestxray14 swin_base /path/to/best_teacher.pth.tar 1 debug=true"
    echo "  ./apptainer-run-arkplus-finetune.sh chexpert resnet50 /path/to/last_teacher.pth.tar 4 train.batch_size=64"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

DATASET="$1"
MODEL="$2"
ARKPLUS_CHECKPOINT="$3"
shift 3

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="$1"
    shift
else
    NUM_GPUS="${ARKPLUS_FINETUNE_NUM_GPUS:-1}"
fi

OPTIMIZER="${ARKPLUS_FINETUNE_OPTIMIZER:-sgd}"
LR="${ARKPLUS_FINETUNE_LR:-0.01}"
BATCH_SIZE="${ARKPLUS_FINETUNE_BATCH_SIZE:-64}"
EPOCHS="${ARKPLUS_FINETUNE_EPOCHS:-200}"
OUTPUT_DIR="${ARKPLUS_FINETUNE_OUTPUT_DIR:-${SCRATCH:-$PWD}/outputs-Xray-Classification-Intro-v2/${CLUSTER:-local}}"
mkdir -p "$OUTPUT_DIR"

APPTAINER_CMD=(./cuda-apptainer.sh exec)
TRAIN_ARGS=(
    src/train_v1_classic.py
    dataset="$DATASET"
    model="$MODEL"
    model.pretrained=false
    model.arkplus_checkpoint="$ARKPLUS_CHECKPOINT"
    model.arkplus_checkpoint_key=teacher
    model.arkplus_load_mode=encoder
    train.batch_size="$BATCH_SIZE"
    train.optimizer="$OPTIMIZER"
    train.lr="$LR"
    train.epochs="$EPOCHS"
    train.weight_decay=0.0
    scheduler=cosine_annealing
    hydra.run.dir="$OUTPUT_DIR"
)

if [ "$NUM_GPUS" -gt 1 ]; then
    FREE_PORT=$("${APPTAINER_CMD[@]}" python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' | tail -n 1)
    "${APPTAINER_CMD[@]}" torchrun --nproc_per_node="$NUM_GPUS" --master_port="$FREE_PORT" "${TRAIN_ARGS[@]}" "$@"
else
    "${APPTAINER_CMD[@]}" python "${TRAIN_ARGS[@]}" "$@"
fi
