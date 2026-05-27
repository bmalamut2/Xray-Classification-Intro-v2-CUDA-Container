#!/usr/bin/env bash
#SBATCH --job-name=arkplus-swin-base
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH -o /scratch/bmalamut/classification_ark_swin_%j.out
#SBATCH -e /scratch/bmalamut/classification_ark_swin_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="bmalamut@asu.edu"
#SBATCH --export=NONE

set -euo pipefail
cd /scratch/bmalamut/Xray-Classification-Intro-v2-CUDA-Container/Pytorch

# Examples:
# ./apptainer-run-arkplus-swin_base.sh dataset_dir=$XRAY_DATASET debug=true ark.pretrain_epochs=1
# ./apptainer-run-arkplus-swin_base.sh 4 ark.global_batch_size=200 ark.workers=8

MODEL="swin_base"
GLOBAL_BATCH_SIZE="${ARKPLUS_GLOBAL_BATCH_SIZE:-200}"
OUTPUT_DIR="${ARKPLUS_OUTPUT_DIR:-${SCRATCH:-$PWD}/outputs-Xray-Classification-Intro-v2/${CLUSTER:-local}}"
mkdir -p "$OUTPUT_DIR"

APPTAINER_CMD=(/scratch/bmalamut/Xray-Classification-Intro-v2-CUDA-Container/Pytorch/cuda-apptainer.sh exec)

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
