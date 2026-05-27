#!/usr/bin/env bash
#SBATCH --job-name=arkplus-swin-base
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH -o /scratch/bmalamut/classification_ark_swin_continued_%j.out
#SBATCH -e /scratch/bmalamut/classification_ark_swin_continued_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="bmalamut@asu.edu"
#SBATCH --export=NONE

cd /scratch/bmalamut/Xray-Classification-Intro-v2-CUDA-Container/Pytorch

echo "Start: $(date)"
nvidia-smi

./apptainer-run-arkplus-swin_base.sh 4 ark.resume=/scratch/bmalamut/outputs-Xray-Classification-Intro-v2/local/ArkPlus_swin_base_patch4_window7_224_sgd_0_3_run/checkpoints/last_teacher.pth.tar ark.workers=1 ark.save_test_metrics=false

echo "End: $(date)"
