#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

cd /work/xg24i002/x10041/DoRA_SR/commonsense_reasoning

PYTHON_PATH="/work/xg24i002/x10041/lora-ns/.venv/bin/"
LORA_WEIGHT="/work/xg24i002/x10041/DoRA_SR/commonsense_reasoning/outputs/eva_hp/Llama-3.1-8B/R16/Llama-3.1-8B_r16_alpha1_eva_lora_normal_seed17_20251121182901"

export HF_HOME="/work/xg24i002/x10041/hf_home"
export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

timestamp=$(date +%Y%m%d_%H%M%S)

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset piqa \
    --base_model meta-llama/Llama-3.1-8B \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights "${LORA_WEIGHT}"