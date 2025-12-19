#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

cd /work/xg24i002/x10041/DoRA_SR/commonsense_reasoning

PYTHON_PATH="/work/xg24i002/x10041/lora-ns/.venv/bin/"
LORA_WEIGHT="/work/xg24i002/x10041/DoRA_SR/commonsense_reasoning/outputs/Llama-3.1-8B/R16/3"

export HF_HOME="/work/xg24i002/x10041/hf_home"
export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

timestamp=$(date +%Y%m%d_%H%M%S)

"${PYTHON_PATH}python" my_commonsense_evaluate.py \
    --base_model meta-llama/Llama-3.1-8B \
    --lora_weights_dir "${LORA_WEIGHT}" \
    --output_dir "${timestamp}-aaa"