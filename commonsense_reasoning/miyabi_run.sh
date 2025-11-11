#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"
echo "Current working directory: $(pwd)"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-accelerate_config.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

export MASTER_ADDR MASTER_PORT
export ACCELERATE_CONFIG_FILE="$ACCELERATE_CONFIG"

ENV_VARS=("MASTER_ADDR=${MASTER_ADDR}" "MASTER_PORT=${MASTER_PORT}" "ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG}")
ENV_LIST=$(IFS=,; echo "${ENV_VARS[*]}")
if [[ -n "${OMPI_MCA_mca_base_env_list:-}" ]]; then
    export OMPI_MCA_mca_base_env_list="${OMPI_MCA_mca_base_env_list},${ENV_LIST}"
else
    export OMPI_MCA_mca_base_env_list="${ENV_LIST}"
fi

PYTHON_PATH="/work/xg24i002/x10041/lora-ns/.venv/bin/python"

HF_HOME="/work/xg24i002/x10041/hf_home"
HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

mpirun --mca mpi_abort_print_stack 1 \
       --report-bindings \
       --bind-to core \
       -np "${WORLD_SIZE}" \
       /usr/bin/env \
           MASTER_ADDR="${MASTER_ADDR}" \
           MASTER_PORT="${MASTER_PORT}" \
           ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG}" \
       bash -c "set -euo pipefail; \
                : \"\${MASTER_ADDR:?MASTER_ADDR not set}\"; \
                : \"\${MASTER_PORT:?MASTER_PORT not set}\"; \
                : \"\${ACCELERATE_CONFIG_FILE:?ACCELERATE_CONFIG_FILE not set}\"; \
                export RANK=\$OMPI_COMM_WORLD_RANK; \
                export WORLD_SIZE=\$OMPI_COMM_WORLD_SIZE; \
                export LOCAL_RANK=\$OMPI_COMM_WORLD_LOCAL_RANK; \
                export LOCAL_WORLD_SIZE=\$OMPI_COMM_WORLD_LOCAL_SIZE; \
                export HF_HOME='${HF_HOME}'; \
                export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'; \
                echo 'Running on rank' \$RANK 'out of' \$WORLD_SIZE; \
                ${PYTHON_PATH} python finetune.py \
                    --base_model 'Qwen/Qwen3-1.7B' \
                    --data_path '/home/yu/workspace/LLM-Adapters/ft-training_set/commonsense_170k.json' \
                    --output_dir ./output \
                    --batch_size 16  --micro_batch_size 4 --num_epochs 3 \
                    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
                    --eval_step 80 --save_step 80  --adapter_name dora \
                    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
                    --lora_r 8 --lora_alpha 16 --use_gradient_checkpointing 
