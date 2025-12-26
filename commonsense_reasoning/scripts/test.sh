#!/bin/bash

cd commonsense_reasoning

python my_finetune_go.py  \
    --base_model=Qwen/Qwen3-1.7B \
    --data_path=/home/yu/workspace/LLM-Adapters/ft-training_set/commonsense_170k.json \
    --output_dir=./outputs/ \
    --batch_size=32 \
    --per_device_train_batch_size=2 \
    --num_epochs=0.005 \
    --learning_rate=5e-4 \
    --lr_scheduler_type=linear \
    --warmup_step=160 \
    --weight_decay=0.0 \
    --cutoff_len=512 \
    --val_set_size=1024 \
    --eval_step=100 \
    --save_step=100 \
    --adapter_name=lora \
    --target_modules="[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]" \
    --lora_r=64 \
    --lora_alpha=4 \
    --lora_dropout=0.0 \
    --bf16 \
    --init_lora_weights=True \
    --timestamp='"${timestamp}"' \
    --seed=17 \
    --wandb_project=cs_qwen \
    --enable_torch_compile \
    --sr_init_steps=10 \
    --adjust_lora_alpha=1 \
    --do_refactor=True \
    --keep_s=True \
    --min_alpha_ratio=0.8 \
    --max_alpha_ratio=1.6 \