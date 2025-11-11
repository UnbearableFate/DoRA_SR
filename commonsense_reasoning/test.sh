# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

python finetune.py \
    --base_model 'Qwen/Qwen3-1.7B' \
    --data_path '/home/yu/workspace/LLM-Adapters/ft-training_set/commonsense_170k.json' \
    --output_dir ./output \
    --batch_size 2  --per_device_train_batch_size 2 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name dora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 8 --lora_alpha 16 --use_gradient_checkpointing