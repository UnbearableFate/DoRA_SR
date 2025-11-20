# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

python my_finetune.py \
    --base_model 'Qwen/Qwen3-1.7B' \
    --data_path '/work/xg24i002/x10041/LLM-Adapters/ft-training_set/commonsense_170k.json' \
    --output_dir ./outputs/test \
    --batch_size 32  --per_device_train_batch_size 4 --num_epochs 3 \
    --learning_rate 1.5e-4 --cutoff_len 256 --val_set_size 128 \
    --eval_step 20 --save_step 20  --adapter_name rslora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --bf16 \
    --init_lora_weights True \
    #--enable_torch_compile \
    #--disable_cudnn_sdpa --disable_flash_sdpa 