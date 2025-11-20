python my_commonsense_evaluate.py \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --batch_size 16 \
    --lora_weights '/work/xg24i002/x10041/DoRA_SR/commonsense_reasoning/outputs/Llama-3.1-8B/R32/Llama-3.1-8B_r32_alpha64_True_lora_normal_seed17_20251120061346'