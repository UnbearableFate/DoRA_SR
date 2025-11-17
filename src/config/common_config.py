
import torch
from .training_config import LoraHyperparameters, OptimizationConfig, RuntimeConfig, TrainingConfig, DataConfig ,ModelConfig, WandbConfig

def build_config() -> TrainingConfig:
    data_cfg = DataConfig(
        dataset_path="/work/xg24i002/x10041/LLM-Adapters/ft-training_set/commonsense_170k.json",
        cutoff_len=512,
        val_size=1024,
        num_proc=8,
        train_on_inputs=True,
        seed=42,
    )
    model_cfg = ModelConfig(
        base_model="meta-llama/Llama-3.1-8B",
        attn_implementation="eager",
        torch_dtype=torch.float32,
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True,
        gradient_checkpointing=False,
        device_map=None,
    )
    lora_cfg = LoraHyperparameters(
        variant="lora",
        r=16,
        alpha=32,
        dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        init_lora_weights=True,
        init_num_samples=512,
        init_batch_size=8,
    )
    optim_cfg = OptimizationConfig(
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.06,
        num_train_epochs=2,
        batch_size=64,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        max_grad_norm=1.0,
    )
    runtime_cfg = RuntimeConfig(
        output_dir="./outputs/r16",
        seed=42,
        bf16=True,
        fp16=False,
        tf32=False,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        use_wandb=True,
    )
    run_name = model_cfg.base_model.replace("/", "_") + f"_lora_r{lora_cfg.r}_alpha{lora_cfg.alpha}_seed{runtime_cfg.seed}"
    wandb_cfg = WandbConfig(
        project="commonsense_reasoning",
        entity=None,
        run_name="Llama-3.1-8B_lora_r16_alpha32_seed42",
        tags=[
            "Llama-3.1-8B",
            "lora",
            "bs64",
            "lr0.0002",
            "lora_r16",
            "lora_alpha32",
            "seed42",
        ],
        mode="online",
        enabled=True,
    )
    return TrainingConfig(
        data=data_cfg,
        model=model_cfg,
        lora=lora_cfg,
        optim=optim_cfg,
        runtime=runtime_cfg,
        wandb=wandb_cfg,
    )
