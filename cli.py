from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import wandb
from src.config.common_config import build_common_config
from src.data import load_commonsense_dataset
from src.models.lora_loader import attach_lora_adapter,get_lora_config,  load_base_model, load_tokenizer
from src.trainers import build_trainer
from src.utils import set_seed
from accelerate import Accelerator

def main():
    #parser = build_parser()
    #args = parser.parse_args()
    cfg = build_common_config()
    accelerator = Accelerator()

    Path(cfg.runtime.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.runtime.seed)

    tokenizer = load_tokenizer(cfg.model)
    train_dataset, eval_dataset = load_commonsense_dataset(tokenizer, cfg.data)
    model = load_base_model(cfg.model)
    model.to("cuda")
    if getattr(model, "get_input_embeddings", None) is not None:
        model.resize_token_embeddings(len(tokenizer))

    lora_config = get_lora_config(cfg.lora)
    
    model = attach_lora_adapter(
        model,
        lora_config,
        train_dataset,
        tokenizer, 
        init_num_samples=cfg.lora.init_num_samples,
        batch_size=cfg.lora.init_batch_size,
        seed=cfg.runtime.seed,
    )
    model.print_trainable_parameters()
    model.to("cuda")
    
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, cfg)

    wandb_cfg = cfg.wandb
    wandb_cfg.enabled = cfg.runtime.use_wandb and cfg.wandb.enabled

    wandb_run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.run_name,
            tags=wandb_cfg.tags,
            config=wandb_cfg.config_payload,
        )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(cfg.runtime.output_dir)

if __name__ == "__main__":
    main()
