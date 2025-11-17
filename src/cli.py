from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union
from config.common_config import build_config_from_args
from src.data import load_commonsense_dataset
from src.loggers import WandbSession
from src.models.lora_loader import attach_lora_adapter,get_lora_config,  load_base_model, load_tokenizer
from src.trainers import build_trainer
from src.utils import set_seed


def main():
    #parser = build_parser()
    #args = parser.parse_args()
    cfg = build_config_from_args()

    Path(cfg.runtime.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.runtime.seed)

    tokenizer = load_tokenizer(cfg.model)
    train_dataset, eval_dataset = load_commonsense_dataset(tokenizer, cfg.data)
    model = load_base_model(cfg.model)
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
    
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, cfg)

    wandb_cfg = cfg.wandb
    wandb_cfg.enabled = cfg.runtime.use_wandb and cfg.wandb.enabled

    with WandbSession(wandb_cfg, cfg.as_flat_dict()):
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(cfg.runtime.output_dir)
        tokenizer.save_pretrained(cfg.runtime.output_dir)
        trainer.save_state()


if __name__ == "__main__":
    main()
