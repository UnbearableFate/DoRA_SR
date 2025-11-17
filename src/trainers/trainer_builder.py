from __future__ import annotations

from typing import Optional

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from trl import SFTTrainer

from src.config import TrainingConfig


def _compute_gradient_accumulation(cfg) -> int:
    if cfg.gradient_accumulation_steps:
        return cfg.gradient_accumulation_steps
    if cfg.batch_size:
        return max(1, cfg.batch_size // cfg.per_device_train_batch_size)
    return 1


def build_trainer(model, tokenizer, train_dataset, eval_dataset, cfg: TrainingConfig) -> Trainer:
    grad_accum = _compute_gradient_accumulation(cfg.optim)
    eval_strategy = "steps" if eval_dataset is not None else "no"
    report_to = ["wandb"] if cfg.runtime.use_wandb and cfg.wandb.enabled else []

    training_args = TrainingArguments(
        output_dir=cfg.runtime.output_dir,
        per_device_train_batch_size=cfg.optim.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.optim.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        warmup_ratio=cfg.optim.warmup_ratio,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        num_train_epochs=cfg.optim.num_train_epochs,
        logging_steps=cfg.optim.logging_steps,
        save_steps=cfg.optim.save_steps,
        save_total_limit=cfg.optim.save_total_limit,
        eval_steps=cfg.optim.eval_steps if eval_dataset is not None else None,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        max_grad_norm=cfg.optim.max_grad_norm,
        bf16=cfg.runtime.bf16,
        fp16=cfg.runtime.fp16,
        tf32=cfg.runtime.tf32,
        dataloader_num_workers=cfg.runtime.dataloader_num_workers,
        dataloader_prefetch_factor=cfg.runtime.dataloader_prefetch_factor,
        dataloader_pin_memory=cfg.runtime.dataloader_pin_memory,
        report_to=report_to or None,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    return trainer


__all__ = ["build_trainer"]
