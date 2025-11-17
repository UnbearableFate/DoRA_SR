from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

from datasets import DatasetDict, load_dataset

from src.config import DataConfig


def _build_prompt(example: Dict[str, str]) -> str:
    if example.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
            f"{example['output']}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )


def _tokenize_example(
    example: Dict[str, str],
    tokenizer,
    cutoff_len: int,
    train_on_inputs: bool,
):
    prompt = _build_prompt(example)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
    )

    if tokenized["input_ids"][-1] != tokenizer.eos_token_id and len(tokenized["input_ids"]) < cutoff_len:
        tokenized["input_ids"].append(tokenizer.eos_token_id)
        tokenized["attention_mask"].append(1)

    labels = tokenized["input_ids"].copy()
    if not train_on_inputs:
        user_prompt = _build_prompt({**example, "output": ""})
        user_ids = tokenizer(
            user_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )["input_ids"]
        user_len = len(user_ids)
        labels = [-100] * user_len + labels[user_len:]

    tokenized["labels"] = labels
    return tokenized


def load_commonsense_dataset(tokenizer, cfg: DataConfig):
    if cfg.dataset_path.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=cfg.dataset_path)
    else:
        dataset = load_dataset(cfg.dataset_path)

    dataset = dataset["train"]
    if cfg.val_size > 0:
        split = dataset.train_test_split(test_size=cfg.val_size, seed=cfg.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = dataset, None

    tokenization_fn = partial(
        _tokenize_example,
        tokenizer=tokenizer,
        cutoff_len=cfg.cutoff_len,
        train_on_inputs=cfg.train_on_inputs,
    )

    train_dataset = train_dataset.shuffle(seed=cfg.seed).map(tokenization_fn, num_proc=cfg.num_proc)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.shuffle(seed=cfg.seed).map(tokenization_fn, num_proc=cfg.num_proc)

    return train_dataset, eval_dataset


__all__ = ["load_commonsense_dataset"]
