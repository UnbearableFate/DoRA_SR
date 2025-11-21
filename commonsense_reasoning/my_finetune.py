# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

import wandb

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    LoraRuntimeConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402
from accelerate import Accelerator ,DistributedType

from SpectralRefactorTrainer import SpectralRefactorTrainer


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./output",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        per_device_train_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        init_lora_weights: str = True,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # Dora hyperparams
        dora_simple: bool = True,
        Wdecompose_target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_online: bool = True,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        timestamp: Optional[str] = "",
        seed: int = 42,
        bf16: bool = False,
        # stability params
        attn_implementation: str = "eager",  # use "eager" for stability, "flash_attention_2" after stack upgrade
        disable_cudnn_sdpa: bool = False,  # disable cuDNN SDPA to avoid bf16 instability
        disable_flash_sdpa: bool = False,  # disable Flash SDPA if needed
        enable_torch_compile: bool = False,  # enable after confirming stability
        #trainer
        use_sr_trainer: bool = False,
):
    # Configure SDPA backends for bf16 stability
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # https://github.com/pytorch/pytorch/issues/100005
    if disable_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("Disabled cuDNN SDPA for bf16 stability")
    if disable_flash_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        print("Disabled Flash SDPA")
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(
            f"Finetuning model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"Wdecompose_target_modules: {Wdecompose_target_modules}\n"
            f"dora_simple: {dora_simple}"
            f"bottleneck_size: {bottleneck_size}\n"
            f"non_linearity: {non_linearity}\n"
            f"adapter_dropout: {adapter_dropout}\n"
            f"use_parallel_adapter: {use_parallel_adapter}\n"
            f"use_adapterp: {use_adapterp}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"scaling: {scaling}\n"
            f"adapter_name: {adapter_name}\n"
            f"target_modules: {target_modules}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_online: {wandb_online}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint}\n"
            f"seed: {seed}\n"
            f"bf16: {bf16}\n"
            f"attn_implementation: {attn_implementation}\n"
            f"disable_cudnn_sdpa: {disable_cudnn_sdpa}\n"
            f"enable_torch_compile: {enable_torch_compile}\n"
            f"use_sr_trainer: {use_sr_trainer}\n"
        )
        print(accelerator.state)

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    # device_map="auto" is for inference only; not supported for training
    # https://discuss.huggingface.co/t/what-is-the-proper-way-to-use-device-map-auto-with-trainer/31801
    device_map = None

    seed_everything(seed)

    tags = [base_model.split("/")[-1], adapter_name, f"bs{batch_size}",f'lr{learning_rate}',f"lora_r{lora_r}", f"lora_alpha{lora_alpha}", f"seed{seed}"]
    wandb_run_name = f"{base_model.split('/')[-1]}_r{lora_r}_alpha{lora_alpha}_{init_lora_weights}_{adapter_name}_{"sr" if use_sr_trainer else "normal"}_seed{seed}_{timestamp}"
    output_dir = os.path.join(output_dir,base_model.split('/')[-1],f"R{lora_r}",wandb_run_name)
    
    if accelerator.is_main_process:
        if wandb_online:
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project="commonsense_reasoning",
            name=wandb_run_name,
            config=locals(),
            tags=tags
        )
    
    # Keep master weights in fp32 for stability with bf16 training
    # bf16=True in TrainingArguments uses AMP, not hard-casting
    # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#bf16
    dtype = torch.float32  # Always use fp32 for master weights

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        # Use eager attention for bf16 stability
        # https://huggingface.co/docs/transformers/en/attention_interface
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,  # fp32 master weights
            device_map=device_map,  # None for training
            attn_implementation=attn_implementation,  # "eager" for stability
            trust_remote_code=True,
        )

    print(f"Model loaded successfully. dtype: {model.dtype}, attn_implementation: {attn_implementation}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Fix Llama-3.x tokenizer: add dedicated PAD token
    # Token id 0 is "!" in Llama-3 tokenizers, not a valid PAD
    # https://github.com/turboderp/exllamav2/issues/415
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
        if accelerator.is_main_process:
            print(f"Adding dedicated PAD token (original pad_token_id: {tokenizer.pad_token_id})")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        if accelerator.is_main_process:
            print(f"New pad_token_id: {tokenizer.pad_token_id}, vocab size: {len(tokenizer)}")
    else:
        # A safer fallback is to use the EOS token if a PAD token is not defined
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": "", "answer": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    
    # PEFT keeps adapters in fp32 by default for stability
    # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    # Move model to device (but keep fp32 dtype for master weights)
    model.to('cuda')

    if data_path.endswith((".json", ".jsonl")):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=4)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt , num_proc=4)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=4)
        val_data = None

    from lora_loader import get_lora_config, LoraHyperparameters, attach_lora_adapter
    
    lora_hyperparams = LoraHyperparameters(
        variant=adapter_name,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        init_lora_weights=init_lora_weights,
        init_num_samples=1024,
        init_batch_size=2,
    )

    lora_config = get_lora_config(lora_hyperparams)
    if accelerator.is_main_process:
        print(f"LoRA config: {lora_config}")
    
    model = attach_lora_adapter(
        base_model=model,
        lora_cfg=lora_config,
        train_dataset=train_data,
        tokenizer=tokenizer,
        init_num_samples=lora_hyperparams.init_num_samples,
        batch_size=lora_hyperparams.init_batch_size,
        seed=seed,
    )

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
   
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    if accelerator.distributed_type == DistributedType.MULTI_GPU and accelerator.num_processes > 1:
        gradient_accumulation_steps = gradient_accumulation_steps // accelerator.num_processes
        assert gradient_accumulation_steps * accelerator.num_processes * per_device_train_batch_size == batch_size, f"batch_size {batch_size}, num_processes {accelerator.num_processes}, per_device_train_batch_size {per_device_train_batch_size} not aligned."

    common_args = {
        "model": model,
        "train_dataset": train_data,
        "eval_dataset": val_data,
        "args": transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type="cosine",
            warmup_steps=300,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            # bf16=True uses AMP (automatic mixed precision) for bf16 compute
            # while keeping master weights in fp32 - this is the recommended approach
            # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#bf16
            bf16=bf16,
            fp16=False,
            logging_steps=50,
            optim="adamw_torch",
            max_grad_norm=1.0,
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters= False,
            group_by_length=group_by_length,
            report_to="wandb" if accelerator.is_main_process else None,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            dataloader_num_workers=8,
            dataloader_prefetch_factor=4,
            data_seed=seed,
            seed=seed,
            ),
        "data_collator": transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        }
    if use_sr_trainer:
        if accelerator.is_main_process:
            print(f"Using SpectralRefactorTrainer, refactor_every=100, balance_lambda=0.8")
        trainer = SpectralRefactorTrainer(
            **common_args,
            refactor_every = 100,
            balance_lambda = 0.8,
        )
    else:
        trainer = transformers.Trainer(
            **common_args,
        )
        
    model.config.use_cache = False

    # Only enable torch.compile after confirming training stability
    # https://github.com/pytorch/pytorch/issues/100005
    if torch.__version__ >= "2" and sys.platform != "win32" and enable_torch_compile:
        if accelerator.is_main_process:
            print("Enabling torch.compile")
        model = torch.compile(model)
    elif enable_torch_compile and accelerator.is_main_process:
        print("torch.compile requested but not available (PyTorch < 2.0 or Windows)")

    accelerator.wait_for_everyone()
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if accelerator.is_main_process:
        print(
            f"Training time: {((time.time() - start_time)/60):.2f} minutes for {num_epochs} epochs"
        )
        model.save_pretrained(output_dir)
        wandb.finish()
    
    accelerator.wait_for_everyone()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    accelerator.end_training()

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""{data_point["instruction"]}. {data_point["input"]}. {data_point["output"]}""" # noqa: E501
    else:
        return f"""{data_point["instruction"]}. {data_point["output"]}""" # noqa: E501

# def generate_prompt(data_point):
#     # sorry about the formatting disaster gotta move fast
#     if data_point["input"]:
#         return f"""{data_point["instruction"]}. {data_point["input"]}. The correct answer is {data_point["answer"]}""" # noqa: E501
#     else:
#         return f"""{data_point["instruction"]}. The correct answer is {data_point["answer"]}""" # noqa: E501

if __name__ == "__main__":
    fire.Fire(train)
