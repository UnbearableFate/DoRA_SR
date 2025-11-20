from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding

try:  # Optional dependency for quantization
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - bitsandbytes not always available
    BitsAndBytesConfig = None

from peft import LoraConfig, get_peft_model, initialize_lora_eva_weights, prepare_model_for_kbit_training
from peft.tuners.lora.corda import preprocess_corda
from peft.tuners.lora.config import CordaConfig, EvaConfig

@dataclass
class LoraHyperparameters:
    """Hyperparameters for the LoRA-family adapters."""

    variant: str = "lora"  # lora, dora, qalora, rslora
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    init_lora_weights: Union[bool, str, None] = True # ["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
    init_num_samples: int = 512
    init_batch_size: int = 8
    corda_method: str = "kpm"  # kpm or ipm, only used if init_lora_weights is "corda"

@dataclass
class ModelConfig:
    """Base model loading parameters."""

    base_model: str = "meta-llama/Llama-3.1-8B"
    attn_implementation: Optional[str] = "sdpa"
    torch_dtype = torch.float32
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    gradient_checkpointing: bool = False
    pad_token: str = "<|pad|>"
    device_map: Optional[str] = None

_VARIANT_TO_FLAGS = {
    "lora": {"use_dora": False, "use_rslora": False, "use_qalora": False},
    "dora": {"use_dora": True, "use_rslora": False, "use_qalora": False},
    "rslora": {"use_dora": False, "use_rslora": True, "use_qalora": False},
    "qalora": {"use_dora": False, "use_rslora": False, "use_qalora": True},
}

def _maybe_quant_config(model_cfg: ModelConfig):
    if BitsAndBytesConfig is None:
        return None
    if model_cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if model_cfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.base_model,
        trust_remote_code=model_cfg.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": model_cfg.pad_token})
    tokenizer.pad_token = tokenizer.pad_token or model_cfg.pad_token
    return tokenizer


def load_base_model(model_cfg: ModelConfig):
    quantization_config = _maybe_quant_config(model_cfg)
    dtype = model_cfg.torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=None if quantization_config else dtype,
        device_map=model_cfg.device_map,
        quantization_config=quantization_config,
    )
    if (model_cfg.load_in_4bit or model_cfg.load_in_8bit) and quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model_cfg.gradient_checkpointing)
    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def get_lora_config(lora_cfg: LoraHyperparameters):
    variant = lora_cfg.variant.lower()
    if variant not in _VARIANT_TO_FLAGS:
        raise ValueError(f"Unsupported LoRA variant: {variant}")
    
    corda_config = None
    eva_config = None
    if lora_cfg.init_lora_weights == "corda":
            corda_config = CordaConfig(
                corda_method=lora_cfg.corda_method, # kpm or ipm
            )
    elif lora_cfg.init_lora_weights == "eva":
        eva_config = EvaConfig()

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=list(lora_cfg.target_modules),
        task_type="CAUSAL_LM",
        init_lora_weights=lora_cfg.init_lora_weights,
        corda_config=corda_config,
        eva_config=eva_config,
        **_VARIANT_TO_FLAGS[variant],
    )

    return peft_config

def attach_lora_adapter(base_model,lora_cfg: LoraConfig, train_dataset,tokenizer, init_num_samples:int, batch_size:int,seed: int):
    if lora_cfg.init_lora_weights not in ["corda", "eva"]:
        return get_peft_model(base_model, lora_cfg)
    
    sub_dataset = train_dataset.shuffle(seed=seed).select(range(init_num_samples))
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    columns_to_remove = [col for col in sub_dataset.column_names if col not in columns_to_keep]
    sub_dataset = sub_dataset.remove_columns(columns_to_remove) if columns_to_remove else sub_dataset
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    if lora_cfg.init_lora_weights == "corda":
        return get_peft_model_with_corda(base_model, lora_cfg, sub_dataset,data_collator)
    elif lora_cfg.init_lora_weights == "eva":
        return get_peft_model_with_eva(base_model, lora_cfg, sub_dataset,data_collator ,batch_size)

def get_peft_model_with_corda(base_model,lora_cfg: LoraConfig,sub_dataset,data_collator):
    calib_loader = DataLoader(
        sub_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
    )

    device = base_model.device
    print(f"Running Corda preprocessing on device: {device}")

    @torch.no_grad()
    def _run_model():
        was_training = base_model.training
        base_model.eval()
        # for batch in calib_loader:
        #     batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        #     base_model(**batch)
        for batch in tqdm.tqdm(calib_loader, desc="corda preprocessing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            base_model(**batch)
        if was_training:
            base_model.train()

    print(f"Starting Corda preprocessing... with sub-dataset of size {len(sub_dataset)}")
    preprocess_corda(
        base_model,
        lora_cfg,
        run_model=_run_model,
    )
    return get_peft_model(base_model, lora_cfg)

def get_peft_model_with_eva(
        base_model,
        lora_cfg: LoraConfig,
        sub_dataset,
        data_collator,
        batch_size: int,
    ):
    
    def get_input(examples):
        batch = data_collator(examples)
        print(batch.__class__)
        return {k: v.to(base_model.device) for k, v in batch.items()}
    
    dataloader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        collate_fn=get_input,
    )

    peft_model = get_peft_model(base_model, lora_cfg, low_cpu_mem_usage=True)
    print(f"Initializing Eva LoRA weights... with sub-dataset of size {len(sub_dataset)}")
    initialize_lora_eva_weights(peft_model, dataloader)
    return peft_model

__all__ = [
    "load_base_model",
    "load_tokenizer",
    "get_lora_config",
]
