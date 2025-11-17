from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    project: str = "commonsense_reasoning"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"
    enabled: bool = True


@dataclass
class DataConfig:
    """Dataset and preprocessing parameters."""

    dataset_path: str = "./commonsense_reasoning/commonsense_170k.json"
    cutoff_len: int = 512
    val_size: int = 1024
    num_proc: int = 8
    train_on_inputs: bool = True
    seed: int = 42


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
    init_lora_weights: Union[bool, str, None] = True
    init_num_samples: int = 512
    init_batch_size: int = 8


@dataclass
class OptimizationConfig:
    """Optimizer and scheduler settings."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_train_epochs: int = 2
    batch_size: int = 64
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: Optional[int] = None
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 50
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    max_grad_norm: float = 1.0


@dataclass
class RuntimeConfig:
    """Misc runtime knobs."""

    output_dir: str = "./outputs"
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool = True
    use_wandb: bool = True


@dataclass
class TrainingConfig:
    """Combined configuration bundle passed across the training system."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraHyperparameters = field(default_factory=LoraHyperparameters)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def as_flat_dict(self) -> dict:
        """Return a flattened dictionary for experiment logging."""

        return {
            "data": vars(self.data),
            "model": vars(self.model),
            "lora": {
                **{k: v for k, v in vars(self.lora).items() if k != "target_modules"},
                "target_modules": list(self.lora.target_modules),
            },
            "optim": vars(self.optim),
            "runtime": vars(self.runtime),
            "wandb": vars(self.wandb),
        }
