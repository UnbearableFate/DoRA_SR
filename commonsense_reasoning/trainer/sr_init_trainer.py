from copy import deepcopy
import os
from typing import Optional, Set
from peft import LoraModel, PeftModel
import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

import math

def smooth_asymmetric_power_ratio_math(
    ratio: float,
    beta_pos: float = 0.5,   # r > 1 时的放大强度
    beta_neg: float = 0.25,  # r < 1 时的压缩强度（更接近 1）
    tau: float = 0.4,        # log 域平滑宽度
    eps: float = 1e-12,
) -> float:
    """
    Smoothly maps ratio=v/mean_v (ratio>0) to a multiplicative factor g(ratio),
    where negative log-ratios are compressed (smaller slope) and positive side
    keeps stronger scaling.

    Properties:
      - g(1) = 1
      - For large ratio: g(r) ≈ r^{beta_pos}
      - For small ratio: g(r) ≈ r^{beta_neg}
      - Continuous and differentiable everywhere
    """
    # 数值安全
    r = max(ratio, eps)
    x = math.log(r)  # log-ratio

    # 平滑插值系数 in (0,1)
    s = 0.5 * (1.0 + math.tanh(x / tau))

    # 非对称 beta
    beta_x = beta_neg + (beta_pos - beta_neg) * s

    # 指数映射回比例
    return math.exp(beta_x * x)

import math
from typing import Iterable, Tuple, Optional, List

def _quantile(sorted_x: List[float], q: float) -> float:
    """Linear-interpolated quantile for q in [0,1]. Assumes sorted_x is sorted."""
    n = len(sorted_x)
    if n == 0:
        raise ValueError("Empty data.")
    if q <= 0:
        return sorted_x[0]
    if q >= 1:
        return sorted_x[-1]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_x[lo]
    w = pos - lo
    return sorted_x[lo] * (1 - w) + sorted_x[hi] * w


def infer_betas_from_ratios(
    var_of_layers: Iterable[float],
    alpha_min: float,
    alpha_max: float,
    *,
    use_quantiles: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """
    Infer beta_pos and beta_neg from ratio statistics so that:
      r_high^beta_pos ≈ alpha_max  (for r_high > 1)
      r_low^beta_neg  ≈ alpha_min  (for r_low < 1)

    Returns:
      (beta_pos, beta_neg, r_low_used, r_high_used)

    Notes:
      - If use_quantiles=True, uses (q_low, q_high) quantiles for robustness
        instead of raw min/max.
      - If data do not contain ratios <1 or >1, it falls back to beta=0 on that side.
    """
    mean_v = sum(var_of_layers) / len(var_of_layers)
    ratios = [v / mean_v for v in var_of_layers]
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("alpha_min and alpha_max must be > 0.")
    if alpha_min >= 1.0:
        raise ValueError("For meaningful compression of r<1, alpha_min should be < 1.")
    if alpha_max <= 1.0:
        raise ValueError("For meaningful expansion of r>1, alpha_max should be > 1.")

    xs = [max(float(r), eps) for r in ratios]
    if not xs:
        raise ValueError("ratios is empty.")

    xs.sort()
    r_low = _quantile(xs, q_low) if use_quantiles else xs[0]
    r_high = _quantile(xs, q_high) if use_quantiles else xs[-1]

    # Ensure we have usable sides; otherwise set the corresponding beta to 0.
    # Positive side: need r_high > 1
    if r_high <= 1.0 + 1e-15:
        beta_pos = 0.0
        r_high_used = 1.0
    else:
        beta_pos = math.log(alpha_max) / math.log(r_high)
        r_high_used = r_high

    # Negative side: need r_low < 1
    if r_low >= 1.0 - 1e-15:
        beta_neg = 0.0
        r_low_used = 1.0
    else:
        # log(r_low) < 0 and log(alpha_min) < 0 -> beta_neg > 0
        beta_neg = math.log(alpha_min) / math.log(r_low)
        r_low_used = r_low

    # Guard against pathological huge betas (can happen if r_low ~ 1 or r_high ~ 1)
    # You can adjust these caps if you like.
    beta_pos = max(0.0, min(beta_pos, 10.0))
    beta_neg = max(0.0, min(beta_neg, 10.0))

    return beta_pos, beta_neg, r_low_used, r_high_used


def iter_lora_factors_with_names(model: nn.Module,
                                 target_adapter_keys: Optional[Set[str]] = None):
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for name in module.lora_A.keys():
                if target_adapter_keys and name not in target_adapter_keys:
                    continue
                yield module_name, name, module.lora_B[name].weight, module.lora_A[name].weight


class SvdRefactorInitTrainer(Trainer):
    """
    使用分布式低秩 SVD 重构 LoRA 因子，并根据方差调整 LoRA alpha
    适用于分布式训练环境
    """

    def __init__(
        self,
        *args,
        target_adapter_keys: Optional[Set[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_adapter_keys = set(target_adapter_keys) if target_adapter_keys else None
        self.alpha_log = {}

    @torch.no_grad()
    def distributed_low_rank_refactor(self,adjust_lora_alpha: bool = True,min_alpha_ratio: float = 0.8, max_alpha_ratio: float = 1.6):
        is_dist = self.accelerator.num_processes > 1 and dist.is_initialized()
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index

        model = self.model
        was_training = model.training
        model.eval()

        variance_of_layers = {}
        device_for_broadcast = None
        broadcast_works = []

        for idx, (module_name, name, B, A) in enumerate(
            iter_lora_factors_with_names(model, self.target_adapter_keys)
        ):
            lora_r = A.shape[0]
            compute_here = (not is_dist) or (rank == idx % world_size)
            if device_for_broadcast is None:
                device_for_broadcast = B.device

            if compute_here:
                # 始终对 B @ A 做 SVD，作为正交基
                base = B @ A
                U, S, Vh = torch.svd_lowrank(base.float(), q=lora_r)
                variance_of_layers[f"{module_name}.{name}"] = float((S ** 2).sum().item())
                B_new = U.to(B.dtype)
                A_new = Vh.t().to(A.dtype)
                B.copy_(B_new)
                A.copy_(A_new)

            if is_dist:
                broadcast_works.append(
                    dist.broadcast(B, src=idx % world_size, async_op=True).get_future()
                )
                broadcast_works.append(
                    dist.broadcast(A, src=idx % world_size, async_op=True).get_future()
                )

        if is_dist and broadcast_works:
            torch.futures.wait_all(broadcast_works)

        if is_dist:
            gathered_variances = [None] * world_size
            dist.all_gather_object(gathered_variances, variance_of_layers)
            variance_of_layers = {}
            for part in gathered_variances:
                variance_of_layers.update(part)
        
        beta_pos, beta_neg, r_low_used, r_high_used = infer_betas_from_ratios(variance_of_layers.values(), min_alpha_ratio, max_alpha_ratio)

        if adjust_lora_alpha and variance_of_layers:
            if rank == 0:
                avg_of_global_variance = sum(variance_of_layers.values()) / len(variance_of_layers)
                for module_name, sub_module in model.named_modules():
                    if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                        for adapter_name in sub_module.lora_A.keys():
                            if self.target_adapter_keys and adapter_name not in self.target_adapter_keys:
                                continue
                            if adapter_name not in sub_module.lora_alpha:
                                continue
                            layer_key = f"{module_name}.{adapter_name}"
                            if layer_key not in variance_of_layers:
                                continue
                            layer_var = variance_of_layers[layer_key]
                            ratio = layer_var / avg_of_global_variance
                            ratio_new = smooth_asymmetric_power_ratio_math(ratio, beta_pos=beta_pos, beta_neg=beta_neg)
                            sub_module.lora_alpha[adapter_name] *= ratio_new

            alpha_values = []
            alpha_indices = []
            for module_name, sub_module in model.named_modules():
                if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                    for adapter_name in sub_module.lora_A.keys():
                        if self.target_adapter_keys and adapter_name not in self.target_adapter_keys:
                            continue
                        if adapter_name not in sub_module.lora_alpha:
                            continue
                        alpha_indices.append((module_name, adapter_name))
                        if rank == 0:
                            alpha_values.append(float(sub_module.lora_alpha[adapter_name]))
                        else:
                            alpha_values.append(0.0)

            if alpha_values:
                if device_for_broadcast is None:
                    device_for_broadcast = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                alpha_tensor = torch.tensor(alpha_values, device=device_for_broadcast, dtype=torch.float32)
                if is_dist:
                    dist.broadcast(alpha_tensor, src=0)

                modules_dict = dict(model.named_modules())
                for idx, (module_name, adapter_name) in enumerate(alpha_indices):
                    sub_module = modules_dict[module_name]
                    if f"{module_name}.{adapter_name}" not in self.alpha_log:
                        self.alpha_log[f"{module_name}.{adapter_name}"] = [sub_module.lora_alpha[adapter_name]]
                    sub_module.lora_alpha[adapter_name] = float(alpha_tensor[idx].item())    
                    sub_module.set_scale(adapter_name, 1.0)  # 更新 scale
                    self.alpha_log[f"{module_name}.{adapter_name}"].append(sub_module.lora_alpha[adapter_name])

        if was_training:
            model.train()

def restart_init_train(trainning_args:TrainingArguments,
                       init_steps,model:PeftModel| LoraModel,
                        data_collator,
                        train_dataset,
                        adjust_lora_alpha = True,
                        min_alpha_ratio = 0.8,
                        max_alpha_ratio = 1.6
                        ) -> PeftModel| LoraModel:
    training_arguments0 = deepcopy(trainning_args)
    training_arguments0.num_train_epochs = 0
    training_arguments0.max_steps = init_steps
    training_arguments0.output_dir = os.path.join(training_arguments0.output_dir,"initial_phase")
    training_arguments0.report_to = "none"
    training_arguments0.eval_strategy = "no"
    training_arguments0.save_strategy = "no"
    training_arguments0.load_best_model_at_end = False
    training_arguments0.data_seed = training_arguments0.data_seed* 2 + 1  # to avoid mixing data orders
    training_arguments0.lr_scheduler_type = "constant_with_warmup"
    training_arguments0.logging_steps = 10
    trainer0 = SvdRefactorInitTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = None,
        args = training_arguments0,
        data_collator = data_collator, 
    )
    trainer0.train()
    trainer0.distributed_low_rank_refactor(adjust_lora_alpha=adjust_lora_alpha,min_alpha_ratio=min_alpha_ratio, max_alpha_ratio=max_alpha_ratio)
    import json
    try:
        with open(os.path.join(training_arguments0.output_dir, f"alpha_log_start.json"), "w") as f:
            json.dump(trainer0.alpha_log, f, indent=4)
    except Exception as e:
        print(f"Failed to save alpha log: {e}")
    return model