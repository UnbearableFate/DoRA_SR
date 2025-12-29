from copy import deepcopy
import os
from typing import Optional, Set
from peft import LoraModel, PeftModel
import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

import math
from torch.optim.lr_scheduler import LambdaLR

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


class DistributedSvdRefactorTrainer(Trainer):
    """
    使用分布式低秩 SVD 重构 LoRA 因子，保持 mB A + B mA 的方向不变。
    """

    def __init__(
        self,
        *args,
        refactor_every: int = 100,
        cooldown_steps: int = 0,
        target_adapter_keys: Optional[Set[str]] = None,
        adjust_lora_alpha: bool = True,
        do_refactor: bool = True,
        keep_s: bool = False,
        balance_lambda: float = 0.5,
        variance_ema_decay: float = 0.9,
        basic_alpha: float = 2.0,
        min_alpha_ratio: float = 0.8,
        max_alpha_ratio: float = 1.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.refactor_every = max(1, int(refactor_every))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.target_adapter_keys = set(target_adapter_keys) if target_adapter_keys else None
        self.adjust_lora_alpha = bool(adjust_lora_alpha)
        self.do_refactor = bool(do_refactor)
        self.keep_s = bool(keep_s)
        
        self.balance_lambda = float(balance_lambda)
        self.variance_ema_decay = float(variance_ema_decay)
        self.min_alpha_ratio = float(min_alpha_ratio)
        self.max_alpha_ratio = float(max_alpha_ratio)
        
        self._last_lr_values = None
        self._prev_lr_values = None
        self._lr_restart_last_checked_step = -1
        self.alpha_log = {}
        self._variance_ema = {}
        self.basic_alpha = float(basic_alpha)

    def get_exp_avg(self, param: Tensor) -> Optional[Tensor]:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    return self.optimizer.state.get(p, {}).get("exp_avg", None)
        return None

    @torch.no_grad()
    def distributed_low_rank_refactor(self, do_refactor: bool = True ,adjust_lora_alpha: bool = True, keep_s :bool = False,
                                      min_alpha_ratio: float = 0.8, max_alpha_ratio: float = 1.6):
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

            mB_state = self.get_exp_avg(B)
            mA_state = self.get_exp_avg(A)

            if compute_here:
                # 始终对 B @ A 做 SVD，作为正交基
                base = B @ A
                U, S, Vh = torch.svd_lowrank(base.float(), q=lora_r)
                variance_of_layers[f"{module_name}.{name}"] = float((S ** 2).sum().item())
                if keep_s:
                    # maybe worse result
                    S_bar = S.mean()
                    S_tilde = (1.0 - float(self.balance_lambda)) * S + float(self.balance_lambda) * S_bar
                    S_half = torch.diag(torch.sqrt(torch.clamp(S_tilde, min=0.0)))
                    B_new = (U @ S_half).to(B.dtype)
                    A_new = (S_half @ Vh.t()).to(A.dtype)
                else:
                    B_new = U.to(B.dtype)
                    A_new = Vh.t().to(A.dtype)

                # 计算 T_B^{-1}, T_A^{-1} 以保持 mB A + B mA 的方向
                if mA_state is not None and mB_state is not None and do_refactor:
                    B_pinv = torch.linalg.pinv(B.float())
                    A_pinv = torch.linalg.pinv(A.float())
                    T_B = B_pinv @ B_new.float()  # B_new = B T_B
                    T_A = A_new.float() @ A_pinv  # A_new = T_A A
                    T_B_inv = torch.linalg.pinv(T_B)
                    T_A_inv = torch.linalg.pinv(T_A)
                    mB_new = (
                        mB_state @ T_A_inv.to(mB_state.dtype)
                        if mB_state is not None
                        else torch.zeros_like(B)
                    )
                    mA_new = (
                        T_B_inv.to(mA_state.dtype) @ mA_state
                        if mA_state is not None
                        else torch.zeros_like(A)
                    )
                    mB_state.copy_(mB_new)
                    mA_state.copy_(mA_new)
                
                if do_refactor:
                    B.copy_(B_new)
                    A.copy_(A_new)

            if is_dist and do_refactor:
                broadcast_works.append(
                    dist.broadcast(B, src=idx % world_size, async_op=True).get_future()
                )
                broadcast_works.append(
                    dist.broadcast(A, src=idx % world_size, async_op=True).get_future()
                )
                if mB_state is not None:
                    broadcast_works.append(
                        dist.broadcast(mB_state, src=idx % world_size, async_op=True).get_future()
                    )
                if mA_state is not None:
                    broadcast_works.append(
                        dist.broadcast(mA_state, src=idx % world_size, async_op=True).get_future()
                    )

        if is_dist and broadcast_works and do_refactor:
            torch.futures.wait_all(broadcast_works)

        if is_dist:
            gathered_variances = [None] * world_size
            dist.all_gather_object(gathered_variances, variance_of_layers)
            variance_of_layers = {}
            for part in gathered_variances:
                variance_of_layers.update(part)
        

        if variance_of_layers:
            decay = self.variance_ema_decay
            for layer_key, layer_var in variance_of_layers.items():
                if layer_key in self._variance_ema:
                    self._variance_ema[layer_key] = (
                        self._variance_ema[layer_key] * decay + layer_var * (1.0 - decay)
                    )
                else:
                    self._variance_ema[layer_key] = layer_var
            variance_of_layers = dict(self._variance_ema)

        beta_pos, beta_neg, r_low_used, r_high_used = infer_betas_from_ratios(variance_of_layers.values(), min_alpha_ratio, max_alpha_ratio)

        if adjust_lora_alpha and variance_of_layers:
            if rank == 0:
                avg_of_global_variance = sum(variance_of_layers.values()) / len(variance_of_layers)
                #clip_ratio = self.alpha_clip_ratio
                #beta = self.alpha_beta #* (self.state.max_steps - self.state.global_step) / self.state.max_steps
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
                            sub_module.lora_alpha[adapter_name] = ratio_new * self.basic_alpha

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

    def _is_lr_restart(self):
        if not hasattr(self, "lr_scheduler") or self.lr_scheduler is None:
            return False

        step = self.state.global_step
        scheduler_step = getattr(self.lr_scheduler, "last_epoch", step)
        # Avoid double-processing the same optimizer step when using gradient accumulation
        if self._lr_restart_last_checked_step == step:
            return False
        self._lr_restart_last_checked_step = step

        if not hasattr(self.lr_scheduler, "get_last_lr"):
            return False

        current_lrs = list(self.lr_scheduler.get_last_lr())
        if self._last_lr_values is None:
            self._last_lr_values = current_lrs
            self._prev_lr_values = None
            return False

        # Prefer using LambdaLR's lambda to detect a true "restart" boundary for
        # get_warmup_restart_then_final_decay_scheduler_ratio():
        # a restart is when the schedule stops decreasing and starts increasing again.
        eps = 1e-12
        try:
            from torch.optim.lr_scheduler import LambdaLR

            if isinstance(self.lr_scheduler, LambdaLR) and getattr(self.lr_scheduler, "lr_lambdas", None):
                if scheduler_step >= 2:
                    is_restart = False
                    for lr_lambda in self.lr_scheduler.lr_lambdas:
                        r2 = float(lr_lambda(int(scheduler_step - 2)))
                        r1 = float(lr_lambda(int(scheduler_step - 1)))
                        r0 = float(lr_lambda(int(scheduler_step)))
                        delta_prev = r1 - r2
                        delta_cur = r0 - r1
                        if (delta_cur > eps) and (delta_prev <= eps):
                            is_restart = True
                            break
                    self._prev_lr_values = self._last_lr_values
                    self._last_lr_values = current_lrs
                    return is_restart
        except Exception:
            pass

        # Fallback: detect the point where LR stops decreasing and starts increasing.
        if self._prev_lr_values is None:
            self._prev_lr_values = self._last_lr_values
            self._last_lr_values = current_lrs
            return False

        is_restart = any(
            ((cur_lr - prev_lr) > eps) and ((prev_lr - prev2_lr) <= eps)
            for cur_lr, prev_lr, prev2_lr in zip(current_lrs, self._last_lr_values, self._prev_lr_values)
        )
        self._prev_lr_values = self._last_lr_values
        self._last_lr_values = current_lrs
        return is_restart

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        step = self.state.global_step

        if self.optimizer is None:
            return loss

        """
        if step > self.state.max_steps - self.cooldown_steps:
            return loss
        if step < self.refactor_every or (step % self.refactor_every) != 0:
            return loss
        """
        if self._is_lr_restart():
            if self.accelerator.process_index == 0:
                print(f"Step {step}: Detected LR restart, performing distributed low-rank refactor...")
            self.distributed_low_rank_refactor(
                do_refactor = self.do_refactor,
                adjust_lora_alpha = self.adjust_lora_alpha,
                keep_s = self.keep_s,
                min_alpha_ratio = self.min_alpha_ratio,
                max_alpha_ratio = self.max_alpha_ratio,
                )
        
        return loss

    def save_alpha_log(self, filepath: str):
        if not self.alpha_log or self.accelerator.process_index != 0:
            return
        import json
        with open(filepath, "w") as f:
            json.dump(self.alpha_log, f, indent=4)

def get_warmup_restart_then_final_decay_scheduler_ratio(
    optimizer,
    num_training_steps,
    repeat_n,
    repeat_warmup_ratio,
    repeat_decay_ratio,
    repeat_end_lr_rate,
    final_warmup_ratio,
    min_lr_rate,
    repeat_decay_type="cosine",
    final_decay_type="cosine",
    warmup_start_lr_rate=0.0,
    first_warmup_start_lr_rate=0.0,
    last_epoch=-1,
):

    T = num_training_steps

    repeat_warmup_steps = int(round(repeat_warmup_ratio * T))
    repeat_decay_steps  = int(round(repeat_decay_ratio  * T))
    final_warmup_steps  = int(round(final_warmup_ratio  * T))

    cycle_len = repeat_warmup_steps + repeat_decay_steps
    repeat_total_steps = repeat_n * cycle_len

    def _decay_factor(t, kind):
        t = min(max(t, 0.0), 1.0)
        if kind == "linear":
            return 1.0 - t
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def lr_lambda(step):
        step = max(0, min(step, T))

        # repeated phase
        if step < repeat_total_steps:
            if step < repeat_warmup_steps : # first warmup
                if repeat_warmup_steps == 0:
                    return 1.0
                t = step / repeat_warmup_steps
                return first_warmup_start_lr_rate + (1.0 - first_warmup_start_lr_rate) * t

            pos = step % cycle_len

            if pos < repeat_warmup_steps:
                if repeat_warmup_steps == 0:
                    return 1.0
                t = pos / repeat_warmup_steps
                return warmup_start_lr_rate + (1.0 - warmup_start_lr_rate) * t

            dpos = pos - repeat_warmup_steps
            if repeat_decay_steps == 0:
                return repeat_end_lr_rate
            t = dpos / repeat_decay_steps
            f = _decay_factor(t, repeat_decay_type)
            return repeat_end_lr_rate + (1.0 - repeat_end_lr_rate) * f

        # final phase
        final_pos = step - repeat_total_steps
        final_total = T - repeat_total_steps

        if final_pos < final_warmup_steps:
            if final_warmup_steps == 0:
                return 1.0
            t = final_pos / final_warmup_steps
            return warmup_start_lr_rate + (1.0 - warmup_start_lr_rate) * t

        decay_left = final_total - final_warmup_steps
        if decay_left <= 0:
            return min_lr_rate

        dpos = final_pos - final_warmup_steps
        t = dpos / decay_left
        f = _decay_factor(t, final_decay_type)
        return min_lr_rate + (1.0 - min_lr_rate) * f

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def restart_init_train(trainning_args:TrainingArguments,
                       init_steps,model:PeftModel| LoraModel,
                        data_collator,
                        train_dataset,
                        adjust_lora_alpha=True,
                        basic_alpha=2.0,
                        min_alpha_ratio = 0.8,
                        max_alpha_ratio = 1.6) -> PeftModel| LoraModel:
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
    trainer0 = DistributedSvdRefactorTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = None,
        args = training_arguments0,
        data_collator = data_collator,
        basic_alpha=basic_alpha,
    )
    trainer0.train()
    trainer0.distributed_low_rank_refactor(adjust_lora_alpha=adjust_lora_alpha, do_refactor=True, keep_s=False, min_alpha_ratio=min_alpha_ratio, max_alpha_ratio=max_alpha_ratio)
    trainer0.save_alpha_log(os.path.join(training_arguments0.output_dir,"lora_alpha_log.json"))
    return model