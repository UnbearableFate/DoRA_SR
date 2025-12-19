from typing import Optional, Set
import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import Trainer


def iter_lora_factors_with_names(model: nn.Module,
                                 target_adapter_keys: Optional[Set[str]] = None):
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for name in module.lora_A.keys():
                if target_adapter_keys and name not in target_adapter_keys:
                    continue
                yield module_name, name, module.lora_B[name].weight, module.lora_A[name].weight

"""
use this trainer with lr restart
"""
class DistributedSvdRefactorRestartTrainer(Trainer):

    def __init__(
        self,
        *args,
        warmup_mode = False,
        restart_mode = False,
        refactor_every_step = 101,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._last_lr_values = None
        self._lr_restart_last_checked_step = -1
        self.warmup_mode = warmup_mode
        self.refactor_every_step = refactor_every_step

    def get_exp_avg(self, param: Tensor) -> Optional[Tensor]:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    return self.optimizer.state.get(p, {}).get("exp_avg", None)
        return None

    def _get_param_state(self, param: torch.Tensor) -> Optional[dict]:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    return self.optimizer.state.setdefault(p, {})
        return None

    def _zero_state_tensor(self, param: torch.Tensor, key: str) -> None:
        state = self._get_param_state(param)
        if not state:
            return
        tens = state.get(key, None)
        if tens is None:
            return
        tens.zero_()
    
    def _reset_adam_state(self, param: torch.Tensor) -> None:
        self._zero_state_tensor(param, "exp_avg")
        self._zero_state_tensor(param, "exp_avg_sq")

    @torch.no_grad()
    def _distributed_low_rank_refactor(
        self,
        adjust_alpha: bool = True,
        beta: float = 0.25,                      # 新增：归一化强度（0~1）
        clip_ratio: float = 1.3,               # 新增：clipping；例如 4 表示 α 变化限制在[1/4, 4]
    ) -> None:

        is_dist = self.accelerator.num_processes > 1 
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        variance_of_layers = {}
        device_for_broadcast = None
        wait_list = []

        # === Step 1: 计算每层的奇异值能量 (variance) 并做 SVD 重置 U,V ===
        for idx ,(module_name, sub_module) in enumerate(self.model.named_modules()):
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                name = "default"
                assert name in sub_module.lora_A.keys(), \
                    f"LoRA module {module_name} missing 'default' key"

                A = sub_module.lora_A[name].weight.data
                B = sub_module.lora_B[name].weight.data
                lora_r = A.shape[0]
                if device_for_broadcast is None:
                    device_for_broadcast = A.device
                
                if rank == idx % world_size:
                    temp_weight = (B @ A).float()

                    # SVD decomposition
                    U, S, Vh = torch.svd_lowrank(temp_weight, q=lora_r)

                    # 奇异值平方作为能量
                    S_squared = S ** 2
                    variance_of_layers[module_name] = S_squared.sum().item()

                    # 重置 A = V^T, B = U
                    A.copy_(Vh.t().to(A.dtype))
                    B.copy_(U.to(B.dtype))

                # 将更新后的 A、B 从 rank 0 广播到所有进程
                if is_dist:
                    wait_list.append(dist.broadcast(A, src=idx % world_size,async_op=True).get_future())
                    wait_list.append(dist.broadcast(B, src=idx % world_size,async_op=True).get_future())
        
        if is_dist:
            data = [None] * self.accelerator.num_processes
            dist.all_gather_object(data, variance_of_layers)
            # 合并所有进程的 variance 结果
            variance_of_layers = {}
            for part in data:
                variance_of_layers.update(part)
        
        if not adjust_alpha:
            if is_dist:
                torch.futures.wait_all(wait_list)
            return
        
        # 先在 rank 0 上完成所有层 alpha 的更新
        if rank == 0:
            avg_of_global_variance = sum(variance_of_layers.values()) / len(variance_of_layers)
            for module_name, sub_module in self.model.named_modules():
                if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                    name = "default"
                    assert name in sub_module.lora_A.keys(), \
                        f"LoRA module {module_name} missing 'default' key"

                    if hasattr(sub_module, "lora_alpha"):
                        # ---（1）计算能量比例 ---
                        layer_var = variance_of_layers[module_name]
                        ratio = layer_var / avg_of_global_variance

                        # ---（2）加入指数系数 beta ---
                        ratio_new = ratio ** beta

                        # ---（3）加入 clipping，避免极端缩放 ---
                        if clip_ratio is not None and clip_ratio > 0:
                            ratio_new = max(1.0 / clip_ratio, min(ratio_new, clip_ratio))

                        # ---（4）更新 α ---
                        sub_module.lora_alpha[name] *= ratio_new

        # 收集所有层的 alpha 到一个向量里, 一次性 broadcast
        alpha_values: list[float] = []
        alpha_indices: list[tuple[str, str]] = []
        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                name = "default"
                if name not in sub_module.lora_alpha:
                    continue
                alpha_indices.append((module_name, name))
                if rank == 0:
                    alpha_values.append(float(sub_module.lora_alpha[name]))
                else:
                    alpha_values.append(0.0)
        
        torch.futures.wait_all(wait_list)

        if alpha_values:
            if device_for_broadcast is None:
                device_for_broadcast = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            alpha_tensor = torch.tensor(alpha_values, device=device_for_broadcast, dtype=torch.float32)
            if is_dist:
                dist.broadcast(alpha_tensor, src=0)

            # 将广播后的向量按顺序写回各层
            for idx, (module_name, name) in enumerate(alpha_indices):
                sub_module = dict(self.model.named_modules())[module_name]
                sub_module.lora_alpha[name] = float(alpha_tensor[idx].item())
                
    def _is_lr_restart(self):
        if not hasattr(self, "lr_scheduler") or self.lr_scheduler is None:
            return False

        step = self.state.global_step
        # Avoid double-processing the same optimizer step when using gradient accumulation
        if self._lr_restart_last_checked_step == step:
            return False
        self._lr_restart_last_checked_step = step

        if not hasattr(self.lr_scheduler, "get_last_lr"):
            return False

        try:
            warmup_steps = self.args.get_warmup_steps(self.state.max_steps)
        except Exception:
            warmup_steps = getattr(self.args, "warmup_steps", 0) or 0

        current_lrs = list(self.lr_scheduler.get_last_lr())
        if self._last_lr_values is None:
            self._last_lr_values = current_lrs
            return False

        # After warmup, cosine with hard restarts is strictly decreasing inside a cycle.
        # Any lr increase indicates a restart just happened at the previous step.
        is_restart = step > warmup_steps and any(
            cur_lr > prev_lr * (1.0 + 1e-12) for cur_lr, prev_lr in zip(current_lrs, self._last_lr_values)
        )
        self._last_lr_values = current_lrs
        return is_restart
    
    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.warmup_mode or not self._is_lr_restart():
            return loss

        print(f"LR restart detected at step {self.state.global_step}, performing distributed low-rank refactor...")
        self._distributed_low_rank_refactor(adjust_alpha=False, beta=0.25, clip_ratio=1.3)

        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                name = "default"
                self._reset_adam_state(sub_module.lora_A[name].weight)
                self._reset_adam_state(sub_module.lora_B[name].weight)

        return loss