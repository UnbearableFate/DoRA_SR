import math
import os
from typing import Dict, Iterator, Tuple, Optional, Set
from peft import LoraConfig, PeftModel ,get_peft_model ,LoraModel
import torch
from torch import nn
from transformers import Trainer
from torch import Tensor
from collections import Counter
import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

class NewTrainer(Trainer):
    def __init__(self, *args, rebuid_lora:bool = False , **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuid_lora = rebuid_lora

    @torch.no_grad()
    def init_lora_weight(
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
                    s_mean = S.mean()
                    s_half = torch.sqrt(s_mean) * torch.ones_like(S)
                    diag_s = torch.diag(s_half)
                    # 重置 A = V^T, B = U
                    A.copy_((diag_s@Vh.t()).to(A.dtype))
                    B.copy_((U@diag_s).to(B.dtype))

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
                print(f"Updated alpha for {module_name}.{name} to {sub_module.lora_alpha[name]:.4f} at rank {rank}")

    def lora_rank_distribution_and_init_weight(self, target_rank: int) -> None:
        """
        Compute SVD-based rank distribution following EVA strategy, then reinitialize LoRA weights.
        
        Args:
            target_rank: The desired rank per layer. Total rank budget = num_layers * target_rank.
                        Note: LoRA should be initialized with 2*target_rank per layer before calling this.
        """
        # Step 1: Compute SVD for all layers and collect explained variance ratios
        svd_results = {}  # {module_name.name: (U, S, Vh)}
        exp_vars = {}     # {module_name.name: explained_variance_ratio}
        num_layers = 0
        original_alpha = None
        
        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                for name in sub_module.lora_A.keys():
                    num_layers += 1
                    A = sub_module.lora_A[name].weight.data
                    B = sub_module.lora_B[name].weight.data
                    current_rank = A.shape[0]
                    
                    # Compute low-rank SVD on the current LoRA weight for efficiency
                    temp_weight = B @ A
                    U, S, Vh = torch.svd_lowrank(temp_weight.float(), q=current_rank)
                    
                    # Store SVD results (Vh from svd_lowrank is already transposed)
                    layer_key = f"{module_name}.{name}"
                    svd_results[layer_key] = (U, S, Vh.t())  # transpose Vh to match linalg.svd format
                    
                    # Compute explained variance ratio
                    S_squared = S ** 2
                    total_variance = S_squared.sum()
                    explained_variance_ratio = S_squared / (total_variance + 1e-10)
                    exp_vars[layer_key] = explained_variance_ratio[:current_rank]

                    if original_alpha is None and hasattr(sub_module, "lora_alpha"):
                        original_alpha = float(sub_module.lora_alpha[name])
                    
        
        # Step 2: Apply EVA's rank distribution strategy
        # Total rank budget = num_layers * target_rank
        total_rank_budget = num_layers * target_rank
        
        # Collect all (layer_key, explained_variance) pairs for all components
        keys_values = [(k, c) for k, evr in exp_vars.items() for c in evr]
        keys, values = zip(*keys_values)
        values_tensor = torch.stack(values)
        
        # Sort all components by explained variance (descending)
        idx = values_tensor.argsort(descending=True)
        top_count = min(total_rank_budget, values_tensor.numel())
        top_indices = idx[:top_count]

        print("top_count", top_indices)
        
        # Select top components and count how many each layer gets
        selected_keys = [keys[i] for i in top_indices]
        rank_distribution = Counter(selected_keys)
        
        # Track explained variance share per layer for alpha redistribution
        layer_alpha_contrib = {k: 0.0 for k in exp_vars.keys()}
        temp = {}
        for i in top_indices:
            layer_alpha_contrib[keys[i]] += float(values_tensor[i])
            if keys[i] not in temp:
                temp[keys[i]] = 1
            else:
                temp[keys[i]] += 1
        sum_of_top_evr = float(values_tensor[top_indices].sum()) if len(top_indices) > 0 else 0.0
        print("layer_alpha_contrib", layer_alpha_contrib)
        print("temp", temp)
        # Ensure all layers are in the distribution (some may get 0 rank)
        all_layer_keys = list(exp_vars.keys())
        rank_distribution = {k: rank_distribution.get(k, 0) for k in all_layer_keys}
        
        if original_alpha is None:
            original_alpha = 1.0
        total_alpha_budget = num_layers * original_alpha
        alpha_denominator = sum_of_top_evr if sum_of_top_evr > 0 else None
        
        # Step 3: Reinitialize LoRA weights with the new rank distribution
        rank_pattern = {}
        alpha_pattern = {}
        lora_A = {}
        lora_B = {}
        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                for name in sub_module.lora_A.keys():
                    layer_key = f"{module_name}.{name}"
                    new_rank = rank_distribution[layer_key]
                    
                    if new_rank == 0:
                        # Set to zero if no rank allocated
                        sub_module.lora_A[name].weight.data.zero_()
                        sub_module.lora_B[name].weight.data.zero_()
                        if self.accelerator.is_main_process:
                            logger.warning(f"Layer {layer_key} assigned rank 0, weights zeroed out")
                        continue
                    
                    # Get SVD results
                    U, S, Vh = svd_results[layer_key]
                    A = sub_module.lora_A[name].weight.data
                    B = sub_module.lora_B[name].weight.data
                    current_rank = A.shape[0]
                    
                    if new_rank > current_rank:
                        if self.accelerator.is_main_process:
                            logger.warning(f"Layer {layer_key}: new_rank ({new_rank}) > current_rank ({current_rank}), "
                                         f"using current_rank instead")
                        new_rank = current_rank
                    
                    if self.rebuid_lora:
                        if name == "default":
                            if module_name.startswith("base_model"):
                                key = module_name[17:]  # remove "base_model." prefix
                            else:
                                key = module_name
                            if new_rank < 8 :
                                new_rank = 8
                            rank_pattern[key] = new_rank
                            if alpha_denominator is not None and total_alpha_budget > 0:
                                layer_share = layer_alpha_contrib.get(layer_key, 0.0)
                                alpha_value = (layer_share / alpha_denominator) * total_alpha_budget * math.sqrt (new_rank / max(target_rank, 1))
                            else:
                                alpha_value = (new_rank / max(target_rank, 1)) * original_alpha
                            alpha_pattern[key] = alpha_value
                            lora_A[key] = Vh[:new_rank].to(A.dtype).clone().detach()
                            lora_B[key] = U[:, :new_rank].to(B.dtype).clone().detach()
                        continue
                    
                    # Truncate U, Vh to new_rank and reinitialize
                    # A: [r, in_features], B: [out_features, r]
                    A[:new_rank].copy_(Vh[:new_rank].to(A.dtype))
                    B[:, :new_rank].copy_(U[:, :new_rank].to(B.dtype))
                    
                    # Zero out the unused ranks
                    if new_rank < current_rank:
                        A[new_rank:].zero_()
                        B[:, new_rank:].zero_()
                    
                    if self.accelerator.is_main_process:
                        logger.info(f"Layer {module_name}: reinitialized with rank {new_rank}/{current_rank}")

        if self.rebuid_lora:
            return rank_pattern, alpha_pattern ,lora_A, lora_B
        else:
            return None ,None, None, None 
    
    def reinit_lora_weights_without_rank_change(self) -> None:
        """
        Reinitialize existing LoRA weights via SVD and redistribute lora_alpha without changing ranks.
        """
        svd_results = {}
        exp_vars = {}
        layer_ranks = {}
        num_layers = 0
        original_alpha = None

        def _get_alpha_value(module, adapter_name):
            alpha_attr = getattr(module, "lora_alpha", None)
            if alpha_attr is None:
                return None
            if isinstance(alpha_attr, dict):
                value = alpha_attr.get(adapter_name, None)
            else:
                value = alpha_attr
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                for name in sub_module.lora_A.keys():
                    num_layers += 1
                    A = sub_module.lora_A[name].weight.data
                    B = sub_module.lora_B[name].weight.data
                    current_rank = A.shape[0]
                    layer_key = f"{module_name}.{name}"
                    layer_ranks[layer_key] = current_rank

                    temp_weight = B @ A
                    U, S, Vh = torch.svd_lowrank(temp_weight.float(), q=current_rank)
                    svd_results[layer_key] = (U, S, Vh.t())

                    S_squared = S ** 2
                    total_variance = S_squared.sum()
                    explained_variance_ratio = S_squared / (total_variance + 1e-10)
                    exp_vars[layer_key] = explained_variance_ratio[:current_rank]

                    if original_alpha is None:
                        alpha_val = _get_alpha_value(sub_module, name)
                        if alpha_val is not None:
                            original_alpha = alpha_val

        if len(exp_vars) == 0:
            logger.warning("No LoRA layers found for SVD-based reinitialization.")
            return

        keys_values = [(k, c) for k, evr in exp_vars.items() for c in evr]
        if not keys_values:
            logger.warning("Explained variance data is empty; skipping LoRA reinitialization.")
            return
        keys, values = zip(*keys_values)
        values_tensor = torch.stack(values)

        total_rank_budget = sum(layer_ranks.values())
        if total_rank_budget == 0:
            logger.warning("Total rank budget is zero; skipping LoRA reinitialization.")
            return
        idx = values_tensor.argsort(descending=True)
        top_count = min(total_rank_budget, values_tensor.numel())
        top_indices = idx[:top_count]

        layer_alpha_contrib = {k: 0.0 for k in exp_vars.keys()}
        for i in top_indices:
            layer_alpha_contrib[keys[i]] += float(values_tensor[i])
        sum_of_top_evr = float(values_tensor[top_indices].sum().item()) if top_indices.numel() > 0 else 0.0

        if original_alpha is None:
            original_alpha = 1.0
        total_alpha_budget = num_layers * original_alpha
        alpha_denominator = sum_of_top_evr if sum_of_top_evr > 0 else None

        def _set_alpha(module, adapter_name, new_alpha, rank):
            alpha_attr = getattr(module, "lora_alpha", None)
            scaling_attr = getattr(module, "scaling", None)

            if isinstance(alpha_attr, dict):
                alpha_attr[adapter_name] = new_alpha
            elif alpha_attr is not None:
                setattr(module, "lora_alpha", new_alpha)

            scaling_value = new_alpha / max(rank, 1)
            if isinstance(scaling_attr, dict):
                scaling_attr[adapter_name] = scaling_value
            elif scaling_attr is not None:
                setattr(module, "scaling", scaling_value)

        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                for name in sub_module.lora_A.keys():
                    layer_key = f"{module_name}.{name}"
                    if layer_key not in svd_results:
                        continue
                    U, S, Vh = svd_results[layer_key]
                    A = sub_module.lora_A[name].weight.data
                    B = sub_module.lora_B[name].weight.data
                    rank = layer_ranks[layer_key]

                    A[:rank].copy_(Vh[:rank].to(A.dtype))
                    B[:, :rank].copy_(U[:, :rank].to(B.dtype))

                    if alpha_denominator is not None and total_alpha_budget > 0:
                        layer_share = layer_alpha_contrib.get(layer_key, 0.0)
                        alpha_value = (layer_share / alpha_denominator) * total_alpha_budget
                    else:
                        alpha_value = original_alpha
                    _set_alpha(sub_module, name, alpha_value, rank)

        if self.accelerator.is_main_process:
            logger.info("Completed SVD-based LoRA weight and alpha reinitialization without rank redistribution.")

    @staticmethod
    def rebuid_lora_weights(
        model: LoraModel | PeftModel,
        rank_pattern: Dict[str, int],
        alpha_pattern: Dict[str, float],
        lora_A: Dict[str, Tensor],
        lora_B: Dict[str, Tensor],
        lora_config: LoraConfig,
    ) -> nn.Module:
        model = model.unload()
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")
        lora_config.rank_pattern = rank_pattern
        lora_config.alpha_pattern = alpha_pattern
        lora_config.init_lora_weights = True
        model = get_peft_model(model, lora_config)
        for module_name, sub_module in model.named_modules():
            key = module_name[17:] if module_name.startswith("base_model") else module_name
            if key in rank_pattern:
                for name in sub_module.lora_A.keys():
                    A = sub_module.lora_A[name].weight.data
                    B = sub_module.lora_B[name].weight.data
                    # Ensure tensors are on the same device
                    A.copy_(lora_A[key].to(A.device))
                    B.copy_(lora_B[key].to(B.device))
        return model