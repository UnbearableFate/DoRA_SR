import math
from typing import Dict, Iterator, Tuple, Optional, Set
from peft import LoraConfig, PeftModel ,get_peft_model ,LoraModel
import torch
from torch import nn
from transformers import Trainer
from torch import Tensor
from collections import Counter
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

class RestartInitTrainer(Trainer):
    def __init__(self, *args, rebuid_lora:bool = False , **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuid_lora = rebuid_lora

    def init_lora_weight(
        self,
    ) -> Iterator[Tuple[nn.Module, str]]:
        for module_name, sub_module in self.model.named_modules():
            if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B"):
                for name in sub_module.lora_A.keys():
                    A = sub_module.lora_A[name].weight.data
                    B = sub_module.lora_B[name].weight.data
                    lora_r = A.shape[0]
                    temp_weight = B @ A
                    U, S, Vh = torch.linalg.svd(temp_weight.float(), full_matrices=False)   # U:[d_out,min(d_out,d_in)], S:[min(d_out,d_in)], Vh:[min(d_out,d_in),d_in]
                    A.copy_(Vh[:lora_r].to(A.dtype))  # Vh[:lora_r] has shape [lora_r, d_in], matching A's shape
                    B.copy_(U[:,:lora_r].to(B.dtype))
    
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

        # Select top components and count how many each layer gets
        selected_keys = [keys[i] for i in top_indices]
        rank_distribution = Counter(selected_keys)
        
        # Ensure all layers are in the distribution (some may get 0 rank)
        all_layer_keys = list(exp_vars.keys())
        rank_distribution = {k: rank_distribution.get(k, 0) for k in all_layer_keys}
        
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