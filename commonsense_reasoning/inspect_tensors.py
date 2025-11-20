import sys
from safetensors.torch import load_file
import torch

def inspect_safetensors_file(state_dict):
    for key, value in state_dict.items():
        print(f"Key: {key}")
        if isinstance(value, torch.Tensor):
            print(f" - Shape: {value.shape}")
            print(f" - Dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f" - Length: {len(value)}")
            if len(value) > 0:
                print(f" - First element type: {type(value[0])}")
        elif isinstance(value, dict):
            inspect_safetensors_file(value)  # Recursive call for nested dicts
        else:
            print(f" - Type: {type(value)}")

if __name__ == "__main__":
    # The user wants to inspect a specific file.
    file_to_inspect = "outputs/test/Qwen3-1.7B/R32/Qwen3-1.7B_r32_alpha64_True_rslora_normal_seed42/checkpoint-40/adapter_model.safetensors"
    state_dict = load_file(file_to_inspect, device="cpu")
    inspect_safetensors_file(state_dict)