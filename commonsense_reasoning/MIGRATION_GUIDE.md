# Migration Guide: Updating Training Scripts for BF16 Stability

## Quick Start

If you have existing training scripts that are experiencing bf16 instability, update your command-line arguments:

### Before (Unstable)
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --adapter_name="dora" \
    --lora_r=32 \
    --lora_alpha=64 \
    --bf16=True
```

### After (Stable)
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --adapter_name="dora" \
    --lora_r=32 \
    --lora_alpha=64 \
    --bf16=True \
    --attn_implementation="eager" \
    --disable_cudnn_sdpa=True \
    --enable_torch_compile=False
```

**Note**: The new parameters have safe defaults, so if you don't specify them, the script will use stable settings automatically.

## Parameter Reference

### New Parameters (All Optional with Safe Defaults)

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `--attn_implementation` | `"eager"` | Attention backend to use | Use `"flash_attention_2"` after CUDA/cuDNN upgrade |
| `--disable_cudnn_sdpa` | `True` | Disable cuDNN SDPA kernels | Set to `False` with upgraded stack |
| `--disable_flash_sdpa` | `False` | Disable Flash SDPA kernels | Set to `True` if still seeing NaNs |
| `--enable_torch_compile` | `False` | Enable torch.compile optimization | Set to `True` after confirming stability |

### Updated Behavior (No Action Needed)

These changes happen automatically:

1. **Model dtype**: Always loads in fp32 (was: conditional bf16)
2. **device_map**: Always `None` for training (was: `"auto"`)
3. **PAD token**: Automatically adds proper PAD for Llama-3.x
4. **torch.compile**: Disabled by default (was: always enabled)

## Existing Scripts Compatibility

### Shell Scripts (e.g., llama3_8B_DoRA.sh)

**No changes required!** All new parameters have safe defaults.

However, you can explicitly set them for clarity:

```bash
#!/bin/bash

accelerate launch finetune.py \
    --base_model "meta-llama/Llama-3.1-8B" \
    --data_path "./commonsense_170k.json" \
    --output_dir "./output" \
    --adapter_name "dora" \
    --lora_r 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --batch_size 128 \
    --per_device_train_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 3e-5 \
    --bf16 True \
    --attn_implementation "eager" \
    --disable_cudnn_sdpa True \
    --enable_torch_compile False \
    --seed 42
```

### Python Scripts

If calling `train()` directly from Python:

```python
from finetune import train

# Stable configuration
train(
    base_model="meta-llama/Llama-3.1-8B",
    adapter_name="dora",
    lora_r=32,
    lora_alpha=64,
    bf16=True,
    # New parameters (optional, these are the defaults)
    attn_implementation="eager",
    disable_cudnn_sdpa=True,
    enable_torch_compile=False,
)
```

## Upgrade Path

### Stage 1: Initial Stabilization (Now)
```bash
--attn_implementation="eager"
--disable_cudnn_sdpa=True
--enable_torch_compile=False
```
**Expected**: Stable training, slower than optimal

### Stage 2: After CUDA/cuDNN Upgrade
```bash
--attn_implementation="flash_attention_2"
--disable_cudnn_sdpa=False
--enable_torch_compile=False
```
**Expected**: Faster training, test for stability

### Stage 3: Full Optimization
```bash
--attn_implementation="flash_attention_2"
--disable_cudnn_sdpa=False
--enable_torch_compile=True
```
**Expected**: Maximum speed, test for stability

## Troubleshooting

### Still seeing NaN at step 0?

1. **Disable all fused attention:**
   ```bash
   --disable_flash_sdpa=True
   ```

2. **Verify parameters are loaded correctly:**
   Check the training output for:
   ```
   attn_implementation: eager
   disable_cudnn_sdpa: True
   ```

3. **Test with fp16 instead:**
   ```bash
   --bf16=False --fp16=True
   ```

### Training is very slow?

This is expected with `eager` attention. To speed up:

1. **Upgrade CUDA/cuDNN stack** to latest versions
2. **Switch to Flash Attention 2:**
   ```bash
   --attn_implementation="flash_attention_2"
   --disable_cudnn_sdpa=False
   ```
3. **Monitor first 100 steps** - if stable, enable compile:
   ```bash
   --enable_torch_compile=True
   ```

### Need to test stability quickly?

Run a short diagnostic:
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./commonsense_170k.json" \
    --adapter_name="dora" \
    --num_epochs=1 \
    --batch_size=16 \
    --per_device_train_batch_size=2 \
    --eval_step=10 \
    --save_step=999999 \
    --bf16=True
```

Check the first 10 steps for:
- ✅ Finite loss values (not NaN or 0.0)
- ✅ Reasonable grad_norm (0.1-10.0)
- ✅ Loss decreasing over time

## Key Behavioral Changes

### 1. Model Loading
**Before:**
```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float32,
    device_map="auto",  # ❌ Not for training
)
```

**After:**
```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # ✅ fp32 master weights
    device_map=None,            # ✅ Proper for training
    attn_implementation="eager", # ✅ Stable backend
)
```

### 2. Tokenizer Setup
**Before:**
```python
tokenizer.pad_token_id = 0  # ❌ Token 0 is "!" in Llama-3
```

**After:**
```python
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
    # ✅ Proper PAD token
```

### 3. torch.compile
**Before:**
```python
if torch.__version__ >= "2":
    model = torch.compile(model)  # ❌ Always enabled
```

**After:**
```python
if enable_torch_compile and torch.__version__ >= "2":
    model = torch.compile(model)  # ✅ Opt-in after stability
```

## Example Updated Scripts

### Single GPU
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./commonsense_170k.json" \
    --adapter_name="dora" \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --lora_r=32 \
    --lora_alpha=64 \
    --batch_size=128 \
    --per_device_train_batch_size=2 \
    --bf16=True
```

### Multi-GPU (Accelerate)
```bash
accelerate launch --config_file accelerate_config.yaml finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./commonsense_170k.json" \
    --adapter_name="dora" \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --lora_r=32 \
    --lora_alpha=64 \
    --batch_size=128 \
    --per_device_train_batch_size=2 \
    --bf16=True
```

### With Performance Optimization (After Confirming Stability)
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./commonsense_170k.json" \
    --adapter_name="dora" \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --lora_r=32 \
    --lora_alpha=64 \
    --batch_size=128 \
    --per_device_train_batch_size=2 \
    --bf16=True \
    --attn_implementation="flash_attention_2" \
    --disable_cudnn_sdpa=False \
    --enable_torch_compile=True
```

## FAQ

**Q: Do I need to modify my existing shell scripts?**  
A: No, all new parameters have safe defaults. Scripts will work without changes.

**Q: Will training be slower?**  
A: Yes, `eager` attention is ~2x slower than Flash Attention 2. But it's stable. Upgrade your CUDA/cuDNN stack to use FA2.

**Q: Can I use the old behavior?**  
A: Not recommended, but you can override:
```bash
--attn_implementation="sdpa" --disable_cudnn_sdpa=False
```

**Q: How do I know which CUDA/cuDNN version has the fix?**  
A: Check NVIDIA's cuDNN release notes. Generally, CUDA 12.4+ with cuDNN 9.1+ should be stable.

**Q: Does this affect LoRA too or just DoRA?**  
A: Both. The instability is in the attention mechanism, not the adapter type.

**Q: What about other models (Qwen, Mistral, etc.)?**  
A: Same fixes apply. The PAD token fix is Llama-3 specific, but won't hurt other models.
