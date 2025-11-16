# BF16 Training Stability Fixes

## Summary of Changes

This document explains the modifications made to `finetune.py` to resolve bf16 training instability issues, particularly the "step 0 NaN/0.0 loss" problem on Hopper GPUs (GH200).

## Root Cause

The instability was caused by:
1. **Fused SDPA/Flash attention kernels** overflowing in bf16 before any parameter updates
2. **Hard-casting the entire model to bf16**, reducing numerical headroom
3. **cuDNN SDPA regressions** on Ampere/Hopper/Blackwell architectures
4. **Missing PAD token** in Llama-3.x tokenizers (token 0 is "!" not PAD)
5. **device_map="auto"** being used for training (inference-only feature)

## Fixes Applied

### 1. Safe Attention Backend (Critical)
```python
# New parameters
attn_implementation: str = "eager"  # Default to stable backend
disable_cudnn_sdpa: bool = True     # Disable problematic cuDNN SDPA
```

- Uses `attn_implementation="eager"` which avoids fused softmax in bf16
- The "math" backend in eager mode keeps bf16 intermediates in float32
- After upgrading CUDA/cuDNN stack, you can try `"flash_attention_2"`

**References:**
- https://huggingface.co/docs/transformers/en/attention_interface
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

### 2. FP32 Master Weights (Critical)
```python
dtype = torch.float32  # Always use fp32 for master weights

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=dtype,  # fp32 master weights
    attn_implementation=attn_implementation,
    ...
)
```

- Model parameters are kept in fp32
- `bf16=True` in TrainingArguments enables AMP (Automatic Mixed Precision)
- AMP uses bf16 for compute (GEMMs) but fp32 for master weights
- This is the **recommended approach** vs hard-casting to bf16

**References:**
- https://huggingface.co/docs/transformers/en/perf_train_gpu_one#bf16

### 3. SDPA Backend Configuration (Critical)
```python
# Disable problematic backends at startup
if disable_cudnn_sdpa:
    torch.backends.cuda.enable_cudnn_sdp(False)
if disable_flash_sdpa:
    torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

- Disables cuDNN SDPA by default (known regressions)
- Can optionally disable Flash SDPA if needed
- Enables memory-efficient attention

**References:**
- https://github.com/pytorch/pytorch/issues/100005
- NVIDIA cuDNN release notes documenting SDPA fixes

### 4. Dedicated PAD Token for Llama-3.x (Important)
```python
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
```

- Llama-3 tokenizers have token 0 as "!" not PAD
- Training on real tokens as PAD causes instability
- Adds proper PAD token and resizes embeddings before PEFT wrapping

**References:**
- https://github.com/turboderp/exllamav2/issues/415

### 5. Remove device_map="auto" (Important)
```python
device_map = None  # device_map="auto" is for inference only
```

- `device_map="auto"` is not supported for training
- Can interfere with gradients and DDP
- Use single-device, DDP, FSDP, or ZeRO instead

**References:**
- https://discuss.huggingface.co/t/what-is-the-proper-way-to-use-device-map-auto-with-trainer/31801

### 6. Conditional torch.compile (Important)
```python
enable_torch_compile: bool = False  # Enable after confirming stability

if torch.__version__ >= "2" and sys.platform != "win32" and enable_torch_compile:
    model = torch.compile(model)
```

- `torch.compile` can amplify SDPA+bf16 issues
- Now disabled by default; enable after confirming stability
- Open PyTorch issues document bf16 SDPA failures under compile

**References:**
- https://github.com/pytorch/pytorch/issues/100005

### 7. PEFT Adapter Precision (Already Correct)
- PEFT keeps adapters in fp32 by default (confirmed correct behavior)
- Adapters are stored in fp32 and upcast inputs for stability
- No changes needed; documented for clarity

**References:**
- https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py

## Usage

### Default (Stable) Configuration
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --adapter_name="dora" \
    --bf16=True \
    --lora_r=32 \
    --lora_alpha=64
    # attn_implementation="eager" (default)
    # disable_cudnn_sdpa=True (default)
    # enable_torch_compile=False (default)
```

### After Stack Upgrade (Flash Attention 2)
Once you upgrade CUDA drivers, cuDNN, and PyTorch to versions with SDPA fixes:
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --adapter_name="dora" \
    --bf16=True \
    --attn_implementation="flash_attention_2" \
    --disable_cudnn_sdpa=False \
    --enable_torch_compile=True  # Enable after confirming stability
```

### Diagnostic Mode
If you still see issues:
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --adapter_name="dora" \
    --bf16=True \
    --attn_implementation="eager" \
    --disable_cudnn_sdpa=True \
    --disable_flash_sdpa=True  # Disable both fused backends
```

## Quick Diagnostics

1. **Single-batch test**: Run one training step. If loss is finite, the fused attention path was the issue.

2. **Disable cuDNN SDPA only**: If training stabilizes, the issue is cuDNN's SDPA implementation.

3. **Check adapter dtype**: Verify PEFT adapters are fp32 (should be by default).

4. **Monitor first 10 steps**: Watch for:
   - `grad_norm: nan`
   - `loss: 0.0` or `loss: nan`
   - These indicate the fix isn't working

## Performance Notes

- **Eager vs Flash**: Eager is slower but more stable. Flash Attention 2 is ~2x faster but requires a stable stack.
- **torch.compile**: Can provide 10-20% speedup but only enable after confirming stability.
- **FP32 master weights**: Minimal overhead with bf16 AMP; this is the standard training configuration.

## Community References

These fixes are based on documented issues and solutions:
- [Training fails with FA2, succeeds with eager](https://github.com/huggingface/transformers/issues/28687)
- [cuDNN SDPA loss explosions](https://github.com/pytorch/pytorch/issues/100005)
- [Llama-3 tokenizer PAD token issue](https://github.com/turboderp/exllamav2/issues/415)
- [Flash-attention NaNs during training](https://discuss.huggingface.co/t/flash-attention-nans/51863)

## Expected Behavior

After these fixes:
- ✅ Training should start with finite loss at step 0
- ✅ Gradient norms should be stable (typically 0.1-10.0 range)
- ✅ Loss should decrease smoothly
- ✅ No NaN or 0.0 loss values
- ✅ Stable training across different seeds

If issues persist, try:
1. Reduce learning rate (e.g., 1e-5 → 5e-6)
2. Disable all fused attention (`disable_flash_sdpa=True`)
3. Use fp16 instead of bf16 as a diagnostic step
4. Check CUDA/cuDNN versions and upgrade if outdated
