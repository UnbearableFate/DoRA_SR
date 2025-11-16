# BF16 Training Stability Checklist

## Pre-Training Verification

Before starting a full training run, verify these items:

### ✅ Configuration Check

Run your training command with `--num_epochs=1 --save_step=999999` and check the startup logs:

```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./commonsense_170k.json" \
    --adapter_name="dora" \
    --num_epochs=1 \
    --bf16=True
```

Look for these lines in the output:

- [ ] `attn_implementation: eager` ✅
- [ ] `disable_cudnn_sdpa: True` ✅
- [ ] `Model loaded successfully. dtype: torch.float32` ✅
- [ ] `Adding dedicated PAD token` (for Llama-3 only) ✅
- [ ] `Disabled cuDNN SDPA for bf16 stability` ✅

### ✅ First 10 Steps Check

Monitor the first 10 training steps:

```
Step 1:
- [ ] loss: [finite number, not NaN or 0.0]
- [ ] grad_norm: [0.1-10.0 range typically]

Step 2-10:
- [ ] loss: decreasing trend or stable
- [ ] grad_norm: no sudden spikes to NaN
- [ ] No CUDA OOM errors
```

### ✅ Memory & Performance Baseline

Compare with your previous runs:

- [ ] Memory usage: Similar or slightly higher (fp32 params)
- [ ] Training speed: Slower with eager (expected, ~2x vs FA2)
- [ ] Tokens/sec: Should be stable across steps

## Common Failure Patterns (and Fixes)

### ❌ Pattern 1: Immediate NaN at Step 0
```
Step 0: loss: nan, grad_norm: nan
```

**Root Cause**: Fused attention overflowing in bf16

**Fix**: Verify `attn_implementation: eager` in logs. If already set, add:
```bash
--disable_flash_sdpa=True
```

### ❌ Pattern 2: Loss becomes 0.0 after first step
```
Step 0: loss: 2.5, grad_norm: 0.8
Step 1: loss: 0.0, grad_norm: 0.0
```

**Root Cause**: PAD token issue or complete gradient collapse

**Fixes**:
1. Verify PAD token was added (check for "Adding dedicated PAD token" in logs)
2. Reduce learning rate by 10x: `--learning_rate=3e-6`
3. Check data is loading correctly

### ❌ Pattern 3: Random NaNs during training
```
Step 50: loss: 1.8, grad_norm: 0.5
Step 51: loss: nan, grad_norm: nan
```

**Root Cause**: Numerical instability in specific batches

**Fixes**:
1. Enable gradient clipping (already enabled: `max_grad_norm=1.0`)
2. Reduce learning rate: `--learning_rate=1e-5`
3. Add `--disable_flash_sdpa=True`

### ❌ Pattern 4: CUDA OOM
```
RuntimeError: CUDA out of memory
```

**Root Cause**: fp32 params use more memory than bf16

**Fixes**:
1. Reduce batch size: `--per_device_train_batch_size=1`
2. Enable gradient checkpointing: `--use_gradient_checkpointing=True`
3. Use gradient accumulation: Increase `--batch_size` without increasing `per_device_train_batch_size`

## Post-Fix Validation

After applying fixes, your training should show:

### ✅ Healthy Training Pattern

```
Step 0:  loss: 2.347, grad_norm: 0.821
Step 1:  loss: 2.298, grad_norm: 0.756
Step 2:  loss: 2.241, grad_norm: 0.698
Step 5:  loss: 2.087, grad_norm: 0.612
Step 10: loss: 1.923, grad_norm: 0.534
```

**Indicators of health**:
- Loss starts at reasonable value (1.5-3.0 for language modeling)
- Loss decreases over time (may fluctuate but trending down)
- Grad norm is stable (0.1-2.0 typically)
- No NaN or Inf values

### ✅ Expected Performance

With stable configuration (`eager` attention):

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Speed | ~2-5k tokens/sec | Depends on GPU/batch size |
| Memory | ~30-40GB for 8B model | With LoRA/DoRA r=32 |
| Loss curve | Smooth decrease | May plateau after some steps |

## Debugging Commands

### 1. Minimal Test Run
Test stability with minimal resources:
```bash
python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./commonsense_170k.json" \
    --adapter_name="lora" \
    --lora_r=8 \
    --batch_size=8 \
    --per_device_train_batch_size=1 \
    --num_epochs=1 \
    --eval_step=5 \
    --save_step=999999 \
    --bf16=True
```

### 2. Verbose Logging
Add diagnostic prints:
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

python finetune.py [your args]
```

### 3. Single Batch Overfit Test
Verify model can learn on a tiny dataset:
```bash
# Create tiny dataset (first 10 examples)
head -10 commonsense_170k.json > test_10.json

python finetune.py \
    --base_model="meta-llama/Llama-3.1-8B" \
    --data_path="./test_10.json" \
    --adapter_name="dora" \
    --batch_size=10 \
    --num_epochs=100 \
    --val_set_size=0 \
    --bf16=True
```

**Expected**: Loss should drop to near 0 (overfitting on 10 examples)

### 4. Compare FP16 vs BF16
Isolate bf16-specific issues:
```bash
# FP16 run
python finetune.py [args] --bf16=False --fp16=True

# BF16 run
python finetune.py [args] --bf16=True --fp16=False
```

If fp16 is stable but bf16 is not, the issue is definitely bf16-related (and the fixes should help).

## Environment Verification

### Check CUDA/PyTorch Setup
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

**Expected output** (example for GH200):
```
PyTorch: 2.x.x
CUDA: 12.4
cuDNN: 9xxx
BF16 supported: True
GPU: NVIDIA GH200
```

### Check PEFT Version
```bash
python -c "import peft; print(peft.__version__)"
```

**Expected**: Recent version (0.8.0+)

### Check Transformers Version
```bash
python -c "import transformers; print(transformers.__version__)"
```

**Expected**: Recent version (4.35.0+)

## Success Criteria

Your training is stable when:

✅ **No NaN/Inf values** in first 100 steps  
✅ **Loss decreases** smoothly over epochs  
✅ **Grad norm stays finite** (< 100 typically)  
✅ **Memory usage is consistent** across steps  
✅ **Can complete full epoch** without crashes  
✅ **Validation loss improves** (if using validation set)  

## Upgrade Path Checklist

Once training is stable with eager attention, you can gradually enable optimizations:

### Phase 1: Stable Baseline (Current)
```bash
--attn_implementation="eager"
--disable_cudnn_sdpa=True
--enable_torch_compile=False
```
- [ ] Runs without NaN for 1 full epoch
- [ ] Loss decreases as expected
- [ ] Memory usage is acceptable

### Phase 2: Test Flash Attention 2
```bash
--attn_implementation="flash_attention_2"
--disable_cudnn_sdpa=True
--enable_torch_compile=False
```
- [ ] Install flash-attn: `pip install flash-attn --no-build-isolation`
- [ ] Run for 100 steps, verify no NaN
- [ ] Compare speed: should be ~2x faster
- [ ] If stable, keep this configuration

### Phase 3: Enable cuDNN SDPA (if needed)
```bash
--attn_implementation="flash_attention_2"
--disable_cudnn_sdpa=False
--enable_torch_compile=False
```
- [ ] Only after CUDA/cuDNN upgrade
- [ ] Test for 100 steps
- [ ] If NaN appears, revert to `disable_cudnn_sdpa=True`

### Phase 4: Enable torch.compile
```bash
--attn_implementation="flash_attention_2"
--disable_cudnn_sdpa=False  # or True if needed
--enable_torch_compile=True
```
- [ ] PyTorch 2.0+ required
- [ ] First compilation is slow (5-10 min)
- [ ] Test for 100 steps after compilation
- [ ] Should provide 10-20% additional speedup

## Quick Reference: What Changed

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| Model dtype | `torch.bfloat16` (if bf16=True) | `torch.float32` | Stability ⬆️ |
| Attention | `sdpa` (default) | `eager` (default) | Stability ⬆️, Speed ⬇️ |
| device_map | `"auto"` | `None` | Compatibility ✅ |
| PAD token | 0 (="!") | `<\|pad\|>` | Stability ⬆️ |
| cuDNN SDPA | Enabled | Disabled | Stability ⬆️ |
| torch.compile | Always on | Opt-in | Stability ⬆️ |

## Contact/Support

If you've followed this checklist and still see instability:

1. **Capture logs**: Save full output of first 20 steps
2. **Capture config**: Run with `--wandb_online=True` to track config
3. **Check versions**: Include PyTorch, CUDA, cuDNN, transformers, peft versions
4. **Document pattern**: When exactly does NaN appear? Step 0? Later?

Common issues and their specific fixes are in `BF16_STABILITY_FIXES.md`.
