# -*- coding: utf-8 -*-
from typing import Iterator, Tuple, Optional, Set
import torch
from torch import nn
from transformers import Trainer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

def iter_lora_factors(model: nn.Module,
                      target_adapter_keys: Optional[Set[str]] = None
                      ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            for name in m.lora_A.keys():
                if target_adapter_keys and name not in target_adapter_keys:
                    continue
                A_lin = m.lora_A[name]
                B_lin = m.lora_B[name]
                if hasattr(A_lin, "weight") and hasattr(B_lin, "weight"):
                    yield (B_lin.weight, A_lin.weight)


class SpectralRefactorTrainer(Trainer):
    def __init__(self,
                 *args,
                 refactor_every: int = 100,
                 refactor_mode: str = "balanced",
                 balance_lambda: float = 1.0,
                 target_adapter_keys: Optional[Set[str]] = None,
                 warmup_steps: int = 0,
                 preserve_momentum: bool = True,
                 clear_momentum: bool = False,
                 damping_eps: float = 0.0,
                 clip_min_sigma: float = 0.0,
                 only_large_layers: bool = False,
                 large_dim_threshold: int = 1024,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.refactor_every = max(1, int(refactor_every))
        self.refactor_mode = refactor_mode
        self.balance_lambda = float(balance_lambda)
        self.target_adapter_keys = set(target_adapter_keys) if target_adapter_keys else None
        self.warmup_steps = max(0, int(warmup_steps))
        self.preserve_momentum = preserve_momentum
        self.clear_momentum = clear_momentum
        self.damping_eps = float(damping_eps)
        self.clip_min_sigma = float(clip_min_sigma)
        self.only_large_layers = bool(only_large_layers)
        self.large_dim_threshold = int(large_dim_threshold)

    def _layer_is_large(self, B: torch.Tensor, A: torch.Tensor) -> bool:
        d_out, r = B.shape
        r2, d_in = A.shape
        return (max(d_out, d_in) >= self.large_dim_threshold)


    def get_exp_avg(self, p: torch.Tensor) -> Optional[torch.Tensor]:
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param is p:
                        state = self.optimizer.state.get(param, None)
                        if not state:
                            return None
                        return state.get("exp_avg", None)
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
    def _refactor_once(self):
        
        model = self.model
        was_training = model.training
        model.eval()

        #balance_lambda= self.balance_lambda + (1- self.balance_lambda) * self.state.global_step / self.state.max_steps
        #print(f"balance_lambda = {balance_lambda} at step {self.state.global_step}") 
        for (B, A) in iter_lora_factors(model, self.target_adapter_keys):
            if self.only_large_layers and not self._layer_is_large(B, A):
                continue
            mB = self.get_exp_avg(B) if self.preserve_momentum else None
            mA = self.get_exp_avg(A) if self.preserve_momentum else None
            self.spectral_refactor_with_momentum_map(
                B=B.data, A=A.data,
                mB=mB, mA=mA,
            )
            
            if self.clear_momentum:
                self._reset_adam_state(B)
                self._reset_adam_state(A)
                mB = self.get_exp_avg(B)
                mA = self.get_exp_avg(A)
                print(f"  Cleared momentum for B, norm={mB.norm().item() if mB is not None else 'N/A'}")
                print(f"  Cleared momentum for A, norm={mA.norm().item() if mA is not None else 'N/A'}")

        if was_training:
            model.train()
    
    @torch.no_grad()
    def spectral_refactor_with_momentum_map(self,
        B: torch.Tensor, A: torch.Tensor,
        mB: Optional[torch.Tensor] = None, mA: Optional[torch.Tensor] = None,
    ) -> None:
        """
        对 (B,A) 做小核 SVD 谱重构，并将动量 (mB,mA) 同构映射到新坐标，保证 mB A + B mA 不变。
        返回 (mB_new, mA_new)。若 mB/mA 为 None，则对应返回 None。
        """
        dev = B.device
        dtype_B, dtype_A = B.dtype, A.dtype
        mode = self.refactor_mode
        balance_lambda= self.balance_lambda + (1- self.balance_lambda) * self.state.global_step / self.state.max_steps
        damping_eps = self.damping_eps
        clip_min_sigma = self.clip_min_sigma

        # 1) thin-QR
        Qb, Rb = torch.linalg.qr(B.float(), mode="reduced")    # [d_out,r], [r,r] (upper)
        A_T = A.t().float()                                    # [d_in, r]
        Qa, Ra = torch.linalg.qr(A_T, mode="reduced")          # [d_in,r],  [r,r] (upper)
        # 2) small core SVD
        C = Rb @ Ra.t()                                        # [r,r]
        Uc, S, Vh = torch.linalg.svd(C, full_matrices=False)   # Uc:[r,r], S:[r], Vh:[r,r]
        # 3) balance spectrum
        if damping_eps > 0.0:
            S = S + float(damping_eps)
        if clip_min_sigma > 0.0:
            S = torch.clamp(S, min=float(clip_min_sigma))
        if mode == "exact":
            S_tilde = S
        elif mode == "balanced":
            S_bar = S.mean()
            S_tilde = (1.0 - float(balance_lambda)) * S + float(balance_lambda) * S_bar
        else:
            raise ValueError(f"Unknown mode: {mode}")
        S_half = torch.sqrt(torch.clamp(S_tilde, min=0.0))
        # 4) assemble refactored B', A'  via small transforms T_B, T_A
        #    B' = B T_B,  A' = T_A A
        #    T_B = Rb^{-1} Uc S_half
        #    T_A = S_half Vc^T (Ra^T)^{-1} ; note Vc^T = Vh
        eye = torch.eye(S.numel(), device=dev, dtype=torch.float32)
        Rb_inv = torch.linalg.solve_triangular(Rb, eye, upper=True)         # Rb^{-1}
        RaT_inv = torch.linalg.solve_triangular(Ra.t(), eye, upper=False)   # (Ra^T)^{-1}
        VcT = Vh                                                             # [r,r]
        # diag as vector -> expand by unsqueeze with broadcasting
        T_B = (Rb_inv @ (Uc @ torch.diag(S_half))).to(B.dtype).to(dev)      # [r,r]
        T_A = (torch.diag(S_half) @ VcT @ RaT_inv).to(A.dtype).to(dev)      # [r,r]

        # 写回 B,A：B <- B @ T_B,  A <- T_A @ A
        B.copy_(B @ T_B)
        A.copy_(T_A @ A)

        # 5) momentum mapping (preserve mB A + B mA)
        #    mB' = mB T_A^{-1} with T_A^{-1} = Ra^T Vc S_half^{-1}
        #    mA' = T_B^{-1} mA with T_B^{-1} = S_half^{-1} Uc^T Rb
        mB_new, mA_new = None, None
        if (mB is not None) or (mA is not None):
            S_half_inv = torch.diag(1.0 / (S_half + 1e-12))
            T_A_inv = (Ra.t() @ (Vh.t()) @ S_half_inv).to(A.dtype).to(dev)  # Ra^T Vc S^{-1}
            T_B_inv = (S_half_inv @ Uc.t() @ Rb).to(B.dtype).to(dev)        # S^{-1} Uc^T Rb
            if mB is not None:
                # mB' = mB @ T_A^{-1}
                mB_new = (mB @ T_A_inv).to(mB.dtype)
                mB.copy_(mB_new)
            if mA is not None:
                # mA' = T_B^{-1} @ mA
                mA_new = (T_B_inv @ mA).to(mA.dtype)
                mA.copy_(mA_new)

    def training_step(self, model, inputs,num_items_in_batch):
        ret = super().training_step(model, inputs,num_items_in_batch)
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if self.state.global_step < self.warmup_steps:
                return ret
            if self.state.global_step < self.refactor_every or self.state.global_step % self.refactor_every != 0 :
                return ret
            self._refactor_once()
        return ret