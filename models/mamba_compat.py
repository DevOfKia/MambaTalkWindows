"""
Pure-PyTorch drop-in for mamba_ssm.
Implements Mamba (SSM), Block, GatedMLP, MHA stub, MambaConfig.
Slower than CUDA kernels but requires no compiler.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


# ---------------------------------------------------------------------------
# Core Mamba layer (pure PyTorch selective scan)
# ---------------------------------------------------------------------------

class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # dt initialisation
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: d_inner × d_state
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        B, L, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)          # B, L, 2*d_inner
        x, z = xz.chunk(2, dim=-1)               # each B, L, d_inner

        # Causal conv
        x = x.transpose(1, 2)                     # B, d_inner, L
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2)                     # B, L, d_inner
        x = self.act(x)

        # Project to SSM params
        x_dbl = self.x_proj(x)                    # B, L, dt_rank + 2*d_state
        dt, B_ssm, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))          # B, L, d_inner

        A = -torch.exp(self.A_log.float())         # d_inner, d_state
        y = self._selective_scan(x, dt, A, B_ssm, C, self.D)

        y = y * self.act(z)
        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C, D):
        """Pure-PyTorch sequential scan. O(L) memory, slower than CUDA kernel."""
        batch, L, d_inner = u.shape
        d_state = A.shape[1]

        # B, L, d_inner, 1  ×  1, 1, d_inner, d_state  →  B, L, d_inner, d_state
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB_u[:, i]         # B, d_inner, d_state
            y_i = (h * C[:, i, :].unsqueeze(1)).sum(-1)  # B, d_inner
            ys.append(y_i)

        y = torch.stack(ys, dim=1)                 # B, L, d_inner
        return y + u * D


# ---------------------------------------------------------------------------
# Mamba2 stub (not used with default config — ssm_layer defaults to Mamba1)
# ---------------------------------------------------------------------------

class Mamba2(Mamba):
    """Stub — treated as Mamba1 for compatibility."""
    pass


# ---------------------------------------------------------------------------
# GatedMLP
# ---------------------------------------------------------------------------

class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.SiLU,
        bias: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        return self.fc2(y * self.act(gate))


# ---------------------------------------------------------------------------
# MHA stub (not used — attn_layer_idx defaults to [])
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    def __init__(self, d_model, layer_idx=None, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=max(1, d_model // 64), batch_first=True)

    def forward(self, x, inference_params=None, **kwargs):
        out, _ = self.attn(x, x, x)
        return out


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is nn.Identity:
            self.mlp = None
        else:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)

    def forward(self, hidden_states, residual=None, inference_params=None, **kwargs):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        if self.mlp is not None:
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Stubs for unused HF utilities
# ---------------------------------------------------------------------------

class GenerationMixin:
    pass


def load_config_hf(model_name):
    raise NotImplementedError("load_config_hf not available in compat mode")


def load_state_dict_hf(model_name, device=None, dtype=None):
    raise NotImplementedError("load_state_dict_hf not available in compat mode")
