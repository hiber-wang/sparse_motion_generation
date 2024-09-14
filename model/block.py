# Modified from https://github.com/facebookresearch/DiT/blob/main/models.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from torch.utils import checkpoint

from .attn_layers import Attention, CrossAttention
from .norm_layers import FP32_Layernorm, FP32_SiLU, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(dim=1)) + shift.unsqueeze(dim=1)



class MLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            act_layer(),
            nn.Dropout(drop),
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity(),
            nn.Linear(hidden_features, out_features, bias=bias),
            nn.Dropout(drop),
        )


    def forward(self, x):
        return self.model(x)



#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    Modifed from HunYuanDiT block.
    """
    def __init__(self,
                 hidden_size: int,
                 c_emb_size: int,
                 num_heads: int,
                 rotary_embed: Optional[nn.Module] = None,
                 mlp_ratio: float = 4.0,
                 qk_norm: bool = False,
                 norm_type: str = "layer",
                 skip: bool = False,
                 ):
        super().__init__()
        use_ele_affine = True

        if norm_type == "layer":
            norm_layer = FP32_Layernorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, rotary_embed=rotary_embed)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0) # type: ignore

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            FP32_SiLU(),
            nn.Linear(c_emb_size, hidden_size, bias=True)
        )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(hidden_size, hidden_size, num_heads=num_heads, qkv_bias=True,
                                        qk_norm=qk_norm, rotary_embed=rotary_embed)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = norm_layer(2 * hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.gradient_checkpointing = False

    def _forward(self, 
                 x: torch.Tensor, 
                 c: torch.Tensor, 
                 y: Optional[torch.Tensor] = None, 
                 padding_mask: Optional[torch.Tensor] = None, 
                 skip: Optional[torch.Tensor] = None):
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1) # type: ignore
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        attn_inputs = (
            self.norm1(x) + shift_msa, padding_mask,
        )
        x = x + self.attn1(*attn_inputs)[0]

        # Cross-Attention
        cross_inputs = (
            self.norm3(x), y, padding_mask
        )
        x = x + self.attn2(*cross_inputs)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x

    def forward(self,
                x: torch.Tensor, 
                c: torch.Tensor, 
                y: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None, 
                skip: Optional[torch.Tensor] = None):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward, x, c, y, padding_mask, skip)
        return self._forward(x, c, y, padding_mask, skip)



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
