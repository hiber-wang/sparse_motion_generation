from typing import Optional

import torch
import torch.nn as nn


from .embedders import TimestepEmbedder
from .poolers import AttentionPool
from .block import DiTBlock
from .pos_embed import RotaryEmbedding


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ControlNet(nn.Module):
    """
    Modified from HunyuanControlNet
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 1024,
                 num_layers: int = 13,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 cond_dim: int = 130,
                 num_frames: int = 196,
                 learn_sigma: bool = False,
                 norm: str = 'layer',
                 qk_norm: bool = True,
                 use_rotary: bool = True,
                 **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = input_dim * 2 if learn_sigma else input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.norm = norm

        self.x_embedder = nn.Linear(input_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.y_embedder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.extra_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim, bias=True)
        )

        self.pooler = AttentionPool(num_frames, hidden_dim, num_heads=8, output_dim=hidden_dim)
        self.rotary = RotaryEmbedding(hidden_dim // num_heads // 2) if use_rotary else None

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_dim,
                       c_emb_size=hidden_dim,
                       num_heads=num_heads,
                       mlp_ratio=mlp_ratio,
                       qk_norm=qk_norm,
                       norm_type=self.norm,
                       skip=False,
                       rotary_embed=self.rotary
                    )
            for _ in range(num_layers)
        ])


        # Input zero linear for the first block
        self.before_proj = zero_module(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Output zero linear for the every block
        self.after_proj_list = nn.ModuleList(
            [zero_module(nn.Linear(self.hidden_dim, self.hidden_dim)) for _ in range(len(self.blocks))]
        )


    def from_dit(self, dit):
        """
        Load the parameters from a pre-trained HunYuanDiT model.

        Parameters
        ----------
        dit: HunYuanDiT
            The pre-trained HunYuanDiT model.
        """
                
        self.pooler.load_state_dict(dit.pooler.state_dict())
        self.x_embedder.load_state_dict(dit.x_embedder.state_dict())
        self.t_embedder.load_state_dict(dit.t_embedder.state_dict())
        self.y_embedder.load_state_dict(dit.y_embedder.state_dict())
        self.extra_embedder.load_state_dict(dit.extra_embedder.state_dict())
        if not self.rotary is None:
            self.rotary.load_state_dict(dit.rotary.state_dict())

        for i, block in enumerate(self.blocks):
            block.load_state_dict(dit.blocks[i].state_dict())

    def set_trainable(self):
        self.pooler.requires_grad_(False)
        self.x_embedder.requires_grad_(False)
        self.t_embedder.requires_grad_(False)
        self.extra_embedder.requires_grad_(False)
        self.y_embedder.requires_grad_(False)

        self.blocks.requires_grad_(True)
        self.before_proj.requires_grad_(True)
        self.after_proj_list.requires_grad_(True)

        self.blocks.train()
        self.before_proj.train()
        self.after_proj_list.train()

            
    
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True,
                **kwargs,
                ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        return_dict: bool
            Whether to return a dictionary.
        """
        # ========================= Build time and motion embedding =========================
        c = self.t_embedder(t)
        x = self.x_embedder(x)

        # ========================= Deal with Condition =========================
        condition = self.y_embedder(condition)

        # ========================= Forward pass through HunYuanDiT blocks =========================
        controls = []
        x = x + self.before_proj(condition) # add condition
        for layer, block in enumerate(self.blocks):
            x = block(x, c, condition, padding_mask)
            controls.append(self.after_proj_list[layer](x)) # zero linear for output


        if return_dict:
            return {'controls': controls}
        return controls