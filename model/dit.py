from typing import List, Dict, Tuple, Optional, Union  

import torch
from torch import nn

from .block import DiTBlock, FinalLayer
from .pos_embed import RotaryEmbedding
from .poolers import AttentionPool
from .embedders import TimestepEmbedder


class MotionDiT(nn.Module):
    """
    Modified from HunYuanDiT.
    """
    def __init__(self,
                 input_dim: int = 315,
                 hidden_dim: int = 1024,
                 num_layers: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 cond_dim: int = 130,
                 num_frames: int = 196,
                 learn_sigma: bool = False,
                 norm: str = 'layer',
                 qk_norm: bool = True,  # See http://arxiv.org/abs/2302.05442 for details.
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
                       skip=layer > num_layers // 2,
                       rotary_embed=self.rotary
                    )
            for layer in range(num_layers)
        ])

        self.final_layer = FinalLayer(hidden_dim, 1, self.output_dim)

        self.initialize_weights()

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                controls: Optional[List[torch.Tensor]] = None,
                **kwargs,
                ) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, L, D_smpl)
        t: torch.Tensor
            (B)
        kp2d: torch.Tensor
            2D keypoints, (B, L, D_2d)

        """
        # ========================= Build time and motion embedding =========================
        c = self.t_embedder(t)
        x = self.x_embedder(x)

        # ========================= Concatenate all extra vectors =========================
        if not condition is None:
            condition = self.y_embedder(condition)
            extra_vec = self.pooler(condition)

            c = c + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Forward pass through HunYuanDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.num_layers // 2:
                if controls is not None:
                    skip = skips.pop() + controls.pop()
                else:
                    skip = skips.pop()
                x = block(x, c, condition, padding_mask, skip)   # (N, L, D)
            else:
                x = block(x, c, condition, padding_mask)         # (N, L, D)

            if layer < (self.num_layers // 2 - 1):
                skips.append(x)
        if controls is not None and len(controls) != 0:
            raise ValueError("The number of controls is not equal to the number of skip connections.")

        # ========================= Final layer =========================
        x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
        if kwargs.get('x_clean') is not None:
            x = kwargs.get('obs_mask', 0.0) * x + ~kwargs.get('obs_mask', 0.0) * kwargs['x_clean']
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder[2].weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.extra_embedder[0].weight, std=0.02)
        nn.init.normal_(self.extra_embedder[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in HunYuanDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.default_modulation[-1].weight, 0)
            nn.init.constant_(block.default_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def _replace_module(self, parent, child_name, new_module, child) -> None:
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.get_base_layer()
        elif hasattr(child, "quant_linear_module"):
            # TODO maybe not necessary to have special treatment?
            child = child.quant_linear_module

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            # if any(prefix in name for prefix in PREFIXES):
            #     module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)




#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################


def get_mask(batch: int, length: int, mask_ratio: float, device: Union[str, torch.device] = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
    return: 
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {'mask': mask, 
            'ids_keep': ids_keep, 
            'ids_restore': ids_restore}



def mask_out_tokens(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    *_, d = x.shape
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(dim=-1).repeat(1, 1, d))
    return x_masked


def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    b, l, d = x.shape
    l_origin = ids_restore.shape[1]
    mask_tokens = mask_token.repeat(b, l_origin - l, 1)
    x = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(dim=-1).repeat(1, 1, d))
    return x


class MaskDiT(nn.Module):
    def __init__(self,
                 input_dim: int = 471,
                 hidden_dim: int = 1024,
                 num_layers: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 learn_sigma: bool = False,
                 norm: str = 'layer',
                 qk_norm: bool = True,
                 use_rotary: bool = True,
                 use_decoder: bool = True,
                 decoder_depth: Optional[int] = None,
                 decoder_num_heads: Optional[int] = None,
                 **kwargs):
        super().__init__()
        # Ordinary DiT configurations
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = input_dim * 2 if learn_sigma else input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.norm = norm

        self.x_embedder = nn.Linear(input_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.rotary = RotaryEmbedding(hidden_dim // num_heads // 2) if use_rotary else None

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_dim,
                       c_emb_size=hidden_dim,
                       num_heads=num_heads,
                       mlp_ratio=mlp_ratio,
                       qk_norm=qk_norm,
                       norm_type=self.norm,
                       skip=layer > num_layers // 2,
                       rotary_embed=self.rotary
                    )
            for layer in range(num_layers)
        ])

        self.final_layer = FinalLayer(hidden_dim, 1, self.output_dim)

        # Decoders, if any
        if use_decoder:
            if decoder_num_heads is None:
                decoder_num_heads = num_heads
            if decoder_depth is None:
                decoder_depth = num_layers // 2 - 1

            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.adapter = FinalLayer(hidden_dim, 1, hidden_dim)
            self.decoder = nn.ModuleList([
                DiTBlock(hidden_size=hidden_dim,
                           c_emb_size=hidden_dim,
                           num_heads=decoder_num_heads,
                           mlp_ratio=mlp_ratio,
                           qk_norm=qk_norm,
                           norm_type=self.norm,
                           skip=False,
                           rotary_embed=self.rotary,
                        )
                for _ in range(decoder_depth)
            ])
        else:
            self.decoder = None

        self.initialize_weights()


    def encode(self, 
               x: torch.Tensor,
               t: torch.Tensor,
               padding_mask: Optional[torch.Tensor] = None,
               mask_ratio: float = 0.5, 
               mask_dict: Optional[Dict[str, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.x_embedder(x)
        c = self.t_embedder(t)

        if mask_ratio > 0.:
            if mask_dict is None:
                mask_dict = get_mask(x.shape[0], x.shape[1], mask_ratio, x.device)  
            x = mask_out_tokens(x, mask_dict['ids_keep'])
        
        # Encoders are skip-connected
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.num_layers // 2:
                skip = skips.pop()
                x = block(x, c, padding_mask=padding_mask, skip=skip)
            else:
                x = block(x, c, padding_mask=padding_mask)
            
            if layer < (self.num_layers // 2 - 1):
                skips.append(x)

        return x, c, mask_dict # type: ignore
    

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                mask_ratio: float = 0.5,
                mask_dict: Optional[Dict[str, torch.Tensor]] = None,
                controls: Optional[List[torch.Tensor]] = None,
                **kwargs,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        enc, c, mask_dict = self.encode(x, t, padding_mask=padding_mask, mask_ratio=mask_ratio, mask_dict=mask_dict)

        if not self.decoder is None:
            x = self.adapter(enc, c)
            x = unmask_tokens(x, mask_dict['ids_restore'], self.mask_token)

            for block in self.decoder:
                control = controls.pop() # type: ignore
                x = block(x, c, padding_mask=padding_mask, skip=control)

        x = self.final_layer(x, c)

        return x, mask_dict
    
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in HunYuanDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.default_modulation[-1].weight, 0)
            nn.init.constant_(block.default_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if not self.decoder is None:
            nn.init.constant_(self.adapter.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adapter.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.adapter.linear.weight, 0)
            nn.init.constant_(self.adapter.linear.bias, 0)
            for block in self.decoder:
                nn.init.constant_(block.default_modulation[-1].weight, 0)
                nn.init.constant_(block.default_modulation[-1].bias, 0)