import torch
import torch.nn as nn
from typing import Tuple, Union, Any, Optional



class CrossAttention(nn.Module):
    """
    Use QK Normalization.
    """
    def __init__(self,
                 qdim: int,
                 kdim: int,
                 num_heads: int,
                 rotary_embed: Optional[nn.Module] = None,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None,
                 norm_layer: Any = nn.LayerNorm,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        self.rotary_embed = rotary_embed
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        mask: torch.Tensor
            (batch, seqlen1), padding mask for x
        """
        if y is None:
            y = x
        b, s1, c = x.shape     # [b, s1, D]
        _, s2, c = y.shape     # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)   # [b, s1, h, d]
        kv = self.kv_proj(y).view(b, s2, 2, self.num_heads, self.head_dim)    # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2) # [b, s, h, d]
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        q = q.transpose(-2, -3).contiguous()        # q ->  B, L1, H, C - B, H, L1, C
        k = k.permute(0, 2, 3, 1).contiguous()      # k ->  B, L2, H, C - B, H, C, L2
        # Apply RoPE if needed
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
        attn = q @ k                                # attn -> B, H, L1, L2
        if not padding_mask is None:
            attn = attn.masked_fill(~padding_mask.unsqueeze(dim=1).unsqueeze(dim=1), -torch.inf)
        attn = attn.softmax(dim=-1)                 # attn -> B, H, L1, L2
        attn = self.attn_drop(attn)
        x = attn @ v.transpose(-2, -3)              # v -> B, L2, H, C - B, H, L2, C    x-> B, H, L1, C
        context = x.transpose(1, 2)                 # context -> B, H, L1, C - B, L1, H, C

        context = context.contiguous().view(b, s1, -1)

        out = self.out_proj(context)  # context.reshape - B, L1, -1
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 rotary_embed: Optional[nn.Module] = None,
                 qkv_bias: bool = True, 
                 qk_norm: bool = False, 
                 attn_drop: float = 0., 
                 proj_drop: float = 0.,
                 norm_layer: Any = nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.rotary_embed = rotary_embed
        self.scale = self.head_dim ** -0.5

        # qkv --> Wqkv
        self.Wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        B, N, C = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)   # [3, b, h, s, d]
        q, k, v = qkv.unbind(0)     # [b, h, s, d]
        q = self.q_norm(q)          # [b, h, s, d]
        k = self.k_norm(k)          # [b, h, s, d]

        # Apply RoPE if needed
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)              # [b, h, s, d] @ [b, h, d, s]
        if not padding_mask is None:
            attn = attn.masked_fill(~padding_mask.unsqueeze(dim=1).unsqueeze(dim=1), -torch.inf)
        attn = attn.softmax(dim=-1)                 # [b, h, s, s]
        attn = self.attn_drop(attn)
        x = attn @ v                                # [b, h, s, d]

        x = x.transpose(1, 2).reshape(B, N, C)      # [b, s, h, d]
        x = self.out_proj(x)
        x = self.proj_drop(x)

        out_tuple = (x,)

        return out_tuple
    