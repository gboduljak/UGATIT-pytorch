from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from networks import ILN


class PositionalEncoding(nn.Module):
  def __init__(self) -> None:
    super(PositionalEncoding, self).__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    pos_encoding = self.positional_encoding(h * w, c).to(x.device)
    pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
    x = x.view((b, c, h * w)) + pos_encoding
    return x.view((b, c, h, w))

  def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
    depth = depth / 2

    positions = torch.arange(length)
    depths = torch.arange(depth) / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

    pos_encoding = torch.cat(
        (torch.sin(angle_rads), torch.cos(angle_rads)),
        dim=-1
    )

    return pos_encoding


class MultiHeadAttention(nn.Module):
  def __init__(self,
               embedding_dim: int,
               num_heads: int,
               dropout: float = 0.0):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.head_dim = self.embedding_dim // num_heads
    # ensure embedding can be split equally between multiple heads
    assert (self.head_dim * self.num_heads == self.embedding_dim)

    self.dropout = nn.Dropout(dropout)
    self.scale = self.head_dim ** (-0.5)

  def scaled_dot_product_attention(self, q: torch.tensor, k: torch.tensor):
    # q: [batch_size, num_heads, seq_length, head_dim]
    # k: [batch_size, num_heads, seq_length, head_dim]
    # kt = rearrange(k, 'b h n d -> b h d n')
    qkt = einsum(q, k, 'b h i d, b h j d -> b h i j')
    # qkt: [batch_size, num_heads, seq_length, seq_length]
    attention = F.softmax(qkt * self.scale, dim=-1)
    # attention: [batch_size, num_heads, seq_length, seq_length]
    return self.dropout(attention)

  def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor):
    # q: [batch_size, seq_length, embedding_dim]
    # k: [batch_size, seq_length, embedding_dim]
    # v: [batch_size, seq_length, embedding_dim]
    q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
    k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
    v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
    # q: [batch_size, num_heads, seq_length, head_dim]
    # k: [batch_size, num_heads, seq_length, head_dim]
    # v: [batch_size, num_heads, seq_length, head_dim]
    attention_per_head = self.scaled_dot_product_attention(q, k)
    # attention_per_head: [batch_size, num_heads, seq_length, seq_length]
    out_per_head = einsum(attention_per_head, v, 'b h i k, b h k j-> b h i j')
    # out_per_head: [batch_size, num_heads, seq_length, head_dim]
    out = rearrange(out_per_head, 'b h n d -> b n (h d)')
    # out: [batch_size, seq_length, num_heads * head_dim = embedding_dim]
    return out


class MultiHeadSelfAttention(nn.Module):
  def __init__(self,
               channels: int,
               width: int,
               height: int,
               num_heads: int,
               dropout: float = 0.0):
    super().__init__()

    self.pos_encoding = PositionalEncoding()
    self.channels = channels
    self.width = width
    self.height = height
    self.num_heads = num_heads
    self.embedding_dim = channels
    self.qkv = nn.Linear(
        in_features=self.embedding_dim,
        out_features=3 * self.embedding_dim,
        bias=False
    )
    self.mha = MultiHeadAttention(
        embedding_dim=self.embedding_dim,
        num_heads=self.num_heads,
        dropout=dropout
    )
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.qkv.weight)

  def forward(self, x):
    x = self.pos_encoding(x)
    # x: [batch_size, channels, height, width]
    x = rearrange(x, 'b c h w -> b (h w) c')
    # x: [batch_size, seq_length, embedding_dim], seq_length = h * w, embedding_dim = channels
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    # q: [batch_size, seq_length, embedding_dim]
    # k: [batch_size, seq_length, embedding_dim]
    # v: [batch_size, seq_length, embedding_dim]
    out = self.mha(q, k, v)
    out = rearrange(out, 'b (h w) c  -> b c h w', h=self.height, w=self.width)
    return out


class MultiHeadCrossAttention(nn.Module):
  def __init__(self, y_channels: int, y_height: int, y_width: int, num_heads: int = 4, dropout: float = 0.) -> None:
    super().__init__()
    s_channels = y_channels // 2

    self.y_channels = y_channels
    self.y_height = y_height
    self.y_width = y_width
    self.pos_encoding = PositionalEncoding()
    self.blue_s = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=s_channels,
            out_channels=s_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False
        ),
        nn.InstanceNorm2d(s_channels),
        nn.SiLU(True)
    )
    self.blue_y = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=2 * s_channels,
            out_channels=s_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        ),
        nn.InstanceNorm2d(s_channels),
        nn.SiLU(True)
    )
    self.mha = MultiHeadAttention(
        embedding_dim=s_channels,
        num_heads=num_heads,
        dropout=dropout
    )
    self.purple = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=s_channels,
            out_channels=s_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        ),
        ILN(s_channels),
        nn.Sigmoid()
    )
    self.green_y = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=y_channels,
            out_channels=s_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        ),
        ILN(s_channels),
        nn.SiLU(True)
    )

  def forward(self, y: torch.tensor, s: torch.tensor) -> torch.tensor:
    up_y = self.green_y(y)

    y = self.pos_encoding(y)
    s = self.pos_encoding(s)

    q = self.blue_y(y)
    k = q
    v = self.blue_s(s)

    # q: [batch_size, y_channels, y_height, y_width]
    # k: [batch_size, y_channels, y_height, y_width]
    # v: [batch_size, s_channels, y_height, y_width]

    q = rearrange(q, 'b c h w -> b (h w) c')
    k = rearrange(k, 'b c h w -> b (h w) c')
    v = rearrange(v, 'b c h w -> b (h w) c')

    mha_out = self.mha(q, k, v)
    mha_out = rearrange(
        mha_out,
        'b (h w) c  -> b c h w',
        h=self.y_height,
        w=self.y_width
    )
    z = self.purple(mha_out) * s
    return torch.cat((z, up_y), dim=1)


class MultiHeadCrossAttentionUp(nn.Module):
  def __init__(self,
               y_channels: int,
               y_height: int,
               y_width: int,
               num_heads: int = 4,
               dropout: float = 0) -> None:
    super().__init__()
    self.mhca = MultiHeadCrossAttention(
        y_channels=y_channels,
        y_height=y_height,
        y_width=y_width,
        num_heads=num_heads,
        dropout=dropout,
    )
    self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=y_channels,
            out_channels=int(y_channels / 2),
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        ),
        ILN(int(y_channels / 2)),
        nn.SiLU(True)
    )

  def forward(self, y: torch.tensor, s: torch.tensor):
    return self.conv(self.mhca(y, s))


class LeaarnedAttentionAggregation(nn.Module):
  # Taken from https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py.
  def __init__(
      self,
      dim: int,
      num_heads: int = 1,
      qkv_bias: bool = False,
      qk_scale: Optional[float] = None,
      attn_drop: float = 0.0,
      proj_drop: float = 0.0,
  ):
    super().__init__()
    self.num_heads = num_heads
    head_dim: int = dim // num_heads
    # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
    self.scale = qk_scale or head_dim**-0.5

    self.q = nn.Linear(dim, dim, bias=qkv_bias)
    self.k = nn.Linear(dim, dim, bias=qkv_bias)
    self.v = nn.Linear(dim, dim, bias=qkv_bias)
    self.id = nn.Identity()
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    q = self.q(x[:, 0]).unsqueeze(1).reshape(
        B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, C //
                          self.num_heads).permute(0, 2, 1, 3)

    q = q * self.scale
    v = self.v(x).reshape(B, N, self.num_heads, C //
                          self.num_heads).permute(0, 2, 1, 3)

    attn = q @ k.transpose(-2, -1)
    attn = self.id(attn)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
    x_cls = self.proj(x_cls)
    x_cls = self.proj_drop(x_cls)

    return x_cls
