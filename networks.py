import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch.nn.parameter import Parameter


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


class SimpleConcatSkipUp(nn.Module):
  def __init__(self, y_channels: int):
    super(SimpleConcatSkipUp, self).__init__()
    self.y_channels = y_channels
    self.s_channels = self.y_channels // 2
    self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=self.y_channels,
            out_channels=self.s_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        ),
        ILN(self.s_channels),
        nn.ReLU(True)
    )
    self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_channels=2 * self.s_channels,
            out_channels=self.s_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        ),
        ILN(self.s_channels),
        nn.ReLU(True)
    )

  def forward(self, y, s):
    x = torch.cat((self.up(y), s), dim=1)
    x = self.conv(x)
    return x


class NoisyDownBlock(nn.Module):
  def __init__(self, ngf, mult, noise_channels, spectral_norm=False, instance_norm=True, kernel_size=3, stride=2, bias=False, activation=nn.ReLU(True)):
    super().__init__()

    def normalize(conv):
      if spectral_norm:
        return nn.utils.spectral_norm(conv)
      else:
        return conv

    self.block = nn.Sequential(*[
        nn.ReflectionPad2d(1),
        normalize(nn.Conv2d(
            in_channels=ngf * mult + noise_channels,
            out_channels=ngf * mult * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias
        )),
        nn.InstanceNorm2d(ngf * mult * 2) if instance_norm else nn.Identity(),
        activation
    ])

  def forward(self, x, f):
    # spatially expand
    f_spatial = f.view(f.size(0), f.size(1), 1, 1)
    f_spatial = f_spatial.expand(
        f_spatial.size(0), f_spatial.size(1), x.size(2), x.size(3)
    )
    x_with_f = torch.cat([x, f_spatial], 1)  # concatenate on channels axis

    return self.block(x_with_f)


class NoisyResnetBlock(nn.Module):
  def __init__(self, dim, noise_channels, use_bias, spectral_norm=False):
    super(NoisyResnetBlock, self).__init__()

    def normalize(conv):
      if spectral_norm:
        return nn.utils.spectral_norm(conv)
      else:
        return conv

    conv_block = []
    conv_block += [nn.ReflectionPad2d(1),
                   normalize(nn.Conv2d(dim + noise_channels, dim, kernel_size=3, stride=1,
                             padding=0, bias=use_bias)),
                   nn.InstanceNorm2d(2*dim),
                   nn.ReLU(True)]

    conv_block += [nn.ReflectionPad2d(1),
                   normalize(nn.Conv2d(dim, dim, kernel_size=3,
                             stride=1, padding=0, bias=use_bias)),
                   nn.InstanceNorm2d(2*dim)]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x, f):
    # spatially expand
    f_spatial = f.view(f.size(0), f.size(1), 1, 1)
    f_spatial = f_spatial.expand(
        f_spatial.size(0), f_spatial.size(1), x.size(2), x.size(3)
    )
    x_with_f = torch.cat([x, f_spatial], 1)  # concatenate on channels axis

    out = x + self.conv_block(x_with_f)
    return out


class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, noise_dim=256, n_classes=3, num_heads=4, condition_on_latent=False):
    assert (n_blocks >= 0)
    super(Generator, self).__init__()
    self.input_nc = input_nc
    self.output_nc = output_nc
    self.ngf = ngf
    self.n_blocks = n_blocks
    self.img_size = img_size
    self.n_classes = n_classes
    self.class_embedding = nn.Embedding(n_classes, noise_dim)

    self.init = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_nc, ngf, kernel_size=7,
                  stride=1, padding=0, bias=False),
        nn.InstanceNorm2d(ngf),
        nn.ReLU(True)
    )
    self.condition_on_latent = condition_on_latent
    # Down-Sampling

    n_downsampling = 3
    self.n_downsampling = n_downsampling

    DownBlock = []
    for i in range(n_downsampling):
      mult = 2**i
      DownBlock += [NoisyDownBlock(ngf, mult, noise_dim,
                                   activation=nn.ReLU(True))]

    # Down-Sampling Bottleneck
    mult = 2**n_downsampling
    for i in range(n_blocks):
      DownBlock += [NoisyResnetBlock(ngf * mult,
                                     noise_channels=noise_dim, use_bias=False)]

    self.mhsa = MultiHeadSelfAttention(
        channels=ngf * mult,
        width=img_size // mult,
        height=img_size // mult,
        num_heads=num_heads
    )

    self.gap_fc = nn.Linear(ngf * mult, self.n_classes, bias=False)
    self.gmp_fc = nn.Linear(ngf * mult, self.n_classes, bias=False)

    self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult,
                             kernel_size=1, stride=1, bias=True)
    self.ReLU = nn.ReLU(True)

    # Gamma, Beta block

    FC = [nn.Linear(ngf * mult + noise_dim, ngf * mult, bias=False),
          nn.ReLU(True),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.ReLU(True)]

    self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
    self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

    # Up-Sampling Bottleneck
    for i in range(n_blocks):
      setattr(self, 'UpBlock1_' + str(i+1),
              ResnetAdaILNBlock(ngf * mult, use_bias=False))

    # Up-Sampling
    UpBlock2 = []
    for i in range(n_downsampling):
      mult = 2**(n_downsampling - i)
      UpBlock2 += [
          SimpleConcatSkipUp(ngf * mult)
      ]

    self.DownBlock = nn.Sequential(*DownBlock)
    self.FC = nn.Sequential(*FC)
    self.UpBlock2 = nn.Sequential(*UpBlock2)

    self.out = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7,
                  stride=1, padding=0, bias=False),
        nn.Tanh()
    )

  def forward(self, x, f, c):
    # f = torch.nn.functional.normalize(f, p=2, dim=1)
    cond_vec = self.class_embedding(
        torch.tensor(c, device=x.device).view(x.shape[0], )
    )

    if self.condition_on_latent:
      # combine latent with cond_vec
      pass

    x = self.init(x)
    down_skips = [x]
    for (i, down) in enumerate(self.DownBlock):
      x = down(x, cond_vec)
      if i < self.n_downsampling - 1:
        down_skips.append(x)

    down_skips = list(reversed(down_skips))

    x = self.mhsa(x)

    # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
    # gap_weight = list(self.gap_fc.parameters())[0][c].view((1, -1))

    # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
    # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
    # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
    # gmp_weight = list(self.gmp_fc.parameters())[0][c].view((1, -1))
    # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

    # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
    # x = torch.cat([gap, gmp], 1)
    # x = self.ReLU(self.conv1x1(x))
    # heatmap = torch.sum(x, dim=1, keepdim=True)

    x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    f_ = cond_vec.unsqueeze(-1).unsqueeze(-1)
    x_ = torch.cat([x_, f_], dim=1)
    x_ = self.FC(x_.view(x_.shape[0], -1))

    gamma, beta = self.gamma(x_), self.beta(x_)

    for i in range(self.n_blocks):
      x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)

    for (up, skip) in zip(self.UpBlock2, down_skips):
      x = up(x, skip)

    out = self.out(x)
    return out


class ResnetAdaILNBlock(nn.Module):
  def __init__(self, dim, use_bias):
    super(ResnetAdaILNBlock, self).__init__()
    self.pad1 = nn.ReflectionPad2d(1)
    self.conv1 = nn.Conv2d(dim, dim, kernel_size=3,
                           stride=1, padding=0, bias=use_bias)
    self.norm1 = adaILN(dim)
    self.relu1 = nn.ReLU(True)

    self.pad2 = nn.ReflectionPad2d(1)
    self.conv2 = nn.Conv2d(dim, dim, kernel_size=3,
                           stride=1, padding=0, bias=use_bias)
    self.norm2 = adaILN(dim)

  def forward(self, x, gamma, beta):
    out = self.pad1(x)
    out = self.conv1(out)
    out = self.norm1(out, gamma, beta)
    out = self.relu1(out)
    out = self.pad2(out)
    out = self.conv2(out)
    out = self.norm2(out, gamma, beta)

    return out + x


class adaILN(nn.Module):
  def __init__(self, num_features, eps=1e-5):
    super(adaILN, self).__init__()
    self.eps = eps
    self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.rho.data.fill_(0.9)

  def forward(self, input, gamma, beta):
    in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(
        input, dim=[2, 3], keepdim=True)
    out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
    ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(
        input, dim=[1, 2, 3], keepdim=True)
    out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
    out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + \
        (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
    out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

    return out


class ILN(nn.Module):
  def __init__(self, num_features, eps=1e-5):
    super(ILN, self).__init__()
    self.eps = eps
    self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.rho.data.fill_(0.0)
    self.gamma.data.fill_(1.0)
    self.beta.data.fill_(0.0)

  def forward(self, input):
    in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(
        input, dim=[2, 3], keepdim=True)
    out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
    ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(
        input, dim=[1, 2, 3], keepdim=True)
    out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
    out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + \
        (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
    out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + \
        self.beta.expand(input.shape[0], -1, -1, -1)

    return out


class Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=5, n_classes=3, noise_dim=256):
    super(Discriminator, self).__init__()
    self.class_embedding = nn.Embedding(n_classes, noise_dim)
    self.init = nn.Sequential(*[
        nn.ReflectionPad2d(1),
        nn.utils.spectral_norm(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
        nn.LeakyReLU(0.2, True)
    ])
    down = []
    for i in range(1, n_layers - 2):
      mult = 2 ** (i - 1)
      down += [
          NoisyDownBlock(
              ndf, mult,
              noise_channels=noise_dim,
              spectral_norm=True,
              instance_norm=False,
              kernel_size=4,
              stride=2,
              bias=True,
              activation=nn.LeakyReLU(0.2, True)
          )
      ]

    mult = 2 ** (n_layers - 2 - 1)
    down += [NoisyDownBlock(
        ndf, mult,
        noise_channels=noise_dim,
        spectral_norm=True,
        instance_norm=False,
        kernel_size=4,
        stride=1,
        bias=True,
        activation=nn.LeakyReLU(0.2, True)
    )]
    self.down = nn.Sequential(*down)

    self.n_classes = n_classes
    # Class Activation Map
    mult = 2 ** (n_layers - 2)
    self.gap_fc = nn.utils.spectral_norm(
        nn.Linear(ndf * mult, self.n_classes, bias=False))
    self.gmp_fc = nn.utils.spectral_norm(
        nn.Linear(ndf * mult, self.n_classes, bias=False))

    self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult,
                             kernel_size=1, stride=1, bias=True)
    self.leaky_relu = nn.LeakyReLU(0.2, True)

    self.pad = nn.ReflectionPad2d(1)
    self.conv = nn.utils.spectral_norm(
        nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

  def forward(self, input, c, cam=False):
    cond_vec = self.class_embedding(
        torch.tensor(c, device=input.device).view(input.shape[0], )
    )
    x = input
    x = self.init(x)

    for layer in self.down:
      x = layer(x, cond_vec)

    gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
    gap_weight = list(self.gap_fc.parameters())[0][c].view((1, -1))
    gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

    gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
    gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
    gmp_weight = list(self.gmp_fc.parameters())[0][c].view((1, -1))
    gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

    cam_logit = torch.cat([gap_logit, gmp_logit], 1)
    x = torch.cat([gap, gmp], 1)
    x = self.leaky_relu(self.conv1x1(x))
    heatmap = torch.sum(x, dim=1, keepdim=True)

    x = self.pad(x)
    out = self.conv(x)

    if cam:
      return out, cam_logit, heatmap
    else:
      return out


class RhoClipper(object):

  def __init__(self, min, max):
    self.clip_min = min
    self.clip_max = max
    assert min < max

  def __call__(self, module):

    if hasattr(module, 'rho'):
      w = module.rho.data
      w = w.clamp(self.clip_min, self.clip_max)
      module.rho.data = w
