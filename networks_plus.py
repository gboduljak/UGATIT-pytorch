import torch
import torch.nn as nn
from einops import rearrange

from networks import ILN
from transformer_networks import (LearnedAttentionAggregation,
                                  MultiHeadCrossAttentionUp,
                                  MultiHeadSelfAttention)


class ResnetBlock(nn.Module):
  def __init__(self, dim, use_bias):
    super(ResnetBlock, self).__init__()
    conv_block = []
    conv_block += [nn.ReflectionPad2d(1),
                   nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                             padding=0, bias=use_bias),
                   nn.InstanceNorm2d(dim),
                   nn.SiLU(True)]

    conv_block += [nn.ReflectionPad2d(1),
                   nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                             padding=0, bias=use_bias),
                   nn.InstanceNorm2d(dim)]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out


class adaILN(nn.Module):
  def __init__(self, num_features, eps=1e-5):
    super(adaILN, self).__init__()
    self.eps = eps
    self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
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


class ResnetAdaILNBlock(nn.Module):
  def __init__(self, dim, use_bias):
    super(ResnetAdaILNBlock, self).__init__()
    self.pad1 = nn.ReflectionPad2d(1)
    self.conv1 = nn.Conv2d(dim, dim, kernel_size=3,
                           stride=1, padding=0, bias=use_bias)
    self.norm1 = adaILN(dim)
    self.silu1 = nn.SiLU(True)

    self.pad2 = nn.ReflectionPad2d(1)
    self.conv2 = nn.Conv2d(dim, dim, kernel_size=3,
                           stride=1, padding=0, bias=use_bias)
    self.norm2 = adaILN(dim)

  def forward(self, x, gamma, beta):
    out = self.pad1(x)
    out = self.conv1(out)
    out = self.norm1(out, gamma, beta)
    out = self.silu1(out)
    out = self.pad2(out)
    out = self.conv2(out)
    out = self.norm2(out, gamma, beta)

    return out + x


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
        nn.SiLU(True)
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
        nn.SiLU(True)
    )

  def forward(self, y, s):
    x = torch.cat((self.up(y), s), dim=1)
    x = self.conv(x)
    return x


class ResnetPlusGenerator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, num_heads=4, dropout=0.0, light=True):
    assert (n_blocks >= 0)
    super(ResnetPlusGenerator, self).__init__()
    self.input_nc = input_nc
    self.output_nc = output_nc
    self.ngf = ngf
    self.n_blocks = n_blocks
    self.img_size = img_size
    self.light = light

    DownBlock = []
    DownBlock += [nn.Sequential(
                  nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7,
                            stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.SiLU(True)
                  )]

    # Down-Sampling
    n_downsampling = 2
    for i in range(n_downsampling):
      mult = 2**i
      DownBlock += [nn.Sequential(
          nn.ReflectionPad2d(1),
          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                    stride=2, padding=0, bias=False),
          nn.InstanceNorm2d(ngf * mult * 2),
          nn.SiLU(True)
      )]
    self.n_downsampling = n_downsampling

    # Down-Sampling Bottleneck
    mult = 2**n_downsampling
    for i in range(n_blocks):
      DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

    self.mhsa = MultiHeadSelfAttention(
        channels=ngf * mult,
        width=img_size // mult,
        height=img_size // mult,
        num_heads=num_heads
    )
    # Class Activation Map
    self.attn_pool = LearnedAttentionAggregation(
        dim=ngf * mult,
        num_heads=num_heads
    )
    self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
    self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
    self.gattn_fc = nn.Linear(ngf * mult, 1, bias=False)

    self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult,
                             kernel_size=1, stride=1, bias=True)
    self.silu = nn.SiLU(True)

    # Gamma, Beta block
    if self.light:
      FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.SiLU(True),
            nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.SiLU(True)]
    else:
      FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
            nn.SiLU(True),
            nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.SiLU(True)]
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
          # MultiHeadCrossAttentionUp(
          #     y_channels=ngf * mult,
          #     y_height=self.img_size // mult,
          #     y_width=self.img_size // mult,
          #     num_heads=num_heads,
          #     dropout=dropout,
          # )
      ]
    self.out = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7,
                  stride=1, padding=0, bias=False),
        nn.Tanh()
    )

    self.DownBlock = nn.Sequential(*DownBlock)
    self.FC = nn.Sequential(*FC)
    self.UpBlock2 = nn.Sequential(*UpBlock2)

  def forward(self, x):
    down_skips = []

    for (i, down) in enumerate(self.DownBlock):
      x = down(x)
      if i < self.n_downsampling:
        down_skips.append(x)
    down_skips = list(reversed(down_skips))

    x = self.mhsa(x)

    gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
    gap_weight = list(self.gap_fc.parameters())[0]
    gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

    gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
    gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
    gmp_weight = list(self.gmp_fc.parameters())[0]
    gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

    # gattn = self.attn_pool(rearrange(x, 'b c h w -> b (h w) c'))
    # gattn_logit = self.gattn_fc(gattn.view(x.shape[0], -1))
    # gattn_weight = list(self.gattn_fc.parameters())[0]
    # gattn = x * gattn_weight.unsqueeze(2).unsqueeze(3)

    cam_logit = torch.cat([gap_logit, gmp_logit], 1)
    x = torch.cat([gap, gmp], 1)
    x = self.silu(self.conv1x1(x))

    heatmap = torch.sum(x, dim=1, keepdim=True)

    if self.light:
      x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
      x_ = self.FC(x_.view(x_.shape[0], -1))
    else:
      x_ = self.FC(x.view(x.shape[0], -1))
    gamma, beta = self.gamma(x_), self.beta(x_)

    # resnet
    for i in range(self.n_blocks):
      up = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
      x = up

    for (up, skip) in zip(self.UpBlock2, down_skips):
      x = up(x, skip)

    out = self.out(x)
    return out, cam_logit, heatmap
