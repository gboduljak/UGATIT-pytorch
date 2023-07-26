from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import ILN, ResnetAdaILNBlock, ResnetBlock


class DownsampleBlock(nn.Sequential):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int,
               stride: int,
               reflection_padding: int,
               padding: int,
               bias: bool
               ):
    super().__init__(
        nn.ReflectionPad2d(reflection_padding),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        ),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )


class UpsampleBlock(nn.Sequential):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               scale_factor: int,
               kernel_size: int,
               stride: int,
               reflection_padding: int,
               padding: int,
               bias: bool,
               activation: nn.Module
               ):
    super().__init__(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        nn.ReflectionPad2d(reflection_padding),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        ),
        ILN(out_channels),
        activation
    )


class CAMAttention(nn.Module):
  def __init__(self, channels: int):
    super(CAMAttention, self).__init__()

    self.gap_fc = nn.Linear(in_features=channels, out_features=1, bias=False)
    self.gmp_fc = nn.Linear(in_features=channels, out_features=1, bias=False)
    self.conv1x1 = nn.Conv2d(
        in_channels=2 * channels,
        out_channels=channels,
        kernel_size=1,
        stride=1,
        bias=True
    )
    self.relu = nn.ReLU(True)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gap = F.adaptive_avg_pool2d(x, 1)
    gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
    gap_weight = list(self.gap_fc.parameters())[0]
    gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

    gmp = F.adaptive_max_pool2d(x, 1)
    gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
    gmp_weight = list(self.gmp_fc.parameters())[0]
    gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

    cam_logit = torch.cat([gap_logit, gmp_logit], 1)
    x = torch.cat([gap, gmp], 1)
    x = self.relu(self.conv1x1(x))

    heatmap = torch.sum(x, dim=1, keepdim=True)

    return x, cam_logit, heatmap


class AdaLINStyleCodeNetwork(nn.Module):
  def __init__(self, in_channels: int, out_dim: int):
    super(AdaLINStyleCodeNetwork, self).__init__()

    self.nn = nn.Sequential(
        nn.Linear(
            in_features=2*in_channels,
            out_features=in_channels
        ),
        nn.LeakyReLU(0.2),
        nn.Linear(
            in_features=in_channels,
            out_features=in_channels
        ),
        nn.LeakyReLU(0.2),
        nn.Linear(
            in_features=in_channels,
            out_features=in_channels
        ),
        nn.LeakyReLU(0.2),
        nn.Linear(
            in_features=in_channels,
            out_features=out_dim
        ),
    )

  def forward(self, x: torch.Tensor):
    batch_size, _, _, _ = x.shape
    x_avg = F.adaptive_avg_pool2d(x, 1)
    x_max = F.adaptive_max_pool2d(x, 1)
    x_summary = torch.cat((x_avg, x_max), dim=1).view((batch_size, -1))
    return self.nn(x_summary)


class ResnetGenerator(nn.Module):
  def __init__(self,
               input_nc: int,
               output_nc: int,
               ngf=64,
               n_resnet_blocks=9,
               n_downsampling=2,
               style_code_dim: int = 64,
               img_size=256,
               light=True,
               nce_layers_indices: List[int] = [],
               ):
    super(ResnetGenerator, self).__init__()

    self.input_nc = input_nc
    self.output_nc = output_nc
    self.ngf = ngf
    self.n_resnet_blocks = n_resnet_blocks
    self.n_resnet_enc_blocks = n_resnet_blocks // 2
    self.n_resnet_dec_blocks = n_resnet_blocks // 2
    if n_resnet_blocks % 2:
      self.n_resnet_enc_blocks += 1

    self.n_downsampling = n_downsampling
    self.nce_layers_indices = nce_layers_indices
    self.img_size = img_size
    self.light = light

    self.enc_down = nn.ModuleList([
        nn.Identity(),
        DownsampleBlock(
            reflection_padding=3,
            in_channels=input_nc,
            out_channels=ngf,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False
        )
    ])
    for i in range(n_downsampling):
      mult = 2**i
      self.enc_down += [
          DownsampleBlock(
              reflection_padding=1,
              in_channels=ngf * mult,
              out_channels=ngf * mult * 2,
              kernel_size=3,
              stride=2,
              padding=0,
              bias=False
          )
      ]

    self.enc_bottleneck = nn.ModuleList([])
    mult = 2**n_downsampling
    for i in range(self.n_resnet_enc_blocks):
      self.enc_bottleneck += [ResnetBlock(ngf * mult, use_bias=False)]

    self.cam = CAMAttention(ngf * mult)
    self.ada_lin_infer = AdaLINStyleCodeNetwork(
        in_channels=ngf*mult,
        out_dim=style_code_dim
    )

    self.dec_bottleneck = nn.ModuleList([
        ResnetAdaILNBlock(
            dim=ngf * mult,
            use_bias=False,
            style_code_dim=style_code_dim
        ) for _ in range(self.n_resnet_dec_blocks)
    ])

    self.dec_up = nn.ModuleList([])

    for i in range(n_downsampling):
      mult = 2**(n_downsampling - i)
      self.dec_up += [
          UpsampleBlock(
              reflection_padding=1,
              in_channels=ngf * mult,
              out_channels=int(ngf * mult / 2),
              scale_factor=2,
              kernel_size=3,
              stride=1,
              padding=0,
              bias=False,
              activation=nn.ReLU(True)
          )
      ]

    self.dec_up += [
        nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7,
                      stride=1, padding=0, bias=False),
            nn.Tanh()
        )
    ]

    self.layers = dict(
        enumerate(
            self.enc_down +
            self.enc_bottleneck +
            [self.cam] +
            [self.ada_lin_infer] +
            self.dec_bottleneck +
            self.dec_up
        )
    )

  def forward(self, x: torch.Tensor, nce: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[torch.Tensor]]:
    if nce:
      assert self.nce_layers_indices

    nce_layers = [
        self.layers[layer_idx]
        for layer_idx in self.nce_layers_indices
    ]
    final_nce_layer = nce_layers[-1] if nce_layers else None
    nce_layers_outs = []

    for layer in self.enc_down:
      x = layer(x)
      if layer in nce_layers:
        nce_layers_outs.append(x)
      if nce and layer == final_nce_layer:
        return nce_layers_outs

    for layer in self.enc_bottleneck:
      x = layer(x)
      if layer in nce_layers:
        nce_layers_outs.append(x)
      if nce and layer == final_nce_layer:
        return nce_layers_outs

    x, cam_logits, heatmap = self.cam(x)
    if self.cam in nce_layers:
      nce_layers_outs.append(x)
    if nce and self.cam == final_nce_layer:
      return nce_layers_outs

    s = self.ada_lin_infer(x)

    for layer in self.dec_bottleneck:
      x = layer(x, s)
      if layer in nce_layers:
        nce_layers_outs.append(x)
      if nce and layer == final_nce_layer:
        return nce_layers_outs

    for layer in self.dec_up:
      x = layer(x)
      if layer in nce_layers:
        nce_layers_outs.append(x)
      if nce and layer == final_nce_layer:
        return nce_layers_outs

    return x, cam_logits, heatmap
