from typing import List, Optional, Tuple, Union

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


class AdaLINParamsInferenceNet(nn.Module):
  def __init__(self, ngf: int, mult: int, img_size: int, light: bool, noisy: bool, noise_generator: Optional[torch.Generator] = None):
    super(AdaLINParamsInferenceNet, self).__init__()

    self.light = light
    self.noisy = noisy
    self.noise_generator = noise_generator

    noisy_mult = 1 if not noisy else 2

    if self.light:
      self.fc = nn.Sequential(
          nn.Linear(ngf * mult * noisy_mult, ngf * mult, bias=False),
          nn.ReLU(True),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.ReLU(True),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.ReLU(True),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.ReLU(True),
      )
    else:
      if self.noisy:
        raise NotImplementedError
      self.fc = nn.Sequential(
          nn.Linear(
              (img_size // mult * img_size // mult * ngf * mult),
              ngf * mult,
              bias=False
          ),
          nn.ReLU(True),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.ReLU(True)
      )
    self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
    self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

  def forward(self, x: torch.Tensor):
    if self.light:
      x_ = F.adaptive_avg_pool2d(x, 1)
      if self.noisy:
        z_ = torch.randn(
            x_.shape,
            generator=self.noise_generator,
            device=x_.device
        )
        x_ = torch.cat([x_, z_], dim=1)
      x_ = self.fc(x_.view(x_.shape[0], -1))
    else:
      x_ = self.fc(x.view(x.shape[0], -1))
    return self.gamma(x_), self.beta(x_)


class ResnetGenerator(nn.Module):
  def __init__(self,
               input_nc: int,
               output_nc: int,
               ngf=64,
               n_resnet_blocks=9,
               n_downsampling=2,
               img_size=256,
               light=True,
               nce_layers_indices: List[int] = [],
               noisy: bool = False,
               noise_generator: Optional[torch.Generator] = None
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
    self.noisy = noisy
    self.noise_generator = noise_generator

    if self.noisy:
      assert self.noise_generator

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
    self.ada_lin_infer = AdaLINParamsInferenceNet(
        ngf,
        mult,
        img_size,
        light,
        noisy,
        noise_generator
    )

    self.dec_bottleneck = nn.ModuleList([
        ResnetAdaILNBlock(
            dim=ngf * mult,
            use_bias=False,
            noisy=self.noisy,
            noise_generator=self.noise_generator
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

    gamma, beta = self.ada_lin_infer(x)

    for layer in self.dec_bottleneck:
      x = layer(x, gamma, beta)
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
