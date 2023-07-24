
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import CAMAttention, DownsampleBlock, UpsampleBlock
from ittr import HPB, ChanLayerNorm
from networks import adaILN


class AdaLINITTRParamsInferenceNet(nn.Module):
  def __init__(self, ngf: int, mult: int, ff_inner_dim: int, img_size: int,  light: bool):
    super(AdaLINITTRParamsInferenceNet, self).__init__()

    self.light = light
    if self.light:
      self.fc = nn.Sequential(
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.GELU(),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.GELU()
      )
    else:
      self.fc = nn.Sequential(
          nn.Linear(
              img_size // mult * img_size // mult * ngf * mult,
              ngf * mult,
              bias=False
          ),
          nn.GELU(),
          nn.Linear(ngf * mult, ngf * mult, bias=False),
          nn.GELU()
      )
    self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
    self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
    self.gamma_ff = nn.Linear(ngf * mult, ff_inner_dim, bias=False)
    self.beta_ff = nn.Linear(ngf * mult, ff_inner_dim, bias=False)

  def forward(self, x: torch.Tensor):
    if self.light:
      x_ = F.adaptive_avg_pool2d(x, 1)
      x_ = self.fc(x_.view(x_.shape[0], -1))
    else:
      x_ = self.fc(x.view(x.shape[0], -1))
    return ((self.gamma(x_), self.gamma_ff(x_)), (self.beta(x_), self.beta_ff(x_)))


class ITTRGenerator(nn.Module):
  def __init__(self,
               input_nc: int,
               output_nc: int,
               ngf=64,
               n_bottleneck_blocks=9,
               n_downsampling=2,
               img_size=256,
               light=True,
               nce_layers_indices: List[int] = [],
               ff_mult=4,
               ada_lin: bool = True
               ):
    super(ITTRGenerator, self).__init__()

    self.input_nc = input_nc
    self.output_nc = output_nc
    self.ngf = ngf
    self.n_bottleneck_blocks = n_bottleneck_blocks
    self.n_resnet_enc_blocks = n_bottleneck_blocks // 2
    self.n_resnet_dec_blocks = n_bottleneck_blocks // 2
    if n_bottleneck_blocks % 2:
      self.n_resnet_enc_blocks += 1

    self.n_downsampling = n_downsampling
    self.nce_layers_indices = nce_layers_indices
    self.img_size = img_size
    self.light = light
    self.ada_lin = ada_lin

    self.enc_down = nn.ModuleList([
        nn.Identity(),
        DownsampleBlock(
            reflection_padding=3,
            in_channels=input_nc,
            out_channels=ngf,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False,
            activation=nn.GELU()
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
              bias=False,
              activation=nn.GELU()
          )
      ]

    self.enc_bottleneck = nn.ModuleList([])
    mult = 2**n_downsampling
    for i in range(self.n_resnet_enc_blocks):
      self.enc_bottleneck += [
          HPB(
              dim=ngf * mult,
              norm=nn.InstanceNorm2d,
              dpsa_norm=ChanLayerNorm
          )
      ]

    self.cam = CAMAttention(channels=ngf * mult, activation=nn.GELU())

    self.ada_lin_infer = nn.Identity()
    if self.ada_lin:
      self.ada_lin_infer = AdaLINITTRParamsInferenceNet(
          ngf,
          mult,
          ngf * mult * ff_mult,
          img_size,
          light
      )

    self.dec_bottleneck = nn.ModuleList([
        HPB(
            dim=ngf * mult,
            norm=adaILN,
            dpsa_norm=adaILN
        ) if self.ada_lin else
        HPB(
            dim=ngf * mult,
            norm=nn.InstanceNorm2d,
            dpsa_norm=ChanLayerNorm
        )
        for _ in range(self.n_resnet_dec_blocks)
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
              activation=nn.GELU()
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

    if self.ada_lin:
      gamma, beta = self.ada_lin_infer(x)

    for layer in self.dec_bottleneck:
      if self.ada_lin:
        x = layer(x, gamma, beta)
      else:
        x = layer(x)
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
