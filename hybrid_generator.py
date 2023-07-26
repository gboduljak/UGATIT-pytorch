from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import CAMAttention
from stargan_generator import *


class AdaINParamsInferenceNet(nn.Module):
  def __init__(self, in_channels: int, out_dim: int):
    super(AdaINParamsInferenceNet, self).__init__()

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


class HybridGenerator(nn.Module):
  def __init__(self,
               img_size=256,
               use_ada_in_params_net: bool = True,
               ada_in_params_dim=64,
               n_downsampling: int = 2,
               n_bottleneck: int = 9,
               max_conv_dim=256,
               nce_layers_indices: List[int] = []):
    super().__init__()
    dim_in = 2**14 // img_size
    self.img_size = img_size
    self.from_rgb = nn.ModuleList([
        nn.ReflectionPad2d(3),
        nn.Conv2d(
            in_channels=3,
            out_channels=dim_in,
            kernel_size=7,
            stride=1,
        ),
    ])
    self.encode = nn.ModuleList([])
    self.decode = nn.ModuleList()
    self.to_rgb = nn.Sequential(
        nn.InstanceNorm2d(dim_in, affine=True),
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d(3),
        nn.Conv2d(dim_in, 3, 7, 1, 0),
        nn.Tanh()
    )
    self.ada_in_params_dim = ada_in_params_dim
    self.nce_layers_indices = nce_layers_indices

    # down/up-sampling blocks
    repeat_num = n_downsampling

    for _ in range(repeat_num):
      dim_out = min(dim_in*2, max_conv_dim)
      self.encode.append(
          ResBlk(
              dim_in,
              dim_out,
              normalize=True,
              downsample=True
          )
      )
      self.decode.insert(
          0, AdainResBlk(
              dim_out,
              dim_in,
              ada_in_params_dim,
              upsample=True
          )
      )
      dim_in = dim_out
    if use_ada_in_params_net:
      self.ada_infer = AdaINParamsInferenceNet(
          in_channels=max_conv_dim,
          out_dim=ada_in_params_dim
      )
    else:
      self.ada_infer = nn.Identity()  # for now

    # bottleneck blocks
    for _ in range(n_bottleneck // 2):
      self.encode.append(
          ResBlk(
              dim_out,
              dim_out,
              normalize=True
          )
      )
      self.decode.insert(
          0, AdainResBlk(
              dim_out,
              dim_out,
              ada_in_params_dim
          )
      )
    if n_bottleneck % 2:
      self.encode.append(
          ResBlk(
              dim_out,
              dim_out,
              normalize=True
          )
      )

    self.cam = CAMAttention(channels=max_conv_dim)

    self.layers = dict(
        enumerate(
            self.from_rgb +
            self.encode +
            [self.cam] +
            [self.ada_infer] +
            self.decode +
            nn.ModuleList([self.to_rgb])
        )
    )

  def forward(self, x, nce=False):
    if nce:
      assert self.nce_layers_indices

    nce_layers = [
        self.layers[layer_idx]
        for layer_idx in self.nce_layers_indices
    ]
    final_nce_layer = nce_layers[-1] if nce_layers else None
    nce_layers_outs = []
    for block in self.from_rgb:
      x = block(x)
      if block in nce_layers:
        nce_layers_outs.append(x)
      if nce and block == final_nce_layer:
        return nce_layers_outs
    for block in self.encode:
      x = block(x)
      if block in nce_layers:
        nce_layers_outs.append(x)
      if nce and block == final_nce_layer:
        return nce_layers_outs

    x, cam_logits, cam_heatmap = self.cam(x)
    s = self.ada_infer(x)

    for block in self.decode:
      x = block(x, s)

    x = self.to_rgb(x)

    return x, cam_logits, cam_heatmap
