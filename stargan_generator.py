import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import CAMAttention

# Downsamples using average pooling, perhaps try downsampling with convolution


class ResBlk(nn.Module):
  def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
               normalize=False, downsample=False):
    super().__init__()
    self.actv = actv
    self.normalize = normalize
    self.downsample = downsample
    self.learned_sc = dim_in != dim_out
    self._build_weights(dim_in, dim_out)

  def _build_weights(self, dim_in, dim_out):
    self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
    self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
    if self.normalize:
      self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
      self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
    if self.learned_sc:
      self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

  def _shortcut(self, x):
    if self.learned_sc:
      x = self.conv1x1(x)
    if self.downsample:
      x = F.avg_pool2d(x, 2)
    return x

  def _residual(self, x):
    if self.normalize:
      x = self.norm1(x)
    x = self.actv(x)
    x = self.conv1(x)
    if self.downsample:
      x = F.avg_pool2d(x, 2)
    if self.normalize:
      x = self.norm2(x)
    x = self.actv(x)
    x = self.conv2(x)
    return x

  def forward(self, x):
    x = self._shortcut(x) + self._residual(x)
    return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
  def __init__(self, style_dim, num_features):
    super().__init__()
    self.norm = nn.InstanceNorm2d(num_features, affine=False)
    self.fc = nn.Linear(style_dim, num_features*2)

  def forward(self, x, s):
    h = self.fc(s)
    h = h.view(h.size(0), h.size(1), 1, 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
  def __init__(self, dim_in, dim_out, style_dim=64,
               actv=nn.LeakyReLU(0.2), upsample=False):
    super().__init__()
    self.actv = actv
    self.upsample = upsample
    self.learned_sc = dim_in != dim_out
    self._build_weights(dim_in, dim_out, style_dim)

  def _build_weights(self, dim_in, dim_out, style_dim=64):
    self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
    self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
    self.norm1 = AdaIN(style_dim, dim_in)
    self.norm2 = AdaIN(style_dim, dim_out)
    if self.learned_sc:
      self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

  def _shortcut(self, x):
    if self.upsample:
      x = F.interpolate(x, scale_factor=2, mode='nearest')
    if self.learned_sc:
      x = self.conv1x1(x)
    return x

  def _residual(self, x, s):
    x = self.norm1(x, s)
    x = self.actv(x)
    if self.upsample:
      x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = self.conv1(x)
    x = self.norm2(x, s)
    x = self.actv(x)
    x = self.conv2(x)
    return x

  def forward(self, x, s):
    out = self._residual(x, s)
    return out


# Maybe use this to condition style on domain explicitely
class AdaINParamsInferenceNet(nn.Module):
  def __init__(self, in_channels: int, out_dim: int):
    super(AdaINParamsInferenceNet, self).__init__()

    self.nn = nn.Linear(
        in_features=2*in_channels,
        out_features=out_dim
    )
    self.activation = nn.LeakyReLU(0.2)

  def forward(self, x: torch.Tensor):
    batch_size, _, _, _ = x.shape
    x_avg = F.adaptive_avg_pool2d(x, 1)
    x_max = F.adaptive_max_pool2d(x, 1)
    x_summary = torch.cat((x_avg, x_max), dim=1).view((batch_size, -1))
    out = self.nn(x_summary)
    out = self.activation(out)
    return out


class StarGANGenerator(nn.Module):
  def __init__(self,
               img_size=256,
               use_ada_in_params_net: bool = False,
               ada_in_params_dim=64,
               max_conv_dim=512,
               nce_layers_indices: List[int] = []):
    super().__init__()
    dim_in = 2**14 // img_size
    self.img_size = img_size
    self.from_rgb = nn.ModuleList([
        nn.Identity(),  # for easier NCE indexing,
        nn.Conv2d(3, dim_in, 3, 1, 1)
    ])
    self.encode = nn.ModuleList([])
    self.decode = nn.ModuleList()
    self.to_rgb = nn.Sequential(
        nn.InstanceNorm2d(dim_in, affine=True),
        nn.LeakyReLU(0.2),
        nn.Conv2d(dim_in, 3, 1, 1, 0))

    self.ada_in_params_dim = ada_in_params_dim
    self.nce_layers_indices = nce_layers_indices

    # down/up-sampling blocks
    repeat_num = int(np.log2(img_size)) - 4

    for _ in range(repeat_num):
      dim_out = min(dim_in*2, max_conv_dim)
      self.encode.append(
          ResBlk(dim_in, dim_out, normalize=True, downsample=True))
      self.decode.insert(
          0, AdainResBlk(dim_out, dim_in, ada_in_params_dim, upsample=True))  # stack-like
      dim_in = dim_out

    print(use_ada_in_params_net)
    if use_ada_in_params_net:
      self.ada_infer = AdaINParamsInferenceNet(
          in_channels=max_conv_dim,
          out_dim=ada_in_params_dim
      )
    else:
      self.ada_infer = nn.Identity()  # for now

    # bottleneck blocks
    for _ in range(2):
      self.encode.append(
          ResBlk(dim_out, dim_out, normalize=True))
      self.decode.insert(
          0, AdainResBlk(dim_out, dim_out, ada_in_params_dim))

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
