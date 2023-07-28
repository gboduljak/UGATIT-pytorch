from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from strenum import StrEnum

from batch_index_select import *
from paper_qsa_patch_sampler import Normalize

# TODO: We are using different initialization of MLPs in comparison to the paper.
# It might be necessary to verify consequences of this.


class QSAType(StrEnum):
  GLOBAL = 'global'
  LOCAL = 'local'
  GLOBAL_AND_LOCAL = 'global+local'


class QSAPatchSampler(nn.Module):
  def __init__(self,
               patch_embedding_dim: int,
               num_patches_per_layer: int,
               qsa_type: QSAType,
               max_spatial_size: int,
               device: torch.device) -> None:
    super().__init__()
    self.mlps_init = False
    self.patch_embedding_dim = patch_embedding_dim
    self.num_patches_per_layer = num_patches_per_layer
    self.qsa_type = qsa_type
    self.max_spatial_size = max_spatial_size
    assert (self.qsa_type == QSAType.GLOBAL)  # only this is implemented for now
    self.device = device

  def create_mlps_if_necessary(self, layer_outs: List[torch.Tensor]):
    if self.mlps_init:
      return

    for (mlp_id, layer_out) in enumerate(layer_outs):
      B, C, H, W = layer_out.shape
      setattr(
          self,
          f'mlp_{mlp_id}',
          nn.Sequential(
              nn.Linear(
                  in_features=C,
                  out_features=self.patch_embedding_dim
              ),
              nn.ReLU(),
              nn.Linear(
                  in_features=self.patch_embedding_dim,
                  out_features=self.patch_embedding_dim
              )
          ).to(self.device)
      )

    self.mlps_init = True

  def forward(self,
              layer_outs: List[torch.Tensor],
              patch_idx_per_layer: List[Optional[torch.Tensor]] = [],
              attn_map_per_layer: List[Optional[torch.Tensor]] = [],
              apply_mlp: bool = True
              ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # rules: Use patch_idx_per_layer if we have it. Use layer_attn_map if we have it. Otherwise, sample.
    self.create_mlps_if_necessary(layer_outs)

    sampled_patches = []
    sampled_patches_idx = []
    sampled_patches_layer_attn_maps = []

    for layer_idx, layer_out in enumerate(layer_outs):
      B, C, H, W = layer_out.shape
      layer_spatial_size = H * W
      layer_patches = rearrange(layer_out, 'b c h w -> b (h w) c')

      if layer_spatial_size <= self.max_spatial_size:
        if not attn_map_per_layer:
          q = layer_patches
          k = q
          dots = einsum(q, k, 'b i c, b j c -> b i j')
          attn = torch.softmax(dots, dim=2)
          prob = -torch.log(attn)
          prob = torch.where(
              torch.isinf(prob),
              torch.full_like(prob, 0).to(self.device),
              prob
          )
          ent = torch.sum(torch.mul(attn, prob), dim=2)
          _, ent_idx = torch.sort(ent)
          layer_attn_map_idx = ent_idx[:, :self.num_patches_per_layer]
          layer_attn_map = batch_index_select(
              x=attn,
              idx=layer_attn_map_idx
          )
        else:
          layer_attn_map = attn_map_per_layer[layer_idx]

        assert (
            layer_attn_map != None and
            type(layer_attn_map) == torch.Tensor
        )
        v = layer_patches
        layer_sampled_patches = einsum(
            layer_attn_map,
            v,
            'b i j, b j c -> b i c'
        )
        sampled_patches_layer_attn_maps.append(layer_attn_map)
        sampled_patches_idx.append(None)
      else:
        if not patch_idx_per_layer:
          # sample random
          layer_patch_idx = torch.vstack(
              [
                  torch.multinomial(
                      input=torch.ones(layer_spatial_size).to(self.device),
                      num_samples=min(layer_spatial_size,
                                      self.num_patches_per_layer)
                  )
                  for _ in range(B)
              ]
          )
        else:
          # no need to sample, patch idx known
          layer_patch_idx = patch_idx_per_layer[layer_idx]

        assert (
            layer_patch_idx != None and
            type(layer_patch_idx) == torch.Tensor
        )
        layer_sampled_patches = batch_index_select(
            x=layer_patches.to(self.device),
            idx=layer_patch_idx.to(self.device)
        )
        sampled_patches_idx.append(layer_patch_idx)
        sampled_patches_layer_attn_maps.append(None)

      assert (
          layer_sampled_patches != None and
          type(layer_sampled_patches) == torch.Tensor
      )
      layer_mlp = getattr(self, f'mlp_{layer_idx}')
      layer_patch_embeddings = F.normalize(
          input=(
              layer_mlp(layer_sampled_patches)
              if apply_mlp
              else layer_sampled_patches
          ),
          dim=-1,
          p=2
      )
      sampled_patches.append(
          layer_patch_embeddings
      )

    return sampled_patches, sampled_patches_idx, sampled_patches_layer_attn_maps
