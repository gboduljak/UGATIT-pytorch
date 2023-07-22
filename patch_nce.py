import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class PatchNCELoss(nn.Module):
  def __init__(self,
               temperature: float,
               batch_size: int,
               patch_dim: int,
               use_external_patches: bool = False):
    super().__init__()
    self.temperature = temperature
    self.batch_size = batch_size
    self.patch_dim = patch_dim
    self.use_external_patches = use_external_patches

  def forward(self, q_patches: torch.Tensor, k_patches: torch.Tensor):
    # q_patches : [batch_size, num_patches_per_batch, patch_dim]
    # k_patches : [batch_size, num_patches_per_batch, patch_dim]
    _, num_patches_per_batch, _ = q_patches.shape
    # logits: [batch_size, num_patches_per_batch, num_patches_per_batch]
    #       - diagonal entries are positives
    #       - non-diagonal entries are negatives
    logits = (1 / self.temperature) * einsum(
        q_patches,
        k_patches,
        'b i d, b j d -> b i j'
    )
    labels = torch.arange(num_patches_per_batch).to(logits.device).long()
    # out : [batch_size, ]
    return F.cross_entropy(
        input=rearrange(
            logits,
            'b i j -> (b i) j'
        ),
        target=repeat(
            labels,
            'i -> (b i)',
            b=self.batch_size
        ),
        reduction='none'
    )
