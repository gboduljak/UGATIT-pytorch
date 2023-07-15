from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .grl import WarmStartGradientReverseLayer


class GeneralModule(nn.Module):
  def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: nn.Module,
               head: nn.Module, adv_head: nn.Module, grl: Optional[WarmStartGradientReverseLayer] = None,
               finetune: Optional[bool] = True):
    super(GeneralModule, self).__init__()
    self.backbone = backbone
    self.num_classes = num_classes
    self.bottleneck = bottleneck
    self.head = head
    self.adv_head = adv_head
    self.finetune = finetune
    self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                   auto_step=False) if grl is None else grl

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """"""
    features = self.backbone(x)
    features = self.bottleneck(features)
    outputs = self.head(features)
    features_adv = self.grl_layer(features)
    outputs_adv = self.adv_head(features_adv)
    if self.training:
      return outputs, outputs_adv
    else:
      return outputs

  def step(self):
    """
    Gradually increase :math:`\lambda` in GRL layer.
    """
    self.grl_layer.step()

  def get_parameters(self, base_lr=1.0) -> List[Dict]:
    """
    Return a parameters list which decides optimization hyper-parameters,
    such as the relative learning rate of each layer.
    """
    params = [
        {"params": self.backbone.parameters(), "lr": 0.1 *
         base_lr if self.finetune else base_lr},
        {"params": self.bottleneck.parameters(), "lr": base_lr},
        {"params": self.head.parameters(), "lr": base_lr},
        {"params": self.adv_head.parameters(), "lr": base_lr}
    ]
    return params
