
from enum import Enum
from typing import Any, List, Optional, Type, Union

from torchvision.models import ResNet, ResNet50_Weights, WeightsEnum
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNetType(Enum):
  ResNet18 = 18
  ResNet50 = 50


class ResNetBackbone(ResNet):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.out_features = self.fc.in_features
    del self.avgpool
    del self.fc

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


def get_resnet_backbone(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNetBackbone:

  model = ResNetBackbone(block, layers, **kwargs)

  if weights is not None:
    model_dict = model.state_dict()
    pretrained_dict = weights.get_state_dict(progress=progress)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
    }
    model.load_state_dict(pretrained_dict)

  return model


def resnet50_backbone(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any):
  weights = ResNet50_Weights.verify(weights)
  return get_resnet_backbone(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
