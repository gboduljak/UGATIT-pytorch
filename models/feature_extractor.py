
import torch.nn as nn

from models.resnet import ResNet50_Weights, ResNetType, resnet50_backbone

backbones = {
    ResNetType.ResNet50: resnet50_backbone
}
pretrained_backbone_weights = {
    ResNetType.ResNet50: ResNet50_Weights.IMAGENET1K_V2
}


class FeatureExtractor(nn.Module):
  def __init__(
      self,
      backbone_type: ResNetType = ResNetType.ResNet50
  ) -> None:
    super().__init__()

    self.backbone = backbones[backbone_type]()
    self.avgpool = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1)
    )

  def forward(self, x):
    x = self.backbone(x)
    x = self.avgpool(x)
    return x


def feature_extractor(
    backbone_type: ResNetType = ResNetType.ResNet50,
    backbone_pretrained: bool = True
):
  feature_extractor = FeatureExtractor(backbone_type=backbone_type)
  if backbone_pretrained:
    feature_extractor.backbone = backbones[backbone_type](
        weights=pretrained_backbone_weights[backbone_type]
    )
  return feature_extractor
