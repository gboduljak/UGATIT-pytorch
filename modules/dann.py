from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchmetrics.functional as M
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerTypeUnion
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD

from models.feature_extractor import feature_extractor
from models.resnet import ResNetType
from models.tllib import DomainDiscriminator
from models.tllib.dann import DomainAdversarialLoss, ImageClassifier
from models.tllib.grl import WarmStartGradientReverseLayer
from modules.uda_base_module import UDABaseModule


class DANN(UDABaseModule):
  def __init__(self,
               num_classes: int,
               backbone_type: ResNetType = ResNetType.ResNet50,
               backbone_pretrained: bool = True,
               bottleneck_dim: Optional[int] = 256,
               domain_discriminator_hidden_size: int = 1024,
               domain_discriminator_bn: bool = True,
               domain_adv_loss_tradeoff: float = 1.0,
               domain_gamma: float = 1.0,
               lr: float = 0.01,
               lr_alpha: float = 0.001,
               lr_beta: float = 0.75,
               momentum: float = 0.9,
               weight_decay: float = 1e-3,
               nesterov: bool = True,
               ):
    super().__init__()

    self.save_hyperparameters()
    self.automatic_optimization = False
    extractor = feature_extractor(
        backbone_type=backbone_type,
        backbone_pretrained=backbone_pretrained
    )
    self.classifier = ImageClassifier(
        backbone=extractor.backbone,
        pool_layer=extractor.avgpool,
        num_classes=num_classes,
        bottleneck_dim=bottleneck_dim
    )
    self.domain_adv_loss = DomainAdversarialLoss(
        domain_discriminator=DomainDiscriminator(
            in_feature=self.classifier.features_dim,
            hidden_size=domain_discriminator_hidden_size,
            batch_norm=domain_discriminator_bn
        ),
        grl=WarmStartGradientReverseLayer(
            alpha=domain_gamma,  # domain_gamma is 10 in paper, in tllib it is 1.0
            lo=0.,
            hi=1.,
            max_iters=1000,  # max_iters parameter does not always correspond to maximum number of training iterations
            auto_step=True
        )
    )

  def forward(self, x):
    return self.classifier(x)

  def extract_features(self, x):
    f = self.classifier.backbone(x)
    f = self.classifier.pool_layer(f)
    f = self.classifier.bottleneck(f)
    return f

  def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
    optimizer = self.optimizers()
    scheduler = self.lr_schedulers()

    [source_batch, target_batch] = batch

    x_src, labels_src = source_batch
    x_tgt, _ = target_batch
    assert (x_src.shape == x_tgt.shape)

    x = torch.cat((x_src, x_tgt), dim=0)
    
    y, f = self.classifier(x)
    y_s, _ = y.chunk(2, dim=0)
    f_s, f_t = f.chunk(2, dim=0)

    src_cls_loss = F.cross_entropy(y_s, labels_src)
    transfer_loss = self.domain_adv_loss(f_s, f_t)
    loss = src_cls_loss + transfer_loss * self.hparams.domain_adv_loss_tradeoff

    src_cls_acc = M.accuracy(
        preds=y_s,
        target=labels_src,
        task='multiclass',
        num_classes=self.hparams.num_classes
    )
    metrics = {
        'train_src_cls_loss': src_cls_loss,
        'train_src_cls_acc': src_cls_acc,
        'train_domain_adv_loss': transfer_loss,
        'train_loss': loss
    }
    self.log_dict(metrics, prog_bar=True)

    optimizer.zero_grad()
    self.manual_backward(loss)
    optimizer.step()
    self.lr_scheduler_step(scheduler)

  def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion = None, metric: Any | None = None) -> None:
    scheduler.step()

  def configure_optimizers(self) -> Any:
    optimizer = SGD(
        params=(
            self.classifier.get_parameters() +
            self.domain_adv_loss.domain_discriminator.get_parameters()
        ),
        lr=self.hparams.lr,
        momentum=self.hparams.momentum,
        weight_decay=self.hparams.weight_decay,
        nesterov=self.hparams.nesterov
    )
    lr = self.hparams.lr
    alpha = self.hparams.lr_alpha
    beta = self.hparams.lr_beta
    scheduler = LambdaLR(
        optimizer,
        lambda x: lr * (1. + alpha * float(x)) ** (-beta)
    )
    return [optimizer], [scheduler]
