import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional as M


class UDABaseModule(pl.LightningModule):
  def extract_features(self, x):
    raise NotImplementedError()

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    if dataloader_idx == 0:
      # source dataloader
      loss, acc = self.eval_step(batch)
      metrics = {
          'val_src_cls_loss': loss,
          'val_src_cls_acc': acc
      }
      self.log_dict(metrics, prog_bar=True)
    else:
      # target dataloader
      loss, acc = self.eval_step(batch)
      metrics = {
          'val_tgt_cls_loss': loss,
          'val_tgt_cls_acc': acc
      }
      self.log_dict(metrics, prog_bar=True)

  def test_step(self, batch, batch_idx, dataloader_idx=0):
    if dataloader_idx == 0:
      # source datalodaer
      loss, acc = self.eval_step(batch)
      metrics = {
          'test_src_cls_loss': loss,
          'test_src_cls_acc': acc
      }
      self.log_dict(metrics, prog_bar=True)
    else:
      # target dataloader
      loss, acc = self.eval_step(batch)
      metrics = {
          'test_tgt_cls_loss': loss,
          'test_tgt_cls_acc': acc
      }
      self.log_dict(metrics, prog_bar=True)

  def eval_step(self, batch):
    x, y = batch
    logits = self.forward(x)
    loss = F.cross_entropy(logits, y)
    (_, num_classes) = logits.shape
    acc = M.accuracy(
        preds=logits,
        target=y,
        task='multiclass',
        num_classes=num_classes
    )
    return loss, acc
