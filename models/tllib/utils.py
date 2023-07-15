from typing import Optional

import torch


def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
  r"""
  First shift, then calculate log, which can be described as:

  .. math::
      y = \max(\log(x+\text{offset}), 0)

  Used to avoid the gradient explosion problem in log(x) function when x=0.

  Args:
      x (torch.Tensor): input tensor
      offset (float, optional): offset size. Default: 1e-6

  .. note::
      Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
  """
  return torch.log(torch.clamp(x + offset, max=1.))


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
  """Computes the accuracy for binary classification"""
  with torch.no_grad():
    batch_size = target.size(0)
    pred = (output >= 0.5).float().t().view(-1)
    correct = pred.eq(target.view(-1)).float().sum()
    correct.mul_(100. / batch_size)
    return correct


def accuracy(output, target, topk=(1,)):
  r"""
  Computes the accuracy over the k top predictions for the specified values of k

  Args:
      output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
      target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
      topk (sequence[int]): A list of top-N number.

  Returns:
      Top-N accuracies (N :math:`\in` topK).
  """
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().sum(dtype=torch.float32)
      res.append(correct_k * (100.0 / batch_size))
    return res
