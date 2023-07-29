from itertools import chain
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *


def plot_translation_examples(
        generator: nn.Module,
        patch_sampler: nn.Module,
        trainA_iter: Iterable,
        trainA_loader: DataLoader,
        trainB_iter: Iterable,
        trainB_loader: DataLoader,
        valA_iter: Iterable,
        valA_loader: DataLoader,
        valB_iter: Iterable,
        valB_loader: DataLoader,
        testA_iter: Iterable,
        testA_loader: DataLoader,
        testB_iter: Iterable,
        testB_loader: DataLoader,
        device: torch.device,
        A2B_results_filename: str,
        B2B_results_filename: str,
        train_examples_num: int = 4,
        val_examples_num: int = 4,
        test_examples_num: int = 4,
        img_size: int = 256,
        cut_type: str = 'vanilla'
):

  def compute_example_matrices(
      source_examples_iter: Iterable,
      source_examples_loader: DataLoader,
      target_examples_iter: Iterable,
      target_examples_loader: DataLoader,
      examples_num: int
  ):
    with torch.no_grad():
      A2B, B2B = [], []
      for _ in range(examples_num):
        try:
          real_A, _ = next(source_examples_iter)
        except:
          source_examples_iter = iter(source_examples_loader)
          real_A, _ = next(source_examples_iter)
        try:
          real_B, _ = next(target_examples_iter)
        except:
          target_examples_iter = iter(target_examples_loader)
          real_B, _ = next(target_examples_iter)

        real_A, real_B = real_A.to(device), real_B.to(device)
        fake_A2B, _, fake_A2B_heatmap = generator(real_A)
        fake_B2B, _, fake_B2B_heatmap = generator(real_B)

        if cut_type == 'vanilla':
          A2B.append(
              np.vstack([
                  RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                  cam(tensor2numpy(fake_A2B_heatmap[0]), img_size),
                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
              ])
          )
          B2B.append(
              np.vstack([
                  RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                  cam(tensor2numpy(fake_B2B_heatmap[0]), img_size),
                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
              ])
          )
        else:
          loss_attn_A2B = patch_sampler(
              generator(real_A, nce=True),
              return_only_full_attn_maps=True,
          )
          loss_attn_B2B = patch_sampler(
              generator(real_B, nce=True),
              return_only_full_attn_maps=True,
          )
          A2B.append(
              np.vstack([
                  RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                  cam(tensor2numpy(fake_A2B_heatmap[0]), img_size),
                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
              ] + [
                  cam(tensor2numpy(attn), img_size)
                  for attn in loss_attn_A2B
              ])
          )
          B2B.append(
              np.vstack([
                  RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                  cam(tensor2numpy(fake_B2B_heatmap[0]), img_size),
                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0])))
              ] + [
                  cam(tensor2numpy(attn), img_size)
                  for attn in loss_attn_B2B
              ])
          )
      return A2B, B2B

  train_A2B, train_B2B = compute_example_matrices(
      source_examples_iter=trainA_iter,
      source_examples_loader=trainA_loader,
      target_examples_iter=trainB_iter,
      target_examples_loader=trainB_loader,
      examples_num=train_examples_num
  )
  val_A2B, val_B2B = compute_example_matrices(
      source_examples_iter=valA_iter,
      source_examples_loader=valA_loader,
      target_examples_iter=valB_iter,
      target_examples_loader=valB_loader,
      examples_num=val_examples_num
  )
  test_A2B, test_B2B = compute_example_matrices(
      source_examples_iter=testA_iter,
      source_examples_loader=testA_loader,
      target_examples_iter=testB_iter,
      target_examples_loader=testB_loader,
      examples_num=test_examples_num
  )

  A2B = np.hstack(list(chain(train_A2B, val_A2B, test_A2B)))
  B2B = np.hstack(list(chain(train_B2B, val_B2B, test_B2B)))

  cv2.imwrite(A2B_results_filename, A2B * 255.0)
  cv2.imwrite(B2B_results_filename, B2B * 255.0)
