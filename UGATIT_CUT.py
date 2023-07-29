import itertools
import os
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch_fidelity
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageFolder
from generator import ResnetGenerator
from networks import Discriminator, RhoClipper
from patch_nce import PatchNCELoss
from patch_sampler import PatchSampler
from qsa_patch_sampler import QSAPatchSampler
from seed import get_seeded_generator, seed_everything, seeded_worker_init_fn
from utils import *


class UGATIT_CUT(object):
  def __init__(self, args):
    self.light = args.light
    self.cut = args.cut

    if self.light:
      self.model_name = 'UGATIT_CUT_light'
    else:
      self.model_name = 'UGATIT_CUT'

    self.result_dir = args.result_dir
    self.dataset = args.dataset
    self.ckpt = args.ckpt

    self.iteration = args.iteration
    self.decay_flag = args.decay_flag

    self.batch_size = args.batch_size
    self.print_freq = args.print_freq
    self.val_freq = args.val_freq
    self.save_freq = args.save_freq

    self.lr = args.lr
    self.weight_decay = args.weight_decay
    self.ch = args.ch

    """ Weight """
    self.adv_weight = args.adv_weight
    self.cam_weight = args.cam_weight

    """ Generator """
    self.n_res = args.n_res

    """ Discriminator """
    self.n_dis = args.n_dis

    self.img_size = args.img_size
    self.img_ch = args.img_ch

    self.device = args.device
    self.benchmark_flag = args.benchmark_flag
    self.resume = args.resume
    self.seed = args.seed

    """ CUT """
    self.cut_type = args.cut_type
    self.nce_weight = args.nce_weight
    self.nce_temperature = args.nce_temperature
    self.nce_patch_embedding_dim = args.nce_patch_embedding_dim
    self.nce_n_patches = args.nce_n_patches
    self.nce_layers = [int(x) for x in args.nce_layers.split(',')]

    if torch.backends.cudnn.enabled and self.benchmark_flag:
      print('set benchmark !')
      torch.backends.cudnn.benchmark = True

    """QSA """
    self.qsa_max_spatial_size = args.qsa_max_spatial_size

    print()

    print("##### Information #####")
    print("# cut : ", self.cut)
    print("# cut type : ", self.cut_type)

    print("# light : ", self.light)
    print("# dataset : ", self.dataset)
    if self.ckpt:
      print("# ckpt : ", self.ckpt)
    print("# batch_size : ", self.batch_size)
    print("# iteration per epoch : ", self.iteration)
    print("# seed : ", self.seed)

    print()

    print("##### Generator #####")
    print("# residual blocks : ", self.n_res)
    print()

    print("##### Discriminator #####")
    print("# discriminator layer : ", self.n_dis)

    print()

    print("##### Weight #####")
    print("# adv_weight : ", self.adv_weight)
    print("# cam_weight : ", self.cam_weight)
    print("# nce_weight : ", self.nce_weight)

    print("##### CUT #####")
    print("# nce temperature : ", self.nce_temperature)
    print("# nce layers : ", self.nce_layers)
    print("# nce patches : ", self.nce_n_patches)
    print("# nce patch embedding dim : ", self.nce_patch_embedding_dim)

  ##################################################################################
  # Model
  ##################################################################################

  def build_model(self):
    """ Seed everything """
    seed_everything(self.seed)
    trainA_dataloader_generator = get_seeded_generator(self.seed)
    trainB_dataloader_generator = get_seeded_generator(self.seed)

    """ DataLoader """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((self.img_size + 30, self.img_size+30)),
        transforms.RandomCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    self.trainA = ImageFolder(os.path.join(
        'dataset', self.dataset, 'trainA'), train_transform)
    self.trainB = ImageFolder(os.path.join(
        'dataset', self.dataset, 'trainB'), train_transform)
    self.valA = ImageFolder(os.path.join(
        'dataset', self.dataset, 'valA'), test_transform
    )
    self.valB = ImageFolder(os.path.join(
        'dataset', self.dataset, 'valB'), test_transform
    )
    self.testA = ImageFolder(os.path.join(
        'dataset', self.dataset, 'testA'), test_transform)
    self.testB = ImageFolder(os.path.join(
        'dataset', self.dataset, 'testB'), test_transform)
    self.trainA_loader = DataLoader(
        self.trainA,
        batch_size=self.batch_size,
        worker_init_fn=seeded_worker_init_fn,
        generator=trainA_dataloader_generator,
        shuffle=True
    )
    self.trainB_loader = DataLoader(
        self.trainB,
        batch_size=self.batch_size,
        worker_init_fn=seeded_worker_init_fn,
        generator=trainB_dataloader_generator,
        shuffle=True
    )
    self.valA_loader = DataLoader(self.valA, batch_size=1, shuffle=False)
    self.valB_loader = DataLoader(self.valB, batch_size=1, shuffle=False)
    self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
    self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

    """ Define Generator, Discriminator """
    self.generator = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=self.ch,
        n_resnet_blocks=self.n_res,
        nce_layers_indices=self.nce_layers,
        img_size=self.img_size,
        light=self.light
    ).to(self.device)
    self.global_discriminator = Discriminator(
        input_nc=3,
        ndf=self.ch,
        n_layers=7
    ).to(self.device)
    self.local_discriminator = Discriminator(
        input_nc=3,
        ndf=self.ch,
        n_layers=5
    ).to(self.device)

    """ Define Patch Sampler"""
    if self.cut_type == 'vanilla':
      self.patch_sampler = PatchSampler(
          patch_embedding_dim=self.nce_patch_embedding_dim,
          num_patches_per_layer=self.nce_n_patches,
          device=self.device
      )
    else:
      self.patch_sampler = QSAPatchSampler(
          patch_embedding_dim=self.nce_patch_embedding_dim,
          num_patches_per_layer=self.nce_n_patches,
          qsa_type=self.cut_type,
          max_spatial_size=self.qsa_max_spatial_size,
          device=self.device
      )
    print('Generator:')
    print(self.generator)
    print(f'total params: {get_total_model_params(self.generator)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.generator)}'
    )
    print('Global Discriminator:')
    print(self.global_discriminator)
    print(f'total params: {get_total_model_params(self.global_discriminator)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.global_discriminator)}'
    )
    print('Local Discriminator:')
    print(self.local_discriminator)
    print(f'total params: {get_total_model_params(self.local_discriminator)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.local_discriminator)}'
    )

    """ Define Loss """
    self.L1_loss = nn.L1Loss().to(self.device)
    self.MSE_loss = nn.MSELoss().to(self.device)
    self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
    self.NCE_losses = []

    for _ in self.nce_layers:
      self.NCE_losses.append(
          PatchNCELoss(
              temperature=self.nce_temperature,
              use_external_patches=False
          ).to(self.device)
      )

    """ Trainer """
    self.G_optim = torch.optim.Adam(
        itertools.chain(self.generator.parameters()),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    self.D_optim = torch.optim.Adam(
        itertools.chain(
            self.global_discriminator.parameters(),
            self.local_discriminator.parameters()
        ),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    self.P_optim = None  # not initialized now

    """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
    self.Rho_clipper = RhoClipper(0, 1)

    """ Define """
    self.smallest_val_fid = float('inf')

  def init_patch_sampler_optimizer(self):
    self.P_optim = torch.optim.Adam(
        self.patch_sampler.parameters(),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    if self.resume:
      self.P_optim.param_groups[0]['lr'] -= (self.lr / (
          self.iteration // 2)) * (self.start_iter - self.iteration // 2)
    if self.decay_flag and self.step > (self.iteration // 2):
      self.P_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

  def sample_patches(self, feat_q: torch.Tensor, feat_k: torch.Tensor):
    if self.cut_type == 'vanilla':
      feat_k_pool, feat_k_pool_idx = self.patch_sampler(feat_k)
      feat_q_pool, _ = self.patch_sampler(feat_q, feat_k_pool_idx)
      return (feat_q_pool, feat_k_pool)
    else:
      (feat_k_pool,
       feat_k_patch_idx,
       feat_k_attn_map) = self.patch_sampler(feat_k)
      feat_q_pool, _, _ = self.patch_sampler(
          layer_outs=feat_q,
          patch_idx_per_layer=feat_k_patch_idx,
          attn_map_per_layer=feat_k_attn_map
      )
      return (feat_q_pool, feat_k_pool)

  def calculate_nce_loss(self, src: torch.Tensor, tgt: torch.Tensor):
    n_layers = len(self.nce_layers)

    feat_q = self.generator(tgt, nce=True)
    feat_k = self.generator(src, nce=True)

    should_init_optimizer = not self.patch_sampler.mlps_init

    feat_q_pool, feat_k_pool = self.sample_patches(feat_q, feat_k)

    if should_init_optimizer:
      self.init_patch_sampler_optimizer()

    total_nce_loss = 0.0
    for f_q, f_k, pnce, _ in zip(feat_q_pool, feat_k_pool, self.NCE_losses, self.nce_layers):
      total_nce_loss += pnce(f_q, f_k).mean()

    return total_nce_loss / n_layers

  def train(self):
    self.generator.train(),
    self.global_discriminator.train(),
    self.local_discriminator.train(),
    self.patch_sampler.train()

    start_iter = 1
    if self.resume:
      model_list = glob(os.path.join(
          self.result_dir, self.dataset, 'model', '*.pt'))
      if not len(model_list) == 0:
        model_list.sort()
        start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
        self.load(ckpt='params_%07d.pt' % start_iter)
        print(" [*] Load SUCCESS")
        if self.decay_flag and start_iter > (self.iteration // 2):
          self.G_optim.param_groups[0]['lr'] -= (self.lr / (
              self.iteration // 2)) * (start_iter - self.iteration // 2)
          self.D_optim.param_groups[0]['lr'] -= (self.lr / (
              self.iteration // 2)) * (start_iter - self.iteration // 2)
          if self.P_optim:
            self.P_optim.param_groups[0]['lr'] -= (self.lr / (
                self.iteration // 2)) * (start_iter - self.iteration // 2)

    # training loop
    print('training start !')
    start_time = time.time()
    self.start_iter = start_iter

    for step in range(start_iter, self.iteration + 1):
      self.step = step

      if self.decay_flag and step > (self.iteration // 2):
        self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
        self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
        if self.P_optim:
          self.P_optim.param_groups[0]['lr'] -= (
              self.lr / (self.iteration // 2))
      try:
        real_A, _ = next(trainA_iter)
      except:
        trainA_iter = iter(self.trainA_loader)
        real_A, _ = next(trainA_iter)

      try:
        real_B, _ = next(trainB_iter)
      except:
        trainB_iter = iter(self.trainB_loader)
        real_B, _ = next(trainB_iter)

      real_A, real_B = real_A.to(self.device), real_B.to(self.device)

      # Update D
      self.D_optim.zero_grad()

      fake_A2B, _, _ = self.generator(real_A)

      real_GB_logit, real_GB_cam_logit, _ = self.global_discriminator(real_B)
      real_LB_logit, real_LB_cam_logit, _ = self.local_discriminator(real_B)

      fake_GB_logit, fake_GB_cam_logit, _ = self.global_discriminator(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.local_discriminator(fake_A2B)

      D_ad_loss_GB = self.MSE_loss(
          real_GB_logit,
          torch.ones_like(real_GB_logit).to(self.device)
      ) + self.MSE_loss(
          fake_GB_logit,
          torch.zeros_like(fake_GB_logit).to(self.device)
      )
      D_ad_cam_loss_GB = self.MSE_loss(
          real_GB_cam_logit,
          torch.ones_like(real_GB_cam_logit).to(self.device)
      ) + self.MSE_loss(
          fake_GB_cam_logit,
          torch.zeros_like(fake_GB_cam_logit).to(self.device)
      )
      D_ad_loss_LB = self.MSE_loss(
          real_LB_logit,
          torch.ones_like(real_LB_logit).to(self.device)
      ) + self.MSE_loss(
          fake_LB_logit,
          torch.zeros_like(fake_LB_logit).to(self.device)
      )
      D_ad_cam_loss_LB = self.MSE_loss(
          real_LB_cam_logit,
          torch.ones_like(real_LB_cam_logit).to(self.device)
      ) + self.MSE_loss(
          fake_LB_cam_logit,
          torch.zeros_like(fake_LB_cam_logit).to(self.device)
      )

      Discriminator_domain_gan_loss = D_ad_loss_GB + D_ad_loss_LB
      Discriminator_cam_gan_loss = D_ad_cam_loss_GB + D_ad_cam_loss_LB

      Discriminator_loss = self.adv_weight * (
          Discriminator_domain_gan_loss +
          Discriminator_cam_gan_loss
      )
      # Discriminator_loss = self.adv_weight * (D_ad_loss_LB + D_ad_cam_loss_LB)
      Discriminator_loss.backward()
      self.D_optim.step()

      # Update G
      self.G_optim.zero_grad()
      # Update patch sampler. If we are in the first iteration, its optimizer is not initialized.
      # This is because patch sampler MLPs's dimension depends on the feature map shapes which are known in the first forward pass of NCE loss.
      if self.P_optim:
        self.P_optim.zero_grad()

      fake_A2B, fake_A2B_cam_logit, _ = self.generator(real_A)
      fake_B2B, fake_B2B_cam_logit, _ = self.generator(real_B)

      fake_GB_logit, fake_GB_cam_logit, _ = self.global_discriminator(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.local_discriminator(fake_A2B)

      G_ad_loss_GB = self.MSE_loss(
          fake_GB_logit,
          torch.ones_like(fake_GB_logit).to(self.device)
      )
      G_ad_cam_loss_GB = self.MSE_loss(
          fake_GB_cam_logit,
          torch.ones_like(fake_GB_cam_logit).to(self.device)
      )
      G_ad_loss_LB = self.MSE_loss(
          fake_LB_logit,
          torch.ones_like(fake_LB_logit).to(self.device)
      )
      G_ad_cam_loss_LB = self.MSE_loss(
          fake_LB_cam_logit,
          torch.ones_like(fake_LB_cam_logit).to(self.device)
      )

      G_cam_loss = self.BCE_loss(
          fake_A2B_cam_logit,
          torch.ones_like(fake_A2B_cam_logit).to(self.device)
      ) + self.BCE_loss(
          fake_B2B_cam_logit,
          torch.zeros_like(fake_B2B_cam_logit).to(self.device)
      )

      # this is where NCE goes
      nce_loss_x = self.calculate_nce_loss(real_A, fake_A2B)
      nce_loss_y = self.calculate_nce_loss(real_B, fake_B2B)
      nce_both = (nce_loss_x + nce_loss_y) * 0.5

      Generator_domain_gan_loss = G_ad_loss_GB + G_ad_loss_LB
      Generator_cam_gan_loss = G_ad_cam_loss_GB + G_ad_cam_loss_LB

      Generator_loss = (
          self.adv_weight * (
              Generator_domain_gan_loss +
              Generator_cam_gan_loss
          ) +
          self.nce_weight * nce_both +
          self.cam_weight * G_cam_loss
      )

      Generator_loss.backward()
      self.G_optim.step()
      self.P_optim.step()

      # clip parameter of AdaILN and ILN, applied after optimizer step
      self.generator.apply(self.Rho_clipper)

      train_status_line = "[%5d/%5d] time: %4.4f d_loss: %.8f, d_domain_gan_loss: %.8f, d_cam_gan_loss: %.8f, g_loss: %.8f, g_domain_gan_loss: %.8f, g_cam_gan_loss: %.8f, g_cam_loss: %.8f, nce_loss: %.8f, nce_x: %.8f, nce_y: %.8f" % (
          step,
          self.iteration,
          time.time() - start_time,
          Discriminator_loss,
          Discriminator_domain_gan_loss,
          Discriminator_cam_gan_loss,
          Generator_loss,
          Generator_domain_gan_loss,
          Generator_cam_gan_loss,
          G_cam_loss,
          nce_both,
          nce_loss_x,
          nce_loss_y
      )
      print(train_status_line)
      with open(os.path.join(self.result_dir, self.dataset, 'train_log.txt'), 'a') as tl:
        tl.write(f'{train_status_line}\n')

      if step % self.print_freq == 0:
        train_sample_num = 5
        test_sample_num = 5

        if self.cut_type != 'vanilla':
          A2B = np.zeros((self.img_size * 4, 0, 3))
          B2B = np.zeros((self.img_size * 4, 0, 3))
        else:
          A2B = np.zeros((self.img_size * 3, 0, 3))
          B2B = np.zeros((self.img_size * 3, 0, 3))

        self.generator.eval(),
        self.global_discriminator.eval(),
        self.local_discriminator.eval(),
        self.patch_sampler.eval()
        with torch.no_grad():
          for _ in range(train_sample_num):
            try:
              real_A, _ = next(trainA_iter)
            except:
              trainA_iter = iter(self.trainA_loader)
              real_A, _ = next(trainA_iter)

            try:
              real_B, _ = next(trainB_iter)
            except:
              trainB_iter = iter(self.trainB_loader)
              real_B, _ = next(trainB_iter)
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.generator(real_A)
            fake_B2B, _, fake_B2B_heatmap = self.generator(real_B)

            if self.cut_type == 'vanilla':
              A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                        cam(tensor2numpy(
                                                            fake_A2B_heatmap[0]), self.img_size),
                                                        RGB2BGR(tensor2numpy(
                                                            denorm(fake_A2B[0]))),
                                                         ), 0)), 1)

              B2B = np.concatenate((B2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                        cam(tensor2numpy(
                                                            fake_B2B_heatmap[0]), self.img_size),
                                                        RGB2BGR(tensor2numpy(
                                                            denorm(fake_B2B[0]))),
                                                         ), 0)), 1)
            else:
              loss_attn_A2B = self.patch_sampler(
                  self.generator(real_A, nce=True),
                  return_only_full_attn_maps=True,

              )
              loss_attn_B2B = self.patch_sampler(
                  self.generator(real_B, nce=True),
                  return_only_full_attn_maps=True,
              )
              A2B = np.concatenate((A2B, np.concatenate([
                  RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                  cam(tensor2numpy(
                      fake_A2B_heatmap[0]), self.img_size),
                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
              ] + [
                  cam(tensor2numpy(attn), self.img_size)
                  for attn in loss_attn_A2B
              ], 0)), 1)

              B2B = np.concatenate((B2B, np.concatenate([RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                         cam(tensor2numpy(
                                                             fake_B2B_heatmap[0]), self.img_size),
                                                         RGB2BGR(tensor2numpy(
                                                             denorm(fake_B2B[0])))
                                                         ] + [
                  cam(tensor2numpy(attn), self.img_size)
                  for attn in loss_attn_B2B
              ], 0)), 1)

          for _ in range(test_sample_num):
            try:
              real_A, _ = next(testA_iter)
            except:
              testA_iter = iter(self.testA_loader)
              real_A, _ = next(testA_iter)

            try:
              real_B, _ = next(testB_iter)
            except:
              testB_iter = iter(self.testB_loader)
              real_B, _ = next(testB_iter)
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.generator(real_A)

            fake_B2B, _, fake_B2B_heatmap = self.generator(real_B)

            if self.cut_type == 'vanilla':
              A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                        cam(tensor2numpy(
                                                            fake_A2B_heatmap[0]), self.img_size),
                                                        RGB2BGR(tensor2numpy(
                                                            denorm(fake_A2B[0]))),
                                                         ), 0)), 1)

              B2B = np.concatenate((B2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                        cam(tensor2numpy(
                                                            fake_B2B_heatmap[0]), self.img_size),
                                                        RGB2BGR(tensor2numpy(
                                                            denorm(fake_B2B[0]))),
                                                         ), 0)), 1)
            else:
              loss_attn_A2B = self.patch_sampler(
                  self.generator(real_A, nce=True),
                  return_only_full_attn_maps=True,

              )
              loss_attn_B2B = self.patch_sampler(
                  self.generator(real_B, nce=True),
                  return_only_full_attn_maps=True,
              )
              A2B = np.concatenate((A2B, np.concatenate([
                  RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                  cam(tensor2numpy(
                      fake_A2B_heatmap[0]), self.img_size),
                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
              ] + [
                  cam(tensor2numpy(attn), self.img_size)
                  for attn in loss_attn_A2B
              ], 0)), 1)

              B2B = np.concatenate((B2B, np.concatenate([RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                         cam(tensor2numpy(
                                                             fake_B2B_heatmap[0]), self.img_size),
                                                         RGB2BGR(tensor2numpy(
                                                             denorm(fake_B2B[0])))
                                                         ] + [
                  cam(tensor2numpy(attn), self.img_size)
                  for attn in loss_attn_B2B
              ], 0)), 1)

        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'A2B_%07d.png' % step), A2B * 255.0)
        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'B2B_%07d.png' % step), B2B * 255.0)

        self.generator.train(),
        self.global_discriminator.train(),
        self.local_discriminator.train(),
        self.patch_sampler.train()

      if step % self.save_freq == 0:
        self.save(ckpt_file_name='params_%07d.pt' % step)

      if step % 1000 == 0:
        self.save(ckpt_file_name='params_latest.pt')

      if step % self.val_freq == 0:
        self.val(step)

  def save(self, ckpt_file_name: str):
    params = {}
    params['generator'] = self.generator.state_dict()
    params['global_discriminator'] = self.global_discriminator.state_dict()
    params['local_discriminator'] = self.local_discriminator.state_dict()
    params['patch_sampler'] = self.patch_sampler.state_dict()
    torch.save(
        params,
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            ckpt_file_name
        )
    )

  def load(self, cktpt_file_name: str):
    params = torch.load(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            cktpt_file_name
        )
    )
    self.generator.load_state_dict(params['generator'])
    self.global_discriminator.load_state_dict(params['global_discriminator'])
    self.local_discriminator.load_state_dict(params['local_discriminator'])
    # initialize patch sampler
    x = torch.ones(
        (self.batch_size, self.img_ch, self.img_size, self.img_size),
        device=self.device
    )
    self.generator(x, nce=True)
    self.patch_sampler(self.generator(x, nce=True))
    self.patch_sampler.load_state_dict(params['patch_sampler'])

  def val(self, step: int):

    model_val_translations_dir = Path(self.result_dir, self.dataset, 'val')
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    model_with_step_translations_dir = Path(
        model_val_translations_dir,
        'params_%07d' % step
    )

    if not os.path.exists(model_with_step_translations_dir):
      os.mkdir(model_with_step_translations_dir)

    self.generator.eval()

    print('translating val...')
    for n, (real_A, _) in enumerate(self.valA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.valA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      fake_A2B, _, _ = self.generator(real_A)
      cv2.imwrite(
          os.path.join(model_with_step_translations_dir,
                       f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
    # compute metrics
    target_real_val_dir = os.path.join(
        'dataset',
        self.dataset,
        'valB'
    )
    metrics = torch_fidelity.calculate_metrics(
        input1=str(target_real_val_dir),
        input2=str(Path(model_with_step_translations_dir)),  # fake dir,
        isc=True,
        fid=True,
        verbose=False,
        rng_seed=self.seed,
        cuda=torch.cuda.is_available(),
    )
    with open(os.path.join(Path(self.result_dir, self.dataset), 'val_log.txt'), 'a') as tl:
      tl.write(f'step: {step}\n')
      tl.write(
          f'inception_score_mean: {metrics["inception_score_mean"]}\n'
      )
      tl.write(
          f'inception_score_std: {metrics["inception_score_std"]}\n',
      )
      tl.write(
          f'frechet_inception_distance: {metrics["frechet_inception_distance"]}\n'
      )

    if metrics['frechet_inception_distance'] < self.smallest_val_fid:
      self.smallest_val_fid = metrics['frechet_inception_distance']
      self.save(ckpt_file_name=f'params_smallest_val_fid.pt')
      smallest_val_fid_file = Path(
          self.result_dir,
          self.dataset, 'smallest_val_fid.txt'
      )
      if os.path.exists(smallest_val_fid_file):
        os.remove(smallest_val_fid_file)
      with open(smallest_val_fid_file, 'a') as tl:
        tl.write(
            f'step: {step}\n'
        )
        tl.write(
            f'frechet_inception_distance: {metrics["frechet_inception_distance"]}\n'
        )

  def test(self):
    model_list = glob(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            '*.pt'
        )
    )
    if not len(model_list) == 0:
      assert self.ckpt
      self.load(cktpt_file_name=self.ckpt)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    self.generator.eval()
    for n, (real_A, _) in enumerate(self.testA_loader):
      real_A = real_A.to(self.device)

      fake_A2B, _, fake_A2B_heatmap = self.generator(real_A)

      A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                            cam(tensor2numpy(
                                fake_A2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                            ), 0)

      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

    for n, (real_B, _) in enumerate(self.testB_loader):
      real_B = real_B.to(self.device)

      fake_B2B, _, fake_B2B_heatmap = self.generator(real_B)

      B2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                            cam(tensor2numpy(
                                fake_B2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                            ), 0)

      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'test', 'B2B_%d.png' % (n + 1)), B2B * 255.0)

  def translate(self):
    model_list = glob(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            '*.pt'
        )
    )
    if not len(model_list) == 0:
      self.load(cktpt_file_name=self.ckpt)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    if not os.path.exists('translations'):
      os.mkdir('translations')

    model_translations_dir = Path('translations', self.dataset)
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)

    model_with_ckpt_translations_dir = Path(
        model_translations_dir,
        Path(self.ckpt).stem
    )
    if not os.path.exists(model_with_ckpt_translations_dir):
      os.mkdir(model_with_ckpt_translations_dir)

    train_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'train')
    if self.valA:
      val_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'val')
    test_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'test')
    full_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'full')

    if not os.path.exists(train_translated_imgs_dir):
      os.mkdir(train_translated_imgs_dir)
    if not os.path.exists(test_translated_imgs_dir):
      os.mkdir(test_translated_imgs_dir)
    if val_translated_imgs_dir and not os.path.exists(val_translated_imgs_dir):
      os.mkdir(val_translated_imgs_dir)
    if not os.path.exists(full_translated_imgs_dir):
      os.mkdir(full_translated_imgs_dir)

    self.generator.eval()

    print('translating train...')
    for n, (real_A, _) in enumerate(self.trainA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.trainA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      fake_A2B, _, _ = self.generator(real_A)
      print(os.path.join(train_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
      cv2.imwrite(
          os.path.join(train_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
      cv2.imwrite(
          os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
    if val_translated_imgs_dir:
      print('translating val...')
      for n, (real_A, _) in enumerate(self.valA_loader):
        real_A = real_A.to(self.device)
        img_path, _ = self.valA_loader.dataset.samples[n]
        img_name = Path(img_path).name.split('.')[0]
        fake_A2B, _, _ = self.generator(real_A)
        print(os.path.join(val_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
        cv2.imwrite(
            os.path.join(val_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
        cv2.imwrite(
            os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
    print('translating test...')
    for n, (real_A, _) in enumerate(self.testA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.testA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      fake_A2B, _, _ = self.generator(real_A)
      print(os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
      cv2.imwrite(
          os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
      cv2.imwrite(
          os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
