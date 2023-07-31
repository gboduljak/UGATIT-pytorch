import itertools
import os
import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch_fidelity
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageFolder
from networks import *
from seed import get_seeded_generator, seed_everything, seeded_worker_init_fn
from utils import *


class UGATIT(object):
  def __init__(self, args):
    self.light = args.light

    if self.light:
      self.model_name = 'UGATIT_light'
    else:
      self.model_name = 'UGATIT'

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
    self.cycle_weight = args.cycle_weight
    self.identity_weight = args.identity_weight
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

    if torch.backends.cudnn.enabled and self.benchmark_flag:
      print('set benchmark !')
      torch.backends.cudnn.benchmark = True

    print()

    print("##### Information #####")
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
    print("# cycle_weight : ", self.cycle_weight)
    print("# identity_weight : ", self.identity_weight)
    print("# cam_weight : ", self.cam_weight)

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

    self.trainA = ImageFolder(
        os.path.join('dataset', self.dataset, 'trainA'),
        train_transform
    )
    self.trainA_no_aug = ImageFolder(
        os.path.join('dataset', self.dataset, 'trainA'),
        test_transform
    )
    self.trainB = ImageFolder(
        os.path.join('dataset', self.dataset, 'trainB'),
        train_transform
    )
    self.valA = ImageFolder(
        os.path.join('dataset', self.dataset, 'valA'),
        test_transform
    )
    self.valB = ImageFolder(
        os.path.join('dataset', self.dataset, 'valB'),
        test_transform
    )
    self.testA = ImageFolder(
        os.path.join('dataset', self.dataset, 'testA'),
        test_transform
    )
    self.testB = ImageFolder(
        os.path.join('dataset', self.dataset, 'testB'),
        test_transform
    )
    self.trainA_loader = DataLoader(
        self.trainA,
        batch_size=self.batch_size,
        worker_init_fn=seeded_worker_init_fn,
        generator=trainA_dataloader_generator,
        shuffle=True
    )
    self.trainA_no_aug_loader = DataLoader(
        self.trainA_no_aug,
        batch_size=1,
        shuffle=False
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
    self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                  n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
    self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                  n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
    self.disGA = Discriminator(
        input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
    self.disGB = Discriminator(
        input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
    self.disLA = Discriminator(
        input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
    self.disLB = Discriminator(
        input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
    print('Generator:')
    print(self.genA2B)
    print(f'total params: {get_total_model_params(self.genA2B)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.genA2B)}'
    )
    print('Global Discriminator:')
    print(self.disGB)
    print(f'total params: {get_total_model_params(self.disGB)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.disGB)}'
    )
    print('Local Discriminator:')
    print(self.disLB)
    print(f'total params: {get_total_model_params(self.disLB)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.disLB)}'
    )

    """ Define Loss """
    self.L1_loss = nn.L1Loss().to(self.device)
    self.MSE_loss = nn.MSELoss().to(self.device)
    self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

    """ Trainer """
    self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(
    ), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(
    ), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

    """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
    self.Rho_clipper = RhoClipper(0, 1)

  def train(self):
    self.genA2B.train(), self.genB2A.train(), self.disGA.train(
    ), self.disGB.train(), self.disLA.train(), self.disLB.train()

    start_iter = 1
    self.smallest_val_fid = float('inf')
    if self.resume:
      model_list = glob(os.path.join(
          self.result_dir, self.dataset, 'model', '*.pt'))
      if not len(model_list) == 0:
        model_list.sort()
        start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
        self.load(ckpt='iter_%07d.pt' % start_iter)
        print(" [*] Load SUCCESS")
        if self.decay_flag and start_iter > (self.iteration // 2):
          self.G_optim.param_groups[0]['lr'] -= (self.lr / (
              self.iteration // 2)) * (start_iter - self.iteration // 2)
          self.D_optim.param_groups[0]['lr'] -= (self.lr / (
              self.iteration // 2)) * (start_iter - self.iteration // 2)

    # training loop
    print('training start !')
    start_time = time.time()
    for step in range(start_iter, self.iteration + 1):
      if self.decay_flag and step > (self.iteration // 2):
        self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
        self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

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

      fake_A2B, _, _ = self.genA2B(real_A)
      fake_B2A, _, _ = self.genB2A(real_B)

      real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
      real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
      real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
      real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

      fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
      fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
      fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

      D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(
          self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
      D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(
          self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
      D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(
          self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
      D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(
          self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
      D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(
          self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
      D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(
          self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
      D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(
          self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
      D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(
          self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

      D_loss_A = self.adv_weight * \
          (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
      D_loss_B = self.adv_weight * \
          (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

      Discriminator_loss = D_loss_A + D_loss_B
      Discriminator_loss.backward()
      self.D_optim.step()

      # Update G
      self.G_optim.zero_grad()

      fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
      fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

      fake_A2B2A, _, _ = self.genB2A(fake_A2B)
      fake_B2A2B, _, _ = self.genA2B(fake_B2A)

      fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
      fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

      fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
      fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
      fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

      G_ad_loss_GA = self.MSE_loss(
          fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
      G_ad_cam_loss_GA = self.MSE_loss(
          fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
      G_ad_loss_LA = self.MSE_loss(
          fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
      G_ad_cam_loss_LA = self.MSE_loss(
          fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
      G_ad_loss_GB = self.MSE_loss(
          fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
      G_ad_cam_loss_GB = self.MSE_loss(
          fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
      G_ad_loss_LB = self.MSE_loss(
          fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
      G_ad_cam_loss_LB = self.MSE_loss(
          fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

      G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
      G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

      G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
      G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

      G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(
          self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
      G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(
          self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

      G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
          self.cycle_weight * G_recon_loss_A + self.identity_weight * \
          G_identity_loss_A + self.cam_weight * G_cam_loss_A
      G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
          self.cycle_weight * G_recon_loss_B + self.identity_weight * \
          G_identity_loss_B + self.cam_weight * G_cam_loss_B

      Generator_loss = G_loss_A + G_loss_B
      Generator_loss.backward()
      self.G_optim.step()

      # clip parameter of AdaILN and ILN, applied after optimizer step
      self.genA2B.apply(self.Rho_clipper)
      self.genB2A.apply(self.Rho_clipper)

      loss_line = "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step,
                                                                        self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss)

      print(loss_line)
      with open(os.path.join(self.result_dir, self.dataset, 'loss_log.txt'), 'a') as tl:
        tl.write(f'{loss_line}\n')

      if step % self.print_freq == 0:
        train_sample_num = 5
        test_sample_num = 5
        A2B = np.zeros((self.img_size * 7, 0, 3))
        B2A = np.zeros((self.img_size * 7, 0, 3))

        self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(
        ), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
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

          fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
          fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

          fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
          fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

          fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
          fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

          A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                     cam(tensor2numpy(
                                                         fake_A2A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2A[0]))),
                                                     cam(tensor2numpy(
                                                         fake_A2B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2B[0]))),
                                                     cam(tensor2numpy(
                                                         fake_A2B2A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

          B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                     cam(tensor2numpy(
                                                         fake_B2B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2B[0]))),
                                                     cam(tensor2numpy(
                                                         fake_B2A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2A[0]))),
                                                     cam(tensor2numpy(
                                                         fake_B2A2B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

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

          fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
          fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

          fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
          fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

          fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
          fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

          A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                     cam(tensor2numpy(
                                                         fake_A2A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2A[0]))),
                                                     cam(tensor2numpy(
                                                         fake_A2B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2B[0]))),
                                                     cam(tensor2numpy(
                                                         fake_A2B2A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

          B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                     cam(tensor2numpy(
                                                         fake_B2B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2B[0]))),
                                                     cam(tensor2numpy(
                                                         fake_B2A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2A[0]))),
                                                     cam(tensor2numpy(
                                                         fake_B2A2B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'A2B_%07d.png' % step), A2B * 255.0)
        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'B2A_%07d.png' % step), B2A * 255.0)
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(
        ), self.disGB.train(), self.disLA.train(), self.disLB.train()

      if step % self.save_freq == 0:
        self.save(ckpt_file_name='iter_%07d.pt' % step)

      if step % 1000 == 0:
        self.save(ckpt_file_name='latest.pt')

      if step % self.val_freq == 0:
        self.val(step)

  def save(self, ckpt_file_name: str):
    params = {}
    params['genA2B'] = self.genA2B.state_dict()
    params['genB2A'] = self.genB2A.state_dict()
    params['disGA'] = self.disGA.state_dict()
    params['disGB'] = self.disGB.state_dict()
    params['disLA'] = self.disLA.state_dict()
    params['disLB'] = self.disLB.state_dict()
    torch.save(
        params,
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            ckpt_file_name
        )
    )

  def load(self, ckpt):
    params = torch.load(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            ckpt
        )
    )
    self.genA2B.load_state_dict(params['genA2B'])
    self.genB2A.load_state_dict(params['genB2A'])
    self.disGA.load_state_dict(params['disGA'])
    self.disGB.load_state_dict(params['disGB'])
    self.disLA.load_state_dict(params['disLA'])
    self.disLB.load_state_dict(params['disLB'])

  def val(self, step: int):
    results_dir = Path(self.result_dir, self.dataset)

    model_translations_dir = Path(self.result_dir, self.dataset, 'translations')
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)
    model_iter_translations_dir = Path(
        model_translations_dir,
        'iter_%07d' % step
    )

    if not os.path.exists(model_iter_translations_dir):
      os.mkdir(model_iter_translations_dir)

    model_train_translations_dir = Path(model_iter_translations_dir, 'train')
    model_val_translations_dir = Path(model_iter_translations_dir, 'val')

    if not os.path.exists(model_train_translations_dir):
      os.mkdir(model_train_translations_dir)
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    if not os.path.exists(model_train_translations_dir):
      os.mkdir(model_train_translations_dir)
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    self.genA2B.eval()
    with torch.no_grad():
      print('translating train...')
      for n, (real_A, _) in enumerate(self.trainA_no_aug_loader):
        real_A = real_A.to(self.device)
        img_path, _ = self.trainA_no_aug_loader.dataset.samples[n]
        img_name = Path(img_path).name.split('.')[0]
        fake_A2B, _, _ = self.genA2B(real_A)
        cv2.imwrite(
            os.path.join(
                model_train_translations_dir,
                f'{img_name}_fake_B.jpg'
            ),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
      print('translating val...')
      for n, (real_A, _) in enumerate(self.valA_loader):
        real_A = real_A.to(self.device)
        img_path, _ = self.valA_loader.dataset.samples[n]
        img_name = Path(img_path).name.split('.')[0]
        fake_A2B, _, _ = self.genA2B(real_A)
        cv2.imwrite(
            os.path.join(
                model_val_translations_dir,
                f'{img_name}_fake_B.jpg'
            ),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
    # compute metrics
    target_real_train_dir = os.path.join(
        'dataset',
        self.dataset,
        'trainB'
    )
    target_real_val_dir = os.path.join(
        'dataset',
        self.dataset,
        'valB'
    )
    train_metrics = torch_fidelity.calculate_metrics(
        input1=str(target_real_train_dir),
        input2=str(Path(model_train_translations_dir)),  # fake dir
        fid=True,
        verbose=False,
        cuda=torch.cuda.is_available(),
    )
    val_metrics = torch_fidelity.calculate_metrics(
        input1=str(target_real_val_dir),
        input2=str(Path(model_val_translations_dir)),  # fake dir,
        fid=True,
        verbose=False,
        rng_seed=self.seed,
        cuda=torch.cuda.is_available(),
    )

    train_log_file = os.path.join(results_dir, 'train_log.txt')
    val_log_file = os.path.join(results_dir, 'val_log.txt')
    smallest_val_fid_file = os.path.join(results_dir, 'smallest_val_fid.txt')

    with open(train_log_file, 'a') as tl:
      tl.write(f'iter: {step}\n')
      tl.write(
          f'frechet_inception_distance: {train_metrics["frechet_inception_distance"]}\n'
      )

    with open(val_log_file, 'a') as vl:
      vl.write(f'iter: {step}\n')
      vl.write(
          f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
      )

    if val_metrics['frechet_inception_distance'] < self.smallest_val_fid:
      self.smallest_val_fid = val_metrics['frechet_inception_distance']
      print(
          f'{self.smallest_val_fid} is the smallest val fid so far, saving this model...'
      )
      self.save(ckpt_file_name='smallest_val_fid.pt')

      if os.path.exists(smallest_val_fid_file):
        os.remove(smallest_val_fid_file)

      with open(smallest_val_fid_file, 'a') as tl:
        tl.write(
            f'iter: {step}\n'
        )
        tl.write(
            f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
        )

    self.genA2B.train()

  def test(self):
    model_list = glob(os.path.join(
        self.result_dir, self.dataset, 'model', '*.pt'
    ))
    if not len(model_list) == 0:
      model_list.sort()
      iter = int(model_list[-1].split('_')[-1].split('.')[0])
      self.load(ckpt='iter_%07d.pt' % iter)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    self.genA2B.eval(), self.genB2A.eval()
    for n, (real_A, _) in enumerate(self.testA_loader):
      real_A = real_A.to(self.device)

      fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

      fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

      fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

      A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                            cam(tensor2numpy(
                                fake_A2A_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                            cam(tensor2numpy(
                                fake_A2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                            cam(tensor2numpy(
                                fake_A2B2A_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

    for n, (real_B, _) in enumerate(self.testB_loader):
      real_B = real_B.to(self.device)

      fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

      fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

      fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

      B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                            cam(tensor2numpy(
                                fake_B2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                            cam(tensor2numpy(
                                fake_B2A_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                            cam(tensor2numpy(
                                fake_B2A2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)

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
      model_list.sort()
      self.load(self.ckpt)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    if not os.path.exists('translations'):
      os.mkdir('translations')

    model_translations_dir = Path('translations', self.dataset)
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)
    model_with_ckpts_translations_dir = Path(
        model_translations_dir,
        Path(self.ckpt).stem
    )
    if not os.path.exists(model_with_ckpts_translations_dir):
      os.mkdir(model_with_ckpts_translations_dir)

    train_translated_imgs_dir = Path(model_with_ckpts_translations_dir, 'train')
    if self.valA:
      val_translated_imgs_dir = Path(model_with_ckpts_translations_dir, 'val')
    test_translated_imgs_dir = Path(model_with_ckpts_translations_dir, 'test')
    full_translated_imgs_dir = Path(model_with_ckpts_translations_dir, 'full')

    if not os.path.exists(train_translated_imgs_dir):
      os.mkdir(train_translated_imgs_dir)
    if not os.path.exists(test_translated_imgs_dir):
      os.mkdir(test_translated_imgs_dir)
    if val_translated_imgs_dir and not os.path.exists(val_translated_imgs_dir):
      os.mkdir(val_translated_imgs_dir)
    if not os.path.exists(full_translated_imgs_dir):
      os.mkdir(full_translated_imgs_dir)

    self.genA2B.eval(), self.genB2A.eval()
    with torch.no_grad():
      print('translating train...')
      for n, (real_A, _) in enumerate(self.trainA_no_aug_loader):
        real_A = real_A.to(self.device)
        img_path, _ = self.trainA_no_aug_loader.dataset.samples[n]
        img_name = Path(img_path).name.split('.')[0]
        fake_A2B, _, _ = self.genA2B(real_A)
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
          fake_A2B, _, _ = self.genA2B(real_A)
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
        fake_A2B, _, _ = self.genA2B(real_A)
        print(os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
        cv2.imwrite(
            os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
        cv2.imwrite(
            os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
