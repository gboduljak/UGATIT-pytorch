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
from pnce import PatchNCELoss
from psample import PatchSampleF
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

    """ CUT """
    self.nce_weight = args.nce_weight
    self.nce_temperature = args.nce_temperature
    self.nce_net_nc = args.nce_net_nc
    self.nce_n_patches = args.nce_n_patches

    if torch.backends.cudnn.enabled and self.benchmark_flag:
      print('set benchmark !')
      torch.backends.cudnn.benchmark = True

    print()

    print("##### Information #####")
    print("# cut : ", self.cut)
    print("# light : ", self.light)
    print("# dataset : ", self.dataset)
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
    print("# nce_weight : ", self.nce_weight)

    print("##### CUT #####")
    print("# nce weight : ", self.nce_weight)
    print("# nce temperature : ", self.nce_temperature)
    print("# nce patches : ", self.nce_n_patches)
    print("# nce net dim : ", self.nce_net_nc)

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
    self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                  n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
    self.disGB = Discriminator(
        input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
    self.disLB = Discriminator(
        input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

    """ Define F """
    self.netF = PatchSampleF(
        use_mlp=True,
        init_type='normal',
        init_gain=0.02,
        nc=self.nce_net_nc,
        device=self.device
    )

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
    self.NCE_losses = []

    self.nce_layers = [0, 5, 9, 12, 16]  # hardcoded for now
    for _ in self.nce_layers:
      self.NCE_losses.append(
          PatchNCELoss(
              temperature=self.nce_temperature,
              batch_size=self.batch_size
          ).to(self.device)
      )

    """ Trainer """
    self.G_optim = torch.optim.Adam(
        itertools.chain(self.genA2B.parameters()),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    self.D_optim = torch.optim.Adam(
        itertools.chain(
            self.disGB.parameters(),
            self.disLB.parameters()
        ),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    self.F_optim = None  # not initialized now
    """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
    self.Rho_clipper = RhoClipper(0, 1)

  def calculate_weighted_NCE_loss(self, src, tgt):
    n_layers = len(self.nce_layers)
    _, _, _, feat_q = self.genA2B(tgt, nce=True)
    _, _, _, feat_k = self.genA2B(src, nce=True)

    should_init_optimizer = not self.netF.mlp_init

    feat_k_pool, sample_ids = self.netF(feat_k, self.nce_n_patches, None)
    feat_q_pool, _ = self.netF(feat_q, self.nce_n_patches, sample_ids)

    if should_init_optimizer:
      print('once')
      self.F_optim = torch.optim.Adam(
          self.netF.parameters(),
          lr=self.lr,
          betas=(0.5, 0.999),
          weight_decay=self.weight_decay
      )

    total_nce_loss = 0.0
    for f_q, f_k, pnce, _ in zip(feat_q_pool, feat_k_pool, self.NCE_losses, self.nce_layers):
      loss = self.nce_weight * pnce(f_q, f_k)
      total_nce_loss += loss.mean()

    return total_nce_loss / n_layers

  def train(self):
    self.genA2B.train(), self.disGB.train(),  self.disLB.train()

    start_iter = 1
    if self.resume:
      model_list = glob(os.path.join(
          self.result_dir, self.dataset, 'model', '*.pt'))
      if not len(model_list) == 0:
        model_list.sort()
        start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
        self.load(os.path.join(self.result_dir,
                  self.dataset, 'model'), start_iter)
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

      # real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
      real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

      # fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

      # D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(
      #     self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
      # D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(
      #     self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
      D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(
          self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
      D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(
          self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

      # Discriminator_loss = self.adv_weight * \
      #     (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
      Discriminator_loss = self.adv_weight * (D_ad_loss_LB + D_ad_cam_loss_LB)
      Discriminator_loss.backward()
      self.D_optim.step()

      # Update G
      self.G_optim.zero_grad()
      if self.F_optim:
        self.F_optim.zero_grad()

      fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
      fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

      fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

      # G_ad_loss_GB = self.MSE_loss(
      #     fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
      # G_ad_cam_loss_GB = self.MSE_loss(
      #     fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
      G_ad_loss_LB = self.MSE_loss(
          fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
      G_ad_cam_loss_LB = self.MSE_loss(
          fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

      G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(
          self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

      # this is where NCE goes
      loss_NCE_X = self.calculate_weighted_NCE_loss(real_A, fake_A2B)
      loss_NCE_Y = self.calculate_weighted_NCE_loss(real_B, fake_B2B)
      loss_NCE_both = (loss_NCE_X + loss_NCE_Y) * 0.5

      Generator_loss = self.adv_weight * (
          # G_ad_loss_GB +
          # G_ad_cam_loss_GB +
          G_ad_loss_LB +
          G_ad_cam_loss_LB
      ) + loss_NCE_both + self.cam_weight * G_cam_loss_B

      Generator_loss.backward()
      self.G_optim.step()
      if self.F_optim:
        self.F_optim.step()

      # clip parameter of AdaILN and ILN, applied after optimizer step
      self.genA2B.apply(self.Rho_clipper)

      train_status_line = "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, nce_loss: %.8f, nce_x: %.8f, nce_y: %.8f" % (
          step,
          self.iteration,
          time.time() - start_time,
          Discriminator_loss,
          Generator_loss,
          loss_NCE_both,
          loss_NCE_X,
          loss_NCE_Y
      )
      print(train_status_line)
      with open(os.path.join(self.result_dir, self.dataset, 'train_log.txt'), 'a') as tl:
        tl.write(f'{train_status_line}\n')

      if step % self.print_freq == 0:
        train_sample_num = 5
        test_sample_num = 5
        A2B = np.zeros((self.img_size * 3, 0, 3))
        B2B = np.zeros((self.img_size * 3, 0, 3))

        self.genA2B.eval(),  self.disGB.eval(), self.disLB.eval()
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
          fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

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

          fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

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

        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'A2B_%07d.png' % step), A2B * 255.0)
        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'B2B_%07d.png' % step), B2B * 255.0)
        self.genA2B.train(), self.disGB.train(),  self.disLB.train()

      if step % self.save_freq == 0:
        self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

      if step % 1000 == 0:
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(self.result_dir,
                   self.dataset + '_params_latest.pt'))

      if step % self.val_freq == 0:
        self.val(step)

  def save(self, dir, step):
    params = {}
    params['genA2B'] = self.genA2B.state_dict()
    params['disGB'] = self.disGB.state_dict()
    params['disLB'] = self.disLB.state_dict()
    torch.save(params, os.path.join(
        dir, self.dataset + '_params_%07d.pt' % step))

  def load(self, dir, step):
    params = torch.load(os.path.join(
        dir, self.dataset + '_params_%07d.pt' % step))
    self.genA2B.load_state_dict(params['genA2B'])
    self.disGB.load_state_dict(params['disGB'])
    self.disLB.load_state_dict(params['disLB'])

  def val(self, step: int):

    model_val_translations_dir = Path(self.result_dir, self.dataset, 'val')
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    model_with_step_translations_dir = Path(
        model_val_translations_dir,
        f'step-{step}'
    )

    if not os.path.exists(model_with_step_translations_dir):
      os.mkdir(model_with_step_translations_dir)

    self.genA2B.eval()

    print('translating val...')
    for n, (real_A, _) in enumerate(self.valA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.valA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      fake_A2B, _, _ = self.genA2B(real_A)
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
    with open(os.path.join(model_val_translations_dir, 'val_log.txt'), 'a') as tl:
      tl.write(f'step:{step}\n')
      tl.write(
          f'inception_score_mean: {metrics["inception_score_mean"]}\n'
      )
      tl.write(
          f'inception_score_std: {metrics["inception_score_std"]}\n',
      )
      tl.write(
          f'frechet_inception_distance: {metrics["frechet_inception_distance"]}\n'
      )

  def test(self):
    model_list = glob(os.path.join(
        self.result_dir, self.dataset, 'model', '*.pt'))
    if not len(model_list) == 0:
      model_list.sort()
      iter = int(model_list[-1].split('_')[-1].split('.')[0])
      self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    self.genA2B.eval()
    for n, (real_A, _) in enumerate(self.testA_loader):
      real_A = real_A.to(self.device)

      fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

      A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                            cam(tensor2numpy(
                                fake_A2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                            ), 0)

      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

    for n, (real_B, _) in enumerate(self.testB_loader):
      real_B = real_B.to(self.device)

      fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

      B2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                            cam(tensor2numpy(
                                fake_B2B_heatmap[0]), self.img_size),
                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                            ), 0)

      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'test', 'B2B_%d.png' % (n + 1)), B2B * 255.0)

  def translate(self):
    model_list = glob(os.path.join(
        self.result_dir, self.dataset, 'model', '*.pt'))
    if not len(model_list) == 0:
      model_list.sort()
      self.load(os.path.join(self.result_dir,
                self.dataset, 'model'), self.iteration)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    if not os.path.exists('translations'):
      os.mkdir('translations')

    model_translations_dir = Path('translations', self.dataset)
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)

    model_with_iters_translations_dir = Path(
        model_translations_dir, f'iter-{self.iteration}'
    )
    if not os.path.exists(model_with_iters_translations_dir):
      os.mkdir(model_with_iters_translations_dir)

    train_translated_imgs_dir = Path(model_with_iters_translations_dir, 'train')
    if self.valA:
      val_translated_imgs_dir = Path(model_with_iters_translations_dir, 'val')
    test_translated_imgs_dir = Path(model_with_iters_translations_dir, 'test')
    full_translated_imgs_dir = Path(model_with_iters_translations_dir, 'full')

    if not os.path.exists(train_translated_imgs_dir):
      os.mkdir(train_translated_imgs_dir)
    if not os.path.exists(test_translated_imgs_dir):
      os.mkdir(test_translated_imgs_dir)
    if val_translated_imgs_dir and not os.path.exists(val_translated_imgs_dir):
      os.mkdir(val_translated_imgs_dir)
    if not os.path.exists(full_translated_imgs_dir):
      os.mkdir(full_translated_imgs_dir)

    self.genA2B.eval()

    print('translating train...')
    for n, (real_A, _) in enumerate(self.trainA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.trainA_loader.dataset.samples[n]
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
