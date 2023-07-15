import itertools
import time
from glob import glob
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from modules.dann import DANN
from networks import *
from utils import *


class UGATIT(object):
  def __init__(self, args):

    self.model_name = 'UGATIT_light Pipeline'
    self.result_dir = args.result_dir
    self.dataset = args.dataset
    self.n_classes = args.n_classes

    self.iteration = args.iteration
    self.decay_flag = args.decay_flag

    self.batch_size = args.batch_size
    self.print_freq = args.print_freq
    self.save_freq = args.save_freq

    self.lr = args.lr
    self.weight_decay = args.weight_decay
    self.ch = args.ch

    """ Weight """
    self.adv_weight = args.adv_weight
    self.cycle_weight = args.cycle_weight
    self.identity_weight = args.identity_weight
    self.cam_weight = args.cam_weight
    self.class_weight = args.class_weight

    """ Generator """
    self.n_res = args.n_res

    """ Discriminator """
    self.n_dis = args.n_dis

    self.img_size = args.img_size
    self.img_ch = args.img_ch

    self.device = args.device
    self.benchmark_flag = args.benchmark_flag
    self.resume = args.resume

    if torch.backends.cudnn.enabled and self.benchmark_flag:
      print('set benchmark !')
      torch.backends.cudnn.benchmark = True

    print()

    print("##### Information #####")
    print("# dataset : ", self.dataset)
    print("# batch_size : ", self.batch_size)
    print("# iteration per epoch : ", self.iteration)

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
    print("# class_weight : ", self.class_weight)

  ##################################################################################
  # Model
  ##################################################################################

  def build_model(self):
    """ DataLoader """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    source_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    source_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    target_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((self.img_size + 30, self.img_size+30)),
        transforms.RandomCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    target_test_transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    self.source_train = ImageFolder(
        os.path.join('dataset', self.dataset, 'source', 'train'),
        source_train_transform
    )
    self.source_test = ImageFolder(
        os.path.join('dataset', self.dataset, 'source', 'test'),
        source_test_transform
    )
    self.target_train = ImageFolder(
        os.path.join('dataset', self.dataset, 'target', 'train'),
        target_train_transform
    )
    self.target_test = ImageFolder(
        os.path.join('dataset', self.dataset, 'target', 'test'),
        target_test_transform
    )
    self.source_train_loader = DataLoader(
        self.source_train,
        batch_size=self.batch_size,
        shuffle=True
    )
    self.source_test_loader = DataLoader(
        self.source_test,
        batch_size=self.batch_size,
        shuffle=True
    )
    self.target_train_loader = DataLoader(
        self.target_train,
        batch_size=self.batch_size,
        shuffle=True
    )
    self.target_test_loader = DataLoader(
        self.target_test,
        batch_size=1,
        shuffle=False
    )
    """Prepare DANN"""
    self.dann = DANN.load_from_checkpoint(
        'dann-val_tgt_cls_acc=0.9364162087440491.ckpt',
        map_location=self.device
    )
    self.dann.eval()

    """ Define Generator, Discriminator """
    self.g = Generator(
        input_nc=3,
        output_nc=3,
        ngf=self.ch,
        n_blocks=self.n_res,
        n_classes=self.n_classes,
        img_size=self.img_size
    ).to(self.device)
    self.dis_global = Discriminator(
        input_nc=3,
        ndf=self.ch,
        n_layers=7,
        n_classes=self.n_classes,
    ).to(self.device)
    self.dis_local = Discriminator(
        input_nc=3,
        ndf=self.ch,
        n_layers=5,
        n_classes=self.n_classes,
    ).to(self.device)
    """ Define Loss """
    self.L1_loss = nn.L1Loss().to(self.device)
    self.MSE_loss = nn.MSELoss().to(self.device)
    self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
    """ Trainer """
    self.G_optim = torch.optim.Adam(
        self.g.parameters(),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    self.D_optim = torch.optim.Adam(
        itertools.chain(self.dis_global.parameters(),
                        self.dis_local.parameters()),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )

    """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
    self.Rho_clipper = RhoClipper(0, 1)

  def train(self):
    self.g.train(), self.dis_global.train(), self.dis_local.train()

    start_iter = 1
    if self.resume:
      model_list = glob(
          os.path.join(self.result_dir, self.dataset, 'model', '*.pt')
      )
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
        x_tgt_real, _ = next(target_train_iter)
      except:
        target_train_iter = iter(self.target_train_loader)
        x_tgt_real, _ = next(target_train_iter)

      try:
        x_src_real, _ = next(source_train_iter)
      except:
        source_train_iter = iter(self.source_train_loader)
        x_src_real, _ = next(source_train_iter)

      x_src_real = x_src_real.to(self.device)
      x_tgt_real = x_tgt_real.to(self.device)

      with torch.no_grad():
        f_src = self.dann.extract_features(x_src_real)
        f_tgt = self.dann.extract_features(x_tgt_real)
        x_src_real_logits = self.dann.classifier.head(f_src)
        x_src_class = torch.argmax(x_src_real_logits, dim=1).item()
        x_tgt_real_logits = self.dann.classifier.head(f_tgt)
        x_tgt_class = torch.argmax(x_tgt_real_logits, dim=1).item()

      # Update D
      self.D_optim.zero_grad()

      x_tgt_fake_labeled_src = self.g(x_tgt_real, f_src, x_src_class)
      x_tgt_fake_labeled_tgt = self.g(x_tgt_real, f_tgt, x_tgt_class)

      real_GB_logit, real_GB_cam_logit, _ = self.dis_global(
          x_tgt_real, x_tgt_class, cam=True)  # global
      real_LB_logit, real_LB_cam_logit, _ = self.dis_local(
          x_tgt_real, x_tgt_class, cam=True)  # local

      fake_GB_logit_1 = self.dis_global(
          x_tgt_fake_labeled_src, x_src_class
      )
      fake_GB_logit_2 = self.dis_global(
          x_tgt_fake_labeled_tgt, x_tgt_class
      )
      fake_LB_logit_1 = self.dis_local(
          x_tgt_fake_labeled_src, x_src_class
      )
      fake_LB_logit_2 = self.dis_local(
          x_tgt_fake_labeled_tgt, x_tgt_class
      )

      D_ad_loss_global = self.MSE_loss(
          real_GB_logit,
          torch.ones_like(real_GB_logit).to(self.device)
      ) + self.MSE_loss(
          fake_GB_logit_1,
          torch.zeros_like(fake_GB_logit_1).to(self.device)
      )
      + self.MSE_loss(
          fake_GB_logit_2,
          torch.zeros_like(fake_GB_logit_2).to(self.device)
      )
      cam_nets = 2

      D_cam_loss_global = self.MSE_loss(
          real_GB_cam_logit,
          x_src_real_logits.repeat((1, cam_nets))
      )

      D_ad_loss_local = self.MSE_loss(
          real_LB_logit,
          torch.ones_like(real_LB_logit).to(self.device)
      ) + self.MSE_loss(
          fake_LB_logit_1,
          torch.zeros_like(fake_LB_logit_1).to(self.device)
      ) + self.MSE_loss(
          fake_LB_logit_2,
          torch.zeros_like(fake_LB_logit_2).to(self.device)
      )

      D_cam_loss_local = self.MSE_loss(
          real_LB_cam_logit,
          x_src_real_logits.repeat((1, cam_nets))
      )

      Discriminator_loss = self.adv_weight * \
          (D_ad_loss_global + D_ad_loss_local +
           D_cam_loss_global + D_cam_loss_local)
      Discriminator_loss.backward()
      self.D_optim.step()

      # Update G
      self.G_optim.zero_grad()

      x_tgt_fake_labeled_src = self.g(
          x_tgt_real, f_src, x_src_class)
      x_tgt_fake_labeled_tgt = self.g(
          x_tgt_real, f_tgt, x_tgt_class)
      x_tgt_fake_reconstr = self.g(
          x_tgt_fake_labeled_src, f_tgt, x_tgt_class)

      real_GB_logit = self.dis_global(
          x_tgt_real, x_tgt_class)  # global
      real_LB_logit = self.dis_local(
          x_tgt_real, x_tgt_class)  # local

      fake_GB_logit_1 = self.dis_global(
          x_tgt_fake_labeled_src, x_src_class
      )
      fake_GB_logit_2 = self.dis_global(
          x_tgt_fake_labeled_tgt, x_tgt_class
      )
      fake_LB_logit_1 = self.dis_local(
          x_tgt_fake_labeled_src, x_src_class
      )
      fake_LB_logit_2 = self.dis_local(
          x_tgt_fake_labeled_tgt, x_tgt_class
      )

      G_ad_loss_global = self.MSE_loss(
          fake_GB_logit_1,
          torch.ones_like(fake_GB_logit_1).to(self.device)
      ) + self.MSE_loss(
          fake_GB_logit_2,
          torch.ones_like(fake_GB_logit_2).to(self.device)
      )
      G_ad_loss_local = self.MSE_loss(
          fake_LB_logit_1,
          torch.ones_like(fake_LB_logit_1).to(self.device)
      ) + self.MSE_loss(
          fake_LB_logit_2,
          torch.ones_like(fake_LB_logit_2).to(self.device)
      )

      cam_nets = 2

    #   G_cam_loss = self.MSE_loss(
    #       x_tgt_fake_labeled_src_cam_logits,
    #       x_src_real_logits.repeat((1, cam_nets))
    #   ) + self.MSE_loss(
    #       x_tgt_fake_labeled_tgt_cam_logits,
    #       x_tgt_real_logits.repeat((1, cam_nets))
    #   )

      G_recon = self.L1_loss(
          x_tgt_fake_reconstr,
          x_tgt_real
      )  # cycle consistency
      G_identity = self.L1_loss(
          x_tgt_fake_labeled_tgt,
          x_tgt_real
      )  # identity loss

      G_class = self.MSE_loss(
          self.dann(x_tgt_fake_labeled_src), x_src_real_logits
      )
      Generator_loss = self.adv_weight * (G_ad_loss_global + G_ad_loss_local) + \
          self.cycle_weight * G_recon + self.identity_weight * \
          100 * G_identity + self.class_weight * G_class

      Generator_loss.backward()
      self.G_optim.step()

      # clip parameter of AdaILN and ILN, applied after optimizer step
      self.g.apply(self.Rho_clipper)

      train_status_line = "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step,
                                                                                self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss)

      print(train_status_line)
      with open(os.path.join(self.result_dir, self.dataset, 'training_log.txt'), 'a') as tl:
        tl.write(f'{train_status_line}\n')

      if step % self.print_freq == 0:
        train_sample_num = 5
        test_sample_num = 5
        A2B = np.zeros((self.img_size * 4, 0, 3))

        self.g.eval(), self.dis_global.eval(), self.dis_local.eval()

        for _ in range(train_sample_num):
          try:
            x_tgt_real, _ = next(target_train_iter)
          except:
            target_train_iter = iter(self.target_train_loader)
            x_tgt_real, _ = next(target_train_iter)

          try:
            x_src_real, _ = next(source_train_iter)
          except:
            source_train_iter = iter(self.source_train_loader)
            x_src_real, _ = next(source_train_iter)

          x_src_real = x_src_real.to(self.device)
          x_tgt_real = x_tgt_real.to(self.device)
          t = transforms.Compose(
              [transforms.ToPILImage(),
               transforms.Resize((256, 256)),
               transforms.ToTensor()]
          )
          with torch.no_grad():
            f_src = self.dann.extract_features(x_src_real)
            f_tgt = self.dann.extract_features(x_tgt_real)
            x_src_real_logits = self.dann.classifier.head(f_src)
            x_src_class = torch.argmax(x_src_real_logits, dim=1).item()
            x_tgt_real_logits = self.dann.classifier.head(f_tgt)
            x_tgt_class = torch.argmax(x_tgt_real_logits, dim=1).item()

          x_tgt_fake_labeled_src = self.g(
              x_tgt_real, f_src, x_src_class)
          x_tgt_fake_labeled_tgt = self.g(
              x_tgt_real, f_tgt, x_tgt_class)

          A2B = np.concatenate((A2B, np.concatenate((
              RGB2BGR(tensor2numpy(denorm(x_tgt_real[0]))),
              RGB2BGR(tensor2numpy(denorm(t(x_src_real[0])))),
              #   cam(tensor2numpy(
              #       x_tgt_fake_labeled_tgt_heatmap[0]), self.img_size),
              RGB2BGR(tensor2numpy(
                  denorm(x_tgt_fake_labeled_tgt[0]))),
              #   cam(tensor2numpy(
              #       x_tgt_fake_labeled_src_heatmap[0]), self.img_size),
              RGB2BGR(tensor2numpy(denorm(x_tgt_fake_labeled_src[0])))), 0)), 1
          )

    #     for _ in range(test_sample_num):
    #       try:
    #         x_tgt_real, _ = next(testA_iter)
    #       except:
    #         testA_iter = iter(self.testA_loader)
    #         x_tgt_real, _ = next(testA_iter)

    #       try:
    #         real_B, _ = next(testB_iter)
    #       except:
    #         testB_iter = iter(self.testB_loader)
    #         real_B, _ = next(testB_iter)
    #       x_tgt_real, real_B = x_tgt_real.to(
    #           self.device), real_B.to(self.device)

    #       x_tgt_fake, _, x_tgt_fake_heatmap = self.g(x_tgt_real)

    #       fake_B2A2B, _, fake_B2A2B_heatmap = self.g(fake_B2A)

    #       fake_B2B, _, fake_B2B_heatmap = self.g(real_B)

    #       A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(x_tgt_real[0]))),
    #                                                  cam(tensor2numpy(
    #                                                      fake_A2A_heatmap[0]), self.img_size),
    #                                                  RGB2BGR(tensor2numpy(
    #                                                      denorm(fake_A2A[0]))),
    #                                                  cam(tensor2numpy(
    #                                                      x_tgt_fake_heatmap[0]), self.img_size),
    #                                                  RGB2BGR(tensor2numpy(
    #                                                      denorm(x_tgt_fake[0]))),
    #                                                  cam(tensor2numpy(
    #                                                      x_tgt_fake2A_heatmap[0]), self.img_size),
    #                                                  RGB2BGR(tensor2numpy(denorm(x_tgt_fake2A[0])))), 0)), 1)

    #       B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
    #                                                  cam(tensor2numpy(
    #                                                      fake_B2B_heatmap[0]), self.img_size),
    #                                                  RGB2BGR(tensor2numpy(
    #                                                      denorm(fake_B2B[0]))),
    #                                                  cam(tensor2numpy(
    #                                                      fake_B2A_heatmap[0]), self.img_size),
    #                                                  RGB2BGR(tensor2numpy(
    #                                                      denorm(fake_B2A[0]))),
    #                                                  cam(tensor2numpy(
    #                                                      fake_B2A2B_heatmap[0]), self.img_size),
    #                                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'A2B_%07d.png' % step), A2B * 255.0)

        self.g.train(), self.dis_global.train(), self.dis_local.train()

    #   if step % self.save_freq == 0:
    #     self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

    #   if step % 1000 == 0:
    #     params = {}
    #     params['g'] = self.g.state_dict()
    #     params['disGB'] = self.dis_global.state_dict()
    #     params['disLB'] = self.dis_local.state_dict()
    #     torch.save(params, os.path.join(self.result_dir,
    #                self.dataset + '_params_latest.pt'))

  def save(self, dir, step):
    params = {}
    params['g'] = self.g.state_dict()
    params['dis_global'] = self.dis_global.state_dict()
    params['dis_local'] = self.dis_local.state_dict()
    torch.save(params, os.path.join(
        dir, self.dataset + '_params_%07d.pt' % step))

  def load(self, dir, step):
    params = torch.load(
        os.path.join(dir, self.dataset + '_params_%07d.pt' % step)
    )
    self.g.load_state_dict(params['g'])
    self.dis_global.load_state_dict(params['dis_global'])
    self.dis_local.load_state_dict(params['dis_local'])

  def test(self):
    pass
    # model_list = glob(os.path.join(
    #     self.result_dir, self.dataset, 'model', '*.pt'))
    # if not len(model_list) == 0:
    #   model_list.sort()
    #   iter = int(model_list[-1].split('_')[-1].split('.')[0])
    #   self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
    #   print(" [*] Load SUCCESS")
    # else:
    #   print(" [*] Load FAILURE")
    #   return

    # self.g.eval()
    # for n, (x_tgt_real, _) in enumerate(self.testA_loader):
    #   x_tgt_real = x_tgt_real.to(self.device)

    #   x_tgt_fake, _, x_tgt_fake_heatmap = self.g(x_tgt_real)

    #   A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(x_tgt_real[0]))),
    #                         cam(tensor2numpy(
    #                             fake_A2A_heatmap[0]), self.img_size),
    #                         RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
    #                         cam(tensor2numpy(
    #                             x_tgt_fake_heatmap[0]), self.img_size),
    #                         RGB2BGR(tensor2numpy(denorm(x_tgt_fake[0]))),
    #                         cam(tensor2numpy(
    #                             x_tgt_fake2A_heatmap[0]), self.img_size),
    #                         RGB2BGR(tensor2numpy(denorm(x_tgt_fake2A[0])))), 0)

    #   cv2.imwrite(os.path.join(self.result_dir, self.dataset,
    #               'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

    # for n, (real_B, _) in enumerate(self.testB_loader):
    #   real_B = real_B.to(self.device)

    #   fake_B2A2B, _, fake_B2A2B_heatmap = self.g(fake_B2A)

    #   fake_B2B, _, fake_B2B_heatmap = self.g(real_B)

    #   B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
    #                         cam(tensor2numpy(
    #                             fake_B2B_heatmap[0]), self.img_size),
    #                         RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
    #                         cam(tensor2numpy(
    #                             fake_B2A_heatmap[0]), self.img_size),
    #                         RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
    #                         cam(tensor2numpy(
    #                             fake_B2A2B_heatmap[0]), self.img_size),
    #                         RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

    #   cv2.imwrite(os.path.join(self.result_dir, self.dataset,
    #               'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)

  def translate(self):
    pass
    # model_list = glob(os.path.join(
    #     self.result_dir, self.dataset, 'model', '*.pt'))
    # if not len(model_list) == 0:
    #   model_list.sort()
    #   self.load(os.path.join(self.result_dir,
    #             self.dataset, 'model'), self.iteration)
    #   print(" [*] Load SUCCESS")
    # else:
    #   print(" [*] Load FAILURE")
    #   return

    # if not os.path.exists('translations'):
    #   os.mkdir('translations')

    # model_translations_dir = Path('translations', self.dataset)
    # if not os.path.exists(model_translations_dir):
    #   os.mkdir(model_translations_dir)

    # model_with_iters_translations_dir = Path(
    #     model_translations_dir, f'iter-{self.iteration}'
    # )
    # if not os.path.exists(model_with_iters_translations_dir):
    #   os.mkdir(model_with_iters_translations_dir)

    # train_translated_imgs_dir = Path(model_with_iters_translations_dir, 'train')
    # test_translated_imgs_dir = Path(model_with_iters_translations_dir, 'test')
    # full_translated_imgs_dir = Path(model_with_iters_translations_dir, 'full')

    # if not os.path.exists(train_translated_imgs_dir):
    #   os.mkdir(train_translated_imgs_dir)
    # if not os.path.exists(test_translated_imgs_dir):
    #   os.mkdir(test_translated_imgs_dir)
    # if not os.path.exists(full_translated_imgs_dir):
    #   os.mkdir(full_translated_imgs_dir)

    # self.g.eval()

    # print('translating train...')
    # for n, (x_tgt_real, _) in enumerate(self.trainA_loader):
    #   x_tgt_real = x_tgt_real.to(self.device)
    #   img_path, _ = self.trainA_loader.dataset.samples[n]
    #   img_name = Path(img_path).name.split('.')[0]
    #   x_tgt_fake, _, _ = self.g(x_tgt_real)
    #   print(os.path.join(train_translated_imgs_dir, f'{img_name}_fake_B.png'))
    #   cv2.imwrite(
    #       os.path.join(train_translated_imgs_dir, f'{img_name}_fake_B.png'),
    #       RGB2BGR(tensor2numpy(denorm(x_tgt_fake[0]))) * 255.0
    #   )
    #   cv2.imwrite(
    #       os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.png'),
    #       RGB2BGR(tensor2numpy(denorm(x_tgt_fake[0]))) * 255.0
    #   )

    # print('translating test...')
    # for n, (x_tgt_real, _) in enumerate(self.testA_loader):
    #   x_tgt_real = x_tgt_real.to(self.device)
    #   img_path, _ = self.testA_loader.dataset.samples[n]
    #   img_name = Path(img_path).name.split('.')[0]
    #   x_tgt_fake, _, _ = self.g(x_tgt_real)
    #   print(os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.png'))
    #   cv2.imwrite(
    #       os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.png'),
    #       RGB2BGR(tensor2numpy(denorm(x_tgt_fake[0]))) * 255.0
    #   )
    #   cv2.imwrite(
    #       os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.png'),
    #       RGB2BGR(tensor2numpy(denorm(x_tgt_fake[0]))) * 255.0
    #   )
