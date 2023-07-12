import argparse

from UGATIT import UGATIT
from utils import *

"""parsing and configuration"""


def parse_args():
  desc = "Pytorch implementation of U-GAT-IT"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--phase', type=str, default='train',
                      help='[train / test / translate]')
  parser.add_argument('--light', type=str2bool, default=False,
                      help='[U-GAT-IT full version / U-GAT-IT light version]')
  parser.add_argument('--dataset', type=str,
                      default='YOUR_DATASET_NAME', help='dataset_name')

  parser.add_argument('--iteration', type=int, default=1000000,
                      help='The number of training iterations')
  parser.add_argument('--batch_size', type=int, default=1,
                      help='The size of batch size')
  parser.add_argument('--print_freq', type=int, default=1000,
                      help='The number of image print freq')
  parser.add_argument('--save_freq', type=int, default=100000,
                      help='The number of model save freq')
  parser.add_argument('--decay_flag', type=str2bool,
                      default=True, help='The decay_flag')

  parser.add_argument('--lr', type=float, default=0.0001,
                      help='The learning rate')
  parser.add_argument('--weight_decay', type=float,
                      default=0.0001, help='The weight decay')
  parser.add_argument('--adv_weight', type=int,
                      default=1, help='Weight for GAN')
  parser.add_argument('--cycle_weight', type=int,
                      default=10, help='Weight for Cycle')
  parser.add_argument('--identity_weight', type=int,
                      default=10, help='Weight for Identity')
  parser.add_argument('--cam_weight', type=int,
                      default=1000, help='Weight for CAM')

  parser.add_argument('--ch', type=int, default=64,
                      help='base channel number per layer')
  parser.add_argument('--n_res', type=int, default=4,
                      help='The number of resblock')
  parser.add_argument('--n_dis', type=int, default=6,
                      help='The number of discriminator layer')

  parser.add_argument('--img_size', type=int, default=256,
                      help='The size of image')
  parser.add_argument('--img_ch', type=int, default=3,
                      help='The size of image channel')

  parser.add_argument('--result_dir', type=str, default='results',
                      help='Directory name to save the results')
  parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
  parser.add_argument('--benchmark_flag', type=str2bool, default=False)
  parser.add_argument('--resume', type=str2bool, default=False)

  return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
  # --result_dir
  check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
  check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
  check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
  with open(os.path.join(args.result_dir, args.dataset, 'training_log.txt'), 'w'):
    pass
  # --epoch
  try:
    assert args.epoch >= 1
  except:
    print('number of epochs must be larger than or equal to one')

  # --batch_size
  try:
    assert args.batch_size >= 1
  except:
    print('batch size must be larger than or equal to one')
  return args


"""main"""


def main():
  # parse arguments
  args = parse_args()
  if args is None:
    exit()

  # open session
  gan = UGATIT(args)

  # build graph
  gan.build_model()

  if args.phase == 'train':
    gan.train()
    print(" [*] Training finished!")

  if args.phase == 'test':
    gan.test()
    print(" [*] Test finished!")

  if args.phase == 'translate':
    gan.translate()
    print(" [*] Translation finished!")


if __name__ == '__main__':
  main()
