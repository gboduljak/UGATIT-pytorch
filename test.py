import math

from torchinfo import summary

from networks import Generator

generator = Generator(input_nc=3, output_nc=3)
summary(generator, [(1, 3, 256, 256), (1, 256)])
