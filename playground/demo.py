from __future__ import print_function
import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from nltk.corpus import stopwords

from models import Concat, Discriminator, Generator
from dataset import Birds

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='folder' , help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='.', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./Results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--cls', action='store_true', help='activates cls run')

stopwords = set(stopwords.words('english'))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

opt = parser.parse_args()

device = torch.device("cpu")
ngpu = int(opt.ngpu) # num gpus

bird = Birds()
sen = "a bird with blue body and red tail"
embedding_avg = np.empty(bird.encoding_dim)
for word in sen.split():
    if word in stopwords or word not in bird.embeddings:
        continue
    embedding_avg += bird.embeddings[word]
embedding_avg /= len(sen.split())
real_embedding = embedding_avg

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load("netG_epoch_49.pth"))

optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

noise = torch.randn(batch_size, nz, 1, 1, device=device)
#fake = netG(real_embedding, noise)
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
fake = netG(real_embedding ,fixed_noise)
vutils.save_image(fake.detach(),
        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
        normalize=True)
print("done")
