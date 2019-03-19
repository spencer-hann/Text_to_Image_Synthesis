# Fork of dcgan from pytorch.examples.dcgan

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
from dataset2 import TTI_Dataset
from dataset import Birds
from models import Concat, Discriminator, Generator


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='folder' , help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='.', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate, default=0.0004')
parser.add_argument('--beta1', type=float, default=0.3, help='beta1 for adam. default=0.3')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./Results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--desc_per_img', type=int, default=5)
parser.add_argument('--incl_stopwords', type=bool, default=False)

parser.add_argument('--cls', action='store_true', help='activates cls run')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# START

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#dataset = TTI_Dataset(descriptions_per_image=opt.desc_per_img)
dataset = Birds(descriptions_per_image=opt.desc_per_img, incl_stopwords=opt.incl_stopwords)
nc=3

print('loaded dataset')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu) # num gpus
nz = int(opt.nz) # size of noise vector
ngf = int(opt.ngf) # number of filters in generator layer
ndf = int(opt.ndf) # number of filters in discriminator layer

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
loss_by_epoch_D = np.zeros(opt.niter, dtype=np.float_)
loss_by_epoch_G = np.zeros(opt.niter, dtype=np.float_)
print('starting epochs')
starting_time = time.time()
for epoch in range(opt.niter):
    etime = time.time()

    # right image, right embed, wrong embed
    for i, (real_image, real_embedding, wrong_embedding) in enumerate(dataloader, 0):
        if opt.cuda:
            real_image = real_image.to(device)
            real_embedding = real_embedding.to(device)
            wrong_embedding = wrong_embedding.to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real image, real embed
        netD.zero_grad()

        batch_size = real_image.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = netD(real_image, real_embedding)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        if opt.cls:
            # cls real img, fake embed
            label.fill_(fake_label)
            output = netD(real_image, wrong_embedding)
            errD_wrong = criterion(output2, label)
            errD_wrong.backward()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(real_embedding, noise)
        label.fill_(fake_label)
        output = netD(fake.detach(), real_embedding)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        if opt.cls:
            errD = errD_real + errD_fake + errD_wrong
        else:
            errD = errD_real + errD_fake

        D_G_z1 = output.mean().item()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake, real_embedding)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        loss_by_epoch_D[epoch] = errD.item()
        loss_by_epoch_G[epoch] = errG.item()
        print("[{}/{}][{}/{}] Loss_D: {:.4f} | Loss_G: {:.4f} | D(x) {:.4f} | D(G(z)): {:.4f} / {:.4f}"
            .format(epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_image,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(real_embedding ,fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
    print("saving progress to %s/loss_by_epoch_D_descperimg_%d_stopwords_%d.out" % (opt.outf, opt.desc_per_img, opt.incl_stopwords))
    np.savetxt("%s/loss_by_epoch_D_descperimg_%d_stopwords_%d_beta1_%f_lr_%f.out" % (opt.outf, opt.desc_per_img, opt.incl_stopwords, opt.beta1, opt.lr), loss_by_epoch_D)
    print("saving progress to %s/loss_by_epoch_G_descperimg_%d_stopwords_%d.out" % (opt.outf, opt.desc_per_img, opt.incl_stopwords))
    np.savetxt("%s/loss_by_epoch_G_descperimg_%d_stopwords_%d_beta1_%f_lr_%f.out" % (opt.outf, opt.desc_per_img, opt.incl_stopwords, opt.beta1, opt.lr), loss_by_epoch_G)

    print("Epoch time:", time.time() - etime)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
ending_time = time.time()
print("Total runtime for epochs", ending_time - starting_time)
