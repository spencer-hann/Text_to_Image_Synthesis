# Fork of dcgan from pytorch.examples.dcgan

from __future__ import print_function
import argparse
import os
import time
import random
import torch
import math
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dataset2 import TTI_Dataset
from models import Concat, Discriminator, Generator
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='folder' , help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='.', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--cls', action='store_true', help='activates cls run')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def plot(loss, axis_title, title, filename):
    plt.figure()
    x_axis = [x for x in range(len(loss))]
    plt.plot(x_axis, loss)
    plt.xlabel("Epoch")
    plt.ylabel(axis_title)
    plt.title(title)
    plt.savefig(filename + '_' + axis_title)
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

dataset = TTI_Dataset()
nc=3

print('loaded dataset')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, drop_last=True,
                                         num_workers=int(opt.workers))

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
print('starting epochs')
starting_time = time.time()
inception_scores = []
total_runs = math.ceil(len(dataset)/opt.batchSize) - 1
g_loss = []
d_loss = []
for epoch in range(opt.niter):
    etime = time.time()
    tempg = tempd = tot = 0
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
            errD_wrong = criterion(output, label)
            errD_wrong = torch.div(errD_wrong, 2)
            errD_wrong.backward()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(real_embedding, noise)
        label.fill_(fake_label)
        output = netD(fake.detach(), real_embedding)
        errD_fake = criterion(output, label)
        errD_fake = torch.div(errD_fake, 2)
        errD_fake.backward()

        errD = errD_real + errD_fake
        if opt.cls:
            errD += errD_wrong


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

        tempg += errG.item()
        tempd += errD.item()
        tot += 1
        print("[{}/{}][{}/{}] Loss_D: {:.4f} | Loss_G: {:.4f} | D(x) {:.4f} | D(G(z)): {:.8f}"
            .format(epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1 / D_G_z2))
        if i % total_runs == 0:
            vutils.save_image(real_image,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(real_embedding ,fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
    print("Epoch time:", time.time() - etime)
    g_loss.append(tempg/tot)
    d_loss.append(tempd/tot)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
plot(g_loss, 'Generator Loss', "GAN - Generator_Loss", 'gan_losses')
plot(d_loss, 'Discriminator Loss', 'GAN - Discriminator_Loss', 'gan_losses')
ending_time = time.time()
print("Total runtime for epochs", ending_time - starting_time)
