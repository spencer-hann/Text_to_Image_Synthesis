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

'''
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
parser.add_argument('--cls_', action='store_true', help='activates cls_ run')
'''
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
    t = filename + '_' + axis_title + '.png'
    plt.savefig(t)
# START
'''
opt = parser.parse_args()
print(opt)
'''

def train_model(lr1, lr2, dirname, subtitle):
    #parser args
    manualSeed = None
    cuda = True
    ngpu = 1
    batchSize = 64
    niter = 600
    beta1 = 0.5
    lrD = lr1
    lrG = lr2
    outf = dirname
    workers = 2
    nz = 100
    ngf = 64
    ndf = 64
    netG = ''
    netD = ''
    cls_ = True

    try:
        os.makedirs(outf)
    except OSError:
        pass

    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    dataset = TTI_Dataset()
    nc=3

    print('loaded dataset')
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, drop_last=True,
                                             num_workers=int(workers))

    device = torch.device("cuda:0" if cuda else "cpu")

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    #if netG != '':
    #    netG.load_state_dict(torch.load(netG))
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    #if netD != '':
    #    netD.load_state_dict(torch.load(netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    print('starting epochs')
    starting_time = time.time()
    inception_scores = []
    total_runs = math.ceil(len(dataset)/batchSize) - 1
    g_loss = []
    d_loss = []
    for epoch in range(niter):
        etime = time.time()
        tempg = tempd = tot = 0
        # right image, right embed, wrong embed
        for i, (real_image, real_embedding, wrong_embedding) in enumerate(dataloader, 0):
            if cuda:
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

            if cls_:
                # cls_ real img, fake embed
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
            if cls_:
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
                .format(epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1 / D_G_z2))
            if i % total_runs == 0:
                vutils.save_image(real_image,
                        '%s/real_samples.png' % outf,
                        normalize=True)
                fake = netG(real_embedding ,fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                        normalize=True)
        print("Epoch time:", time.time() - etime)
        g_loss.append(tempg/tot)
        d_loss.append(tempd/tot)
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
    plot(g_loss, 'Generator_Loss', "GAN - Generator_Loss" + ' ' + subtitle, subtitle + 'gan_losses')
    plot(d_loss, 'Discriminator_Loss', 'GAN - Discriminator_Loss' + ' ' + subtitle, subtitle + 'gan_losses')
    ending_time = time.time()
    print("Total runtime for epochs", ending_time - starting_time)
