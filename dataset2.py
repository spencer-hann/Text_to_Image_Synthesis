import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchfile
from PIL import Image
import PIL
from torchvision import transforms
import random

EMBED = 0
IMAGE = 1
# test to image dataset
# data type: train, test, valid
class TTI_Dataset(Dataset):
    def __init__(self, data_type='train'):
        self.pairs = []
        self.img_dim = 64
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'data')
        id_path = os.path.join(data_path, data_type + 'ids.txt')
        with open(id_path, 'r') as f:
            self.ids = [line.strip() for line in f]
        class_path = os.path.join(data_path, data_type + 'classes.txt')
        with open(class_path, 'r') as f:
            self.labels = [line.strip() for line in f]
        images_path = os.path.join(cwd, 'images')

        # image handling
        self.transformations = transforms.Compose([
        transforms.Resize((self.img_dim, self.img_dim)),
        transforms.CenterCrop((self.img_dim, self.img_dim)),
        transforms.ToTensor(),])
        # get all t7 files for training/test set
        for l in self.labels:
            cur_dir = os.path.join(data_path, l)
            temp_files = os.listdir(cur_dir)
            for file in temp_files:
                temp = torchfile.load(os.path.join(cur_dir, file))
                emb = temp[b'txt']
                # get associated image
                im_file = temp[b'img'].decode('UTF8')
                im = os.path.join(images_path, im_file)
                self.pairs.append((emb, im))

    def __len__(self):
        return len(self.pairs)

    def _get_image(self, path):
        im = Image.open(path).convert('RGB')
        return self.transformations(im)

    def __getitem__(self, i):
        '''
        takes in index for real images
        Needs to return real img, fake img, real txt, fake txt
        '''
        r1 = random.randint(0, 9)
        r2 = random.randint(0, 9)
        right_image = self._get_image(self.pairs[i][IMAGE])

        right_embed = self.pairs[i][EMBED][r1]
        # generate 2 random nums != i
        population = [x for x in range(len(self.pairs)) if x != i]
        r = random.sample(population, 1)
        wrong_embed = self.pairs[r[0]][EMBED][r2]

        return (right_image, right_embed, wrong_embed)
