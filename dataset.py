import numpy as np
import os
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

Birds_img_dir = "./data/Birds/images"
Birds_txt_dir = "./data/Birds/cub_cvpr/text_c10"

class Birds(Dataset):


    def __init__(self, img_dir=Birds_img_dir, txt_dir=Birds_txt_dir):

        self._load_descriptions(txt_dir)
        #self._load_images(img_dir)

        assert self.N == self.descriptions, \
            "img/txt mismatch in Birds.__init__"

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.images[i],self.descriptions[i]

    def _load_images(self, img_dir):
        self.images = list()

        print("Loading images...")

        for subdir, dirs, files in os.subdirs(img_dir):
            print(subdir,dirs)
            for file in files:
                img = Image.open(file)
                self.images.append(torch.Tensor(img))
        print("done!")

    def _load_descriptions(self, txt_dir):
        self.descriptions = dict()

        print("Loading txt descriptions...")

        subdirs = np.asarray([w for _,w,_ in os.subdirs(txt_dir)][0])
        desc_sets = np.asarray([f for _,_,f in os.subdirs(txt_dir)][1:])

        sorted_indices = np.argsort(subdirs,axis=0)
        subdirs = subdirs[sorted_indices]
        desc_sets = desc_sets[sorted_indices]

        counter = 0
        for i in range(len(desc_sets)):
            tmp = list()
            for dset in desc_sets:
                if dset[-4:] == ".txt":
                    tmp.append(list)
                    counter += 1
            desc_sets[i] = tmp.copy()


        print("done!")
