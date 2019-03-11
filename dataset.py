import numpy as np
import os
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets


Birds_img_dir = "./data/Birds/images"
Birds_txt_dir = "./data/Birds/cub_cvpr/text_c10"

class Birds(Dataset):


    def __init__(self, img_dir=Birds_img_dir, txt_dir=Birds_txt_dir):

        self._load_descriptions(txt_dir)
        self._load_images(img_dir)

        self.N = len(self.images) * 10 # 10 examples/descriptions per image

        assert len(self.images) == len(self.descriptions), \
            "img/txt mismatch in Birds.__init__"

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.images[index],self.descriptions[i]

    def _load_images(self, img_dir):
        print("Loading images...")
        self.img_dim =  180
        # Resizing all images to uniform size
        transformations = transforms.Compose([
        transforms.Resize(self.img_dim),
        transforms.CenterCrop(self.img_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) ## Should look into the values one more time

        image_dataset = datasets.ImageFolder("./data/Birds/images", transform = transformations)
        self.images = torch.empty([len(image_dataset),3,self.img_dim,self.img_dim])

        for i,(img,_ )in enumerate(image_dataset):
            self.images[i] = img
        print("done!")

    def _load_descriptions(self, txt_dir):
        self.descriptions = dict()

        print("Loading txt descriptions...")

        subdirs = np.asarray([w for _,w,_ in os.walk(txt_dir)][0])
        desc_sets = np.asarray([f for _,_,f in os.walk(txt_dir)][1:])

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
