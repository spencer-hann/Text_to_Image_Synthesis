import numpy as np
import os
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets


Birds_img_dir = "./data/Birds/Caltech-UCSD-Birds-200-2011/CUB_200_2011/images"
Birds_txt_dir = "./data/Birds/cub_cvpr/text_c10"

class Birds(Dataset):


    def __init__(self, img_dir=Birds_img_dir, txt_dir=Birds_txt_dir):
        self.desc_per_img = 10 # number of text descriptions per image

        self._load_descriptions(txt_dir)
        self._load_images(img_dir)

        self.N = len(self.images) * 10 # 10 examples/descriptions per image

        #assert len(self.images) == len(self.descriptions), \
        #    "img/txt mismatch in Birds.__init__"

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        i = index // self.desc_per_img
        j = index % self.desc_per_img
        return self.images[i], self.descriptions[i,j], self.file_names[i]

    def _load_images(self, img_dir):
        print("Loading images...")
        self.img_dim =  180
        # Resizing all images to uniform size
        transformations = transforms.Compose([
        transforms.Resize(self.img_dim),
        transforms.CenterCrop(self.img_dim),
        transforms.ToTensor(),])

        image_dataset = datasets.ImageFolder(img_dir, transform = transformations)
        self.images = torch.empty([len(image_dataset),3,self.img_dim,self.img_dim])

        for i,(img,_ )in enumerate(image_dataset):
            self.images[i,:,:,:] = img[:,:,:]
            #TODO: remove during production
            #if i > 100: break
        print("done!")

    def _load_descriptions(self, txt_dir):
        print("Loading txt descriptions...")

        # all folders in current directory
        # each folder/subdir corresponds to a species of Bird
        subdirs = np.asarray([w for _,w,_ in os.walk(txt_dir)][0])
        # all files within each folder
        # each sub-list contains files for a specific species
        file_sets = np.asarray([f for _,_,f in os.walk(txt_dir)][1:])

        # parallel sort by species/class ID
        sorted_indices = np.argsort(subdirs,axis=0)
        subdirs = subdirs[sorted_indices]
        file_sets = file_sets[sorted_indices]

        # remove extra files (non-".txt") and determin number of files remaining
        num_files = 0
        for file_set in file_sets:
            # traverse backwards becuase items are being deleted
            for j in range(len(file_set)-1,-1,-1):
                if file_set[j][-4:] != ".txt":
                    del file_set[j]
            num_files += len(file_set)

        self.file_names = np.empty(num_files, dtype=object) # object is str
        self.descriptions = np.empty((num_files,self.desc_per_img), dtype=object)

        i = 0
        for subdir,file_set in zip(subdirs,file_sets):

            file_set.sort()

            for file_name in file_set:

                self.file_names[i] = file_name

                with open(txt_dir +'/'+ subdir +'/'+ file_name) as f:
                    for j,line in enumerate(f):
                        self.descriptions[i,j] = line.strip()
                    # make sure number of descriptions is corect
                    assert j == self.desc_per_img-1
                i += 1

        print("done!")
