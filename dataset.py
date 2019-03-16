import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from PIL import Image
from gensim.models import Word2Vec
from nltk import word_tokenize


Birds_img_dir = "./data/Birds/Caltech-UCSD-Birds-200-2011/CUB_200_2011/images"
Birds_txt_dir = "./data/Birds/cub_cvpr/text_c10"

class Birds(Dataset):

    def __init__(self,
            img_dir=Birds_img_dir,
            txt_dir=Birds_txt_dir,
            encoding_dim=1024):
        self.desc_per_img = 10 # number of text descriptions per image
        self.encoding_dim=encoding_dim

        #print("Loading images...")
        #self._load_images(img_dir)
        #print("done!")

        #self.N = len(self.images) * self.desc_per_img
        self.N = 117880

        print("Loading txt descriptions...")
        self._load_descriptions(txt_dir)
        print("done!")

        print("Training word embeddings...")
        self._train_word_embeddings()
        print("done!")

        print("Creating text encodings...")
        self._create_txt_encodings()
        print("done!")

    def _train_word_embeddings(self):
        self.embeddings = Word2Vec(self.descriptions, size=self.encoding_dim)

    def _create_txt_encodings(self):
        embedding_lists = np.empty(self.N, dtype=np.ndarray)

        for i,sentence in enumerate(self.descriptions):
            embedding_lists[i] = np.empty((len(sentence),self.encoding_dim))
            for j,word in enumerate(sentence):
                if word not in self.embeddings: continue
                embedding_lists[i][j] = self.embeddings[word]

        self.encodings = torch.empty(self.N,self.encoding_dim)

        for i,embeddings in enumerate(embedding_lists):
            self.encodings[i] = np.mean(embeddings)

    def __len__(self): return self.N

    def __getitem__(self, index):
        i = index // self.desc_per_img
        #j = index % self.desc_per_img
        #return self.images[i], self.encodings[index]
        return self.descriptions[index]

    def _load_images(self, img_dir):
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
            #if i > 100: break

    def _load_descriptions(self, txt_dir):
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
        self.descriptions = np.empty(num_files * self.desc_per_img, dtype=object)
        #self.descriptions = np.empty((num_files,self.desc_per_img), dtype=object)

        i = 0
        for subdir,file_set in zip(subdirs,file_sets):

            file_set.sort()

            for file_name in file_set:

                #self.file_names[i] = file_name

                with open(txt_dir +'/'+ subdir +'/'+ file_name) as f:
                    for j,line in enumerate(f):
                        self.descriptions[i] = line.strip().split()
                        #self.descriptions[i,j] = line.strip()
                        i += 1
                    # make sure number of descriptions is corect
                    assert j == self.desc_per_img-1
                #i += 1


