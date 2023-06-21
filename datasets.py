# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from cgi import test
from distutils import core
from email import parser
import os
from random import random
from tkinter import W
import torch
import json
import random
import numpy as np

from torchvision import datasets, transforms
from nltk.tokenize import sent_tokenize

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


from PIL import Image

MAX_CAP_LEN = 80

from transformers import BertTokenizer
from utils import get_colorization_data


def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item ] # and i<=MAX_CAP_LEN

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, split='train'):
        self.split = split
        self.img_dir = os.path.join(img_dir,self.split)
        self.test = False

        if split == 'val':
            caption_path = os.path.join(img_dir,'cap-img-pairs_val.json')      
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        elif split == 'test':
            self.img_dir = 'example' # 
            self.test = True
            caption_path = os.path.join('example','test.json')

            self.coljitter = transforms.ColorJitter(brightness=0.,contrast=0.1,saturation=0.1,hue=0.5)#,hue=0.5
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        
        self.pairs = json.load(open(caption_path,'r'))
                 
    def get_img(self, img_name):
        img_pth = os.path.join(self.img_dir, img_name)
        img = Image.open(img_pth).convert('RGB')
        #########
        if self.test:    
            img = self.coljitter(img)
        img = self.transform(img)

        return img

    
    def __getitem__(self, index):
        key, cap = self.pairs[index]
        img = self.get_img(key)
        parser_mat = 0
        return img, cap, key, parser_mat

    def __len__(self):
        return len(self.pairs)

def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def inst2bit(inst_label):
    labels = np.unique(inst_label)
    # print('inst_label',inst_label.shape)
    print('labels',labels)
    bit_labels = []
    for label in labels:
        if label == 0:
            continue
        print('label',label)
        bit_label = np.zeros((224,224))
        bit_label[inst_label==label] = 1
        bit_labels.append(np.expand_dims(bit_label,axis=0))
    out = np.concatenate(bit_labels,axis=0)
    # print('out',out.shape)
    return out
