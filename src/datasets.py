import os
import cv2
import hydra
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms,datasets
#from torchvision import datasets

from PIL import Image,ImageFilter

import glob
import re


class MNIST_test(datasets.MNIST):
    '''
    テスト用。ラベルを0-1にしてある。
    
    '''
    def __init__(self,
                 df: pd.DataFrame,
                 datadir,
                 phase: str,
                 config={},):
        super().__init__('data',
        train = True,
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ]),
        download=True)
        
        self.data  = self.data[:1000,:,:]
        self.targets = self.targets[:1000]

    def __getitem__(self,index):
        img,target = super().__getitem__(index)
        target = target/10
        return img.float(),target



class DefaultDataset(data.Dataset):
    def __init__ (self,
                 df: pd.DataFrame,
                 datadir,
                 phase: str,
                 config={},
                 ):
        self.df = df
        self.datadir = datadir
        self.config = config
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.Resize((self.config['img_size'],self.config['img_size'])),
            transforms.ToTensor(),
        ])
# resize sizeが関数伝いにこのモジュールに伝わるため、Dataset classの中で
# transformを定義している
#　もっとかっこいいやり方あるかも

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        image_file_name = self.df['Id'].values[idx] + '.jpg'
        data = Image.open(os.path.join(self.datadir,image_file_name))
        

        if self.phase == 'train':
            data = self.transform(data)
            target = self.df['Pawpularity'].values[idx]

            return data.float(), torch.tensor(target).float()
            
        elif self.phase == 'valid':
            data = self.transform(data)
            target = self.df['Pawpularity'].values[idx]

            return data.float(), torch.tensor(target).float()
        else:
            data = self.transform(data)
            return data.float()