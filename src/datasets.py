import os
import cv2
import hydra
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image,ImageFilter

import glob
import re


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