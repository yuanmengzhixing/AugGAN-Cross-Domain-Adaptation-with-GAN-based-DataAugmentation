import cv2
from data.dataset_utils import *
from pdb import set_trace as st
import random
import numpy as np
import torchvision.datasets as dset
import scipy.io as sio
import torch
import pandas as pd
from data.image_folder import make_dataset

class ImageDataset(dset.ImageFolder):
    def __init__(self, dir_style_A, dir_style_B, load_size, seqs, fine_size, phase, opt_choice):
        self.phase = phase
        self.root_dir = root_dir
        self.load_size = load_size
        self.dir_style_A = dir_style_A
        self.dir_style_B = dir_style_B

        self.A_paths = self.make_dataset(self.dir_style_A)
        self.B_paths = self.make_dataset(self.dir_style_B)
        self.A_label_paths = self.build_dataset(A_seqs, lbl=True)
        self.B_label_paths = self.build_dataset(B_seqs, lbl=True)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_label_paths = sorted(self.A_label_paths)
        self.B_label_paths = sorted(self.B_label_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(load_size=load_size, fine_size=fine_size, opt_choice=opt_choice)
        self.label_transform = get_transform(load_size=load_size, fine_size=fine_size, 
                                             opt_choice=opt_choice, normalize=False)


    def __getitem__(self, index):
        index_A = random.randint(0, self.A_size -1)
        A_path = self.A_paths[index_A]
        A_label_path = self.A_label_paths[index_A]

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_label_path = self.B_label_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_label = Image.open(A_label_path)
        A_label_np = np.asarray(A_label)
        B_label = Image.open(B_label_path)
        B_label_np = np.asarray(B_label)

        A_label_np = cv2.resize(A_label_np, (256, 144), cv2.INTER_NEAREST)
        B_label_np = cv2.resize(B_label_np, (256, 144), cv2.INTER_NEAREST)
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        A_label = torch.from_numpy(A_label_np).long()
        B_label = torch.from_numpy(B_label_np).long()
        
        return {'A': A_img, 'B': B_img, 'A_label': A_label, 'B_label': B_label, 
                'A_Paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'GTAdataset'
    
    def set_domain(self, style_A, style_B):
        weather_list = pd.read_csv(os.path.join(self.root_dir, 'train/weather.txt'), header=None)
        weather_dict = {}
        for k, v in zip(weather_list[1], weather_list[0]):
            if k in weather_dict:
                weather_dict[k].append(v)
            else:
                weather_dict[k] = [v]
        return weather_dict.get(style_A), weather_dict.get(style_B)
    
    def build_dataset(self, seqs, lbl=False):
        paths = []
        tail = 'train/cls/%.03d/' if lbl else 'train/img/%.03d/'
        for seq in seqs:
            path = os.path.join(self.root_dir, tail % seq)
            paths += make_dataset(path)
        return paths