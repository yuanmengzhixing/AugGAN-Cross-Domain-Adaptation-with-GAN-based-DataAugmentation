import os.path
import cv2
from pdb import set_trace as st
import random
import numpy as np
import torchvision.datasets as dset
import scipy.io as sio
import torch
from data.dataset_utils import *
from data.image_folder import make_dataset

class ImageDataset(dset.ImageFolder):
    def __init__(self, root_dir, style_A, style_B, load_size, seqs, fine_size, phase, opt_choice):
        self.phase = phase
        self.root_dir = root_dir
        self.load_size = load_size
        seqs = seqs
        
        self.A_paths = get_dataset(seqs=seqs, style=style_A, root_dir=root_dir, img=True)
        self.B_paths = get_dataset(seqs=seqs, style=style_B, root_dir=root_dir, img=True)
        self.A_label_paths = get_dataset(seqs=seqs, style=style_A, root_dir=root_dir, img=False)
        self.B_label_paths = get_dataset(seqs=seqs, style=style_B, root_dir=root_dir, img=False)

        self.A_paths = sorted(self.A_paths)
        self.A_paths = random.sample(self.A_paths, round(0.5*len(self.A_paths)))
        self.B_paths = sorted(self.B_paths)
        self.B_paths = random.sample(self.B_paths, round(0.5*len(self.B_paths)))
        self.A_label_paths = sorted(self.A_label_paths)
        self.B_label_paths = sorted(self.B_label_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
#        self.A_paths = self.A_paths[:round(0.5*self.A_size)]
#        self.B_paths = self.B_paths[:round(0.5*self.B_size)]
#        self.A_label_paths = self.A_label_paths[:round(0.05*self.A_size)]
#        self.B_label_paths = self.B_label_paths[:round(0.05*self.B_size)]
#        self.A_size = round(self.A_size * 0.05)
#        self.B_size = round(self.B_size * 0.05)
        self.transform = get_transform(load_size=load_size, fine_size=fine_size, opt_choice=opt_choice)


    def __getitem__(self, index):
        index_A = random.randint(0, self.A_size -1)
        A_path = self.A_paths[index_A]
        A_label_path = self.A_label_paths[index_A]

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_label_path = self.B_label_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_label = sio.loadmat(A_label_path)['label']
        B_label = sio.loadmat(B_label_path)['label']
        A_label = cv2.resize(A_label, (256, 152), cv2.INTER_NEAREST)
        B_label = cv2.resize(B_label, (256, 152), cv2.INTER_NEAREST)
        
        A_label[A_label==12] = 0
        A_label[A_label==13] = 0
        A_label[A_label==14] = 0
        A_label[A_label==15] = 12
        A_label = np.clip(A_label, 7, 12) - 7
        
        B_label[B_label==12] = 0
        B_label[B_label==13] = 0
        B_label[B_label==14] = 0
        B_label[B_label==15] = 12
        B_label = np.clip(B_label, 7, 12) - 7
        A_label = A_label.astype(np.uint8)
        B_label = B_label.astype(np.uint8)
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        A_label = torch.from_numpy(A_label).long()
        B_label = torch.from_numpy(B_label).long()
#        print(A_img.size(), B_img.size())
        
        return {'A': A_img, 'B': B_img, 'A_label': A_label, 'B_label': B_label, 'A_Paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'SYNTHIADatset'

def get_dataset(seqs, style, root_dir, img=True, all_=True):
    paths = []
    if img:
        tail_F = '/RGB/Stereo_Left/Omni_F'
        tail_B = '/RGB/Stereo_Left/Omni_B'
        tail_L = '/RGB/Stereo_Left/Omni_L'
        if all_:
            tail_R = '/RGB/Stereo_Left/Omni_R'
        else:
            tail_R = None
    else:
        tail_F = '/GT/LABELS_mat/Stereo_Left/Omni_F_mat'
        tail_B = '/GT/LABELS_mat/Stereo_Left/Omni_B_mat'
        tail_L = '/GT/LABELS_mat/Stereo_Left/Omni_L_mat'
        if all_:
            tail_R = '/GT/LABELS_mat/Stereo_Left/Omni_R_mat'
        else:
            tail_R = None

    for seq in seqs:
        path_F = os.path.join(root_dir, style.format(seq) + tail_F)
        path_B = os.path.join(root_dir, style.format(seq) + tail_B)
        path_L = os.path.join(root_dir, style.format(seq) + tail_L)
        if all_:
            path_R = os.path.join(root_dir, style.format(seq) + tail_R)
            paths_R = make_dataset(path_R)
            paths += paths_R
        else:
            path_R = None
        paths_F = make_dataset(path_F)
        paths_B = make_dataset(path_B)
        paths_L = make_dataset(path_L)
        paths += paths_F 
        paths += paths_B
        paths += paths_L
    
    return paths
