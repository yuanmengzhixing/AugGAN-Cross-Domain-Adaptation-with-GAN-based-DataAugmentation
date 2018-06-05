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
    def __init__(self, root_dir, style_A, style_B, load_size, seqs, fine_size, phase, opt_choice):
        self.phase = phase
        self.root_dir = root_dir
        self.load_size = load_size
        
        A_seqs, B_seqs = self.set_domain(style_A, style_B)
# hand pick day night seqs:
#        B_seqs = [1, 2, 5, 44, 45, 51, 67, 68]
#        A_seqs = [10, 11, 52, 55, 56, 57, 58, 70, 73, 74, 75, 76, 77]
#        A_seqs = [1, 2, 44, 45, 47, 51, 67, 68]
#        B_seqs = [52, 55, 56, 57, 70, 71, 73, 74, 75]
        num_aseqs = len(A_seqs)
        num_bseqs = len(B_seqs)
        num_aseq_sample = int(6000/num_aseqs)
        num_bseq_sample = int(6000/num_bseqs)        

        # the file names in the following two lists are not in the same order
        self.A_paths, self.A_label_paths = self.build_dataset(A_seqs, num_aseq_sample)
#        self.A_label_paths = self.build_dataset(A_seqs, lbl=True)

        # the file order is now the same
        self.A_paths = sorted(self.A_paths)
        self.A_label_paths = sorted(self.A_label_paths)

        # use the same random index to fetch image/label of domain-A
#        random_index_A=random.sample(range(0,len(self.A_paths)), 4000)

#        self.A_paths = [self.A_paths[i] for i in random_index_A]
#        self.A_label_paths = [self.A_label_paths[i] for i in random_index_A]

        # the file names in the following two lists are not in the same order
        self.B_paths, self.B_label_paths = self.build_dataset(B_seqs, num_bseq_sample)
#        self.B_label_paths = self.build_dataset(B_seqs, lbl=True)

        # the file order is now the same
        self.B_paths = sorted(self.B_paths)
        self.B_label_paths = sorted(self.B_label_paths)

        # use the same random index to fetch image/label of domain-B
#        random_index_B=random.sample(range(0,len(self.B_paths)), 4000)

#       self.B_paths = [self.B_paths[i] for i in random_index_B]
#       self.B_label_paths = [self.B_label_paths[i] for i in random_index_B]

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
#        print(self.A_size, self.B_size)
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
        A_img = A_img.crop((0, 0, A_img.size[0], A_img.size[1] - 168))
        B_img = Image.open(B_path).convert('RGB')
        B_img = B_img.crop((0, 0, B_img.size[0], B_img.size[1] - 168))
        A_label = Image.open(A_label_path)
        A_label = A_label.crop((0, 0, A_label.size[0], A_label.size[1] - 168))
        A_label_np = np.asarray(A_label)
        A_label_np.setflags(write=True)
        B_label = Image.open(B_label_path)
        B_label = B_label.crop((0, 0, B_label.size[0], B_label.size[1] - 168))
        B_label_np = np.asarray(B_label)
        B_label_np.setflags(write=True)

        A_label_np[A_label_np==13] = 16
        A_label_np[A_label_np==15] = 0
        A_label_np[A_label_np==16] = 0
        A_label_np[A_label_np==17] = 0
        A_label_np[A_label_np==18] = 0
        A_label_np[A_label_np==19] = 0
        A_label_np[A_label_np==21] = 0
        A_label_np[A_label_np==30] = 0
        A_label_np[A_label_np==31] = 0
        A_label_np[A_label_np==29] = 24
        A_label_np[A_label_np==28] = 24
        A_label_np[A_label_np==27] = 24
        A_label_np[A_label_np==26] = 24
        A_label_np[A_label_np==25] = 24
        
#        A_label_np[A_label_np==3] = 21
        A_label_np[A_label_np==4] = 19
        A_label_np[A_label_np==9] = 18
        A_label_np[A_label_np==12] = 17
        
        A_label_np = np.clip(A_label_np, 16, 24) - 16        
        
        B_label_np[B_label_np==13] = 16
        B_label_np[B_label_np==15] = 0
        B_label_np[B_label_np==16] = 0
        B_label_np[B_label_np==17] = 0
        B_label_np[B_label_np==18] = 0
        B_label_np[B_label_np==19] = 0
        B_label_np[B_label_np==21] = 0
        B_label_np[B_label_np==30] = 0
        B_label_np[B_label_np==31] = 0
        B_label_np[B_label_np==29] = 24
        B_label_np[B_label_np==28] = 24
        B_label_np[B_label_np==27] = 24
        B_label_np[B_label_np==26] = 24
        B_label_np[B_label_np==25] = 24

#        B_label_np[B_label_np==3] = 21
        B_label_np[B_label_np==4] = 19
        B_label_np[B_label_np==7] = 18
        B_label_np[B_label_np==12] = 17

        B_label_np = np.clip(B_label_np, 16, 24) - 16
        
        A_label_np = cv2.resize(A_label_np, (320, 152), cv2.INTER_CUBIC)
        B_label_np = cv2.resize(B_label_np, (320, 152), cv2.INTER_CUBIC)
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
#        print(A_img.size())
#        print(B_img.size())
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
    
    def build_dataset(self, seqs, num_sample):
        img_paths = []
        lbl_paths = []
        tail_lbl = 'train/cls/%.03d/' 
        tail_img = 'train/img/%.03d'
        for seq in seqs:
            num_samp = num_sample
            if seq in [56, 57, 73]:
                print('good sequence')
                num_samp = num_sample*2
            if seq in [8, 9]:
                print('bad sequence')
                num_samp = int(num_sample / 10)

            img_path = os.path.join(self.root_dir, tail_img % seq)
            lbl_path = os.path.join(self.root_dir, tail_lbl % seq)
            img_paths_tmp = make_dataset(img_path)
            img_paths_tmp = sorted(img_paths_tmp)
            lbl_paths_tmp = make_dataset(lbl_path)
            lbl_paths_tmp = sorted(lbl_paths_tmp)
            if seq == 77:
                img_paths_tmp = img_paths_tmp[:200]
                lbl_paths_tmp = lbl_paths_tmp[:200]

            if num_samp > len(img_paths_tmp):
                num_samp = len(img_paths_tmp)
            random_index = random.sample(range(0,len(img_paths_tmp)), num_samp)
            img_paths_tmp = [img_paths_tmp[i] for i in random_index]
            lbl_paths_tmp = [lbl_paths_tmp[i] for i in random_index]
            img_paths += img_paths_tmp
            lbl_paths += lbl_paths_tmp

        return img_paths, lbl_paths
