import os.path
import torchvision.transforms as transforms
from data.image_folder import make_dataset
import cv2
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import numpy as np
import torchvision.datasets as dset
import scipy.io as sio
import torch

class ImageDataset(dset.ImageFolder):
    def __init__(self, root_dir, style_A, style_B, load_size, synthia_seqs, fine_size, phase, opt_choice):
        self.phase = phase
        self.root_dir = root_dir
        self.load_size = load_size
        seq = synthia_seqs
        
#        self.dir_A = os.path.join(root_dir, style_A + '/RGB/Stereo_Left')
#        self.dir_B = os.path.join(root_dir, style_B + '/RGB/Stereo_Left')
#        self.dir_A_label = os.path.join(root_dir, style_A + '/GT/LABELS_mat/Stereo_Left')
#        self.dir_B_label = os.path.join(root_dir, style_B + '/GT/LABELS_mat/Stereo_Left')

#        self.A_paths = make_dataset(self.dir_A)
#        self.B_paths = make_dataset(self.dir_B)
#        self.A_label_paths = make_dataset(self.dir_A_label)

        self.A_paths = get_dataset(seqs=seq, style=style_A, root_dir=root_dir, img=True)
        self.B_paths = get_dataset(seqs=seq, style=style_B, root_dir=root_dir, img=True)
        self.A_label_paths = get_dataset(seqs=seq, style=style_A, root_dir=root_dir, img=False)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_label_paths = sorted(self.A_label_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(load_size=load_size, fine_size=fine_size, opt_choice=opt_choice)


    def __getitem__(self, index):
        index_A = random.randint(0, self.A_size -1)
        A_path = self.A_paths[index_A]
        A_label_path = self.A_label_paths[index_A]

        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_label = sio.loadmat(A_label_path)['label']
        A_label = cv2.resize(A_label, (256, 152), cv2.INTER_NEAREST)
        
        A_label[A_label==12] = 0
        A_label[A_label==13] = 0
        A_label[A_label==14] = 0
        A_label[A_label==15] = 12
        A_label = np.clip(A_label, 7, 12) - 7
        
        A_label = A_label.astype(np.uint8)
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        A_label = torch.from_numpy(A_label).long()

        self.A_paths.pop(index_A)
        self.A_size = len(self.A_paths)
        return {'A': A_img, 'B': B_img, 'A_label': A_label, 'A_Paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size

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

def get_transform(load_size, fine_size, opt_choice):
    transform_list = []
    if opt_choice == 'resize_and_crop':
        osize = load_size
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fine_size))
    elif opt_choice == 'resize':
        osize = fine_size
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    elif opt_choice == 'crop':
        transform_list.append(transforms.RandomCrop(fine_size))
    elif opt_choice == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fine_size)))
    elif opt_choice == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, load_size)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)



'''
    if resize_or_crop == 'resize_and_crop':
        transform_list.append(transforms.Scale(load_size, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fine_size))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fine_size))
    elif resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fine_size)))
    elif resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, load_size)))
        transform_list.append(transforms.RandomCrop(fine_size))
    if isTrain and not no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
'''
    
