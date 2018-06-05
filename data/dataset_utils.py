import os.path
import torchvision.transforms as transforms
from PIL import Image
import PIL

def get_transform(load_size, fine_size, opt_choice, normalize=True):
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

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                      
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
