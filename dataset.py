import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import random
import glob
import os


def denormalize(tensors):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 1)


class photometric_distort(object):
    def __call__(self, image):
        distortions = [F.adjust_brightness,
                       F.adjust_contrast,
                       F.adjust_saturation,
                       F.adjust_hue]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ is 'adjust_hue':
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    adjust_factor = random.uniform(0.5, 1.5)
                # Apply this distortion
                image = d(image, adjust_factor)

        return image


class Rotate90(object):
    def __call__(self, image):
        if random.random() > 0.5:
#             image = image.permute(0, 2, 1)
            image = image.transpose(method=Image.ROTATE_90)

        return image


class ColorShuffle(object):
    def __call__(self, image):
        if random.random() > 0.5:
            r = torch.randperm(3)
            image = image[r]

        return image


class PatchDataset(data.Dataset):
    def __init__(self, scale, patch_size):
        super(PatchDataset, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        
        self.imgpath = 'crop_images/'
        self.image_list = glob.glob(self.imgpath + '*.png')
        
        self.input_transform = transforms.Compose([
            transforms.RandomCrop(patch_size, pad_if_needed=True),
            Rotate90(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            ColorShuffle(),
        ])
#         self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_name = self.image_list[index]
        img = Image.open(image_name).convert('RGB')

        hr_img = self.input_transform(img)
        lr_img = self.hr2lr(hr_img, self.scale)

        return hr_img, lr_img
        
    def hr2lr(self, hr_img, scale):
        h, w = hr_img.size()[1:]
        lr_img = F.resize(hr_img, (h // scale, w // scale), Image.BICUBIC)
        return lr_img

    def __len__(self):
        return len(self.image_list)


class TestDataset(data.Dataset):
    def __init__(self, scale):
        super(TestDataset, self).__init__()
        self.scale = scale
        
        self.hr_imgpath = 'set14/'
        self.hr_image_list = glob.glob(self.hr_imgpath + '*.png')
        
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        hr_image_name = self.hr_image_list[index]
        hr_img = Image.open(hr_image_name).convert('RGB')
        hr_img = self.crop(hr_img, self.scale)
        
        img_name = os.path.basename(hr_image_name)
        lr_img = Image.open("testing_lr_images/{}".format(img_name)).convert('RGB')
        
        return self.to_tensor(hr_img), self.to_tensor(lr_img), img_name
    
    def crop(self, hr_img, scale):
        w, h = hr_img.size
        hr_img = hr_img.crop((0, 0, w // scale * scale, h // scale * scale)) 
        return hr_img
    
    def __len__(self):
        return len(self.hr_image_list)

    
class DIV2K(data.Dataset):
    def __init__(self, scale, patch_size):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        
        self.imgpath = 'DIV2K_train_HR/'
        self.image_list = glob.glob(self.imgpath + '*.png')
        
        self.input_transform = transforms.Compose([
            transforms.RandomCrop(patch_size, pad_if_needed=True),
            Rotate90(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_name = self.image_list[index]
        img = Image.open(image_name).convert('RGB')

        hr_img = self.input_transform(img)
        lr_img = self.hr2lr(hr_img, self.scale)

        return hr_img, lr_img
        
    def hr2lr(self, hr_img, scale):
        h, w = hr_img.size()[1:]
        lr_img = F.resize(hr_img, (h // scale, w // scale), Image.BICUBIC)
        return lr_img

    def __len__(self):
        return len(self.image_list)
    