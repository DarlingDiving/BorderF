import os
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt
import torch.multiprocessing
from PIL import Image
torch.multiprocessing.set_sharing_strategy('file_system')

class CommonDataloader(data.Dataset):

    def __init__(self, root, noise=None, denoised_data=None,
                 transforms=None, train=True, test=False,real_name='FFHQ',fake_name='Stylegan2'):
        self.train   = train
        self.test    = test
        self.real_name = real_name
        self.fake_name = fake_name

        real_dir = root+"/"+self.real_name
        fake_dir = root+"/"+self.fake_name
        if noise:
            real_dir += '_%s'%noise
            fake_dir += '_%s'%noise
        # get img list
        imgs =  [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        imgs += [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        #imgs = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        #imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
        
        self.imgs = imgs
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        if self.train:
            self.transforms = T.Compose(
                [T.RandomHorizontalFlip(),
                T.RandomCrop(opt.img_size, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std)])
        else:
            self.transforms = T.Compose(
                [
                T.CenterCrop(opt.img_size),
                T.ToTensor(),
                T.Normalize(mean, std)])
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __getitem__(self, index):
        img_path = self.imgs[index]
        
        if self.real_name in img_path.split('/'):
            label = 1
        else:
            label = 0
        data = Image.open(img_path) 
        data_ = np.array(data).copy()
        hr = data_.copy()
        # add noise
        noises = np.random.normal(scale=opt.noise_scale, size=data_.shape)
        noises = noises.round()
        data_noise = data_.astype(np.int16) + noises.astype(np.int16)
        data_noise = data_noise.clip(0, 255).astype(np.uint8)
        lr = data_noise
        hr = Image.fromarray(hr).convert('RGB')
        lr = Image.fromarray(lr).convert('RGB')
        hr = self.transforms(hr)
        lr = self.transforms(lr)

        return lr, hr, label

    def __len__(self):
        return len(self.imgs)
