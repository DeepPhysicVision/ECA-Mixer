import cv2
import imageio
import os
import skimage.io
import scipy.io as sio
import sys
import torch
import numpy as np
import configargparse
from utils.augmented_image_loader import ImageLoader
from utils.utils import *
import torch
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import optim
from thop import profile    #pip install thop
from networks import *

dir = 'Siat'
pil2tensor = transforms.ToTensor()

img1 = Image.open(f'./Results/{dir}0/AttentionNet/Amp/Best.bmp')
img1 = pil2tensor(img1)
print(img1.shape)
#img1 = img1[0,:,:]
img1 = img1.view(1,1,img1.shape[-2],img1.shape[-1])

img2 = Image.open(f'./Results/{dir}1/AttentionNet/Amp/Best.bmp')
img2 = pil2tensor(img2)
#img2 = img2[1,:,:]
img2 = img2.view(1,1,img2.shape[-2],img2.shape[-1])

img3 = Image.open(f'./Results/{dir}2/AttentionNet/Amp/Best.bmp')
img3 = pil2tensor(img3)
#img3 = img3[2,:,:]
img3 = img3.view(1,1,img3.shape[-2],img3.shape[-1])

target_amp = Image.open(f'./Data/Generate/2k256.bmp')
target_amp = pil2tensor(target_amp)
target_amp = target_amp.view(1,target_amp.shape[-3],target_amp.shape[-2],target_amp.shape[-1])
print(target_amp.shape)

# list to tensor, scaling
recon_amp = torch.cat((img1,img2,img3), dim=1)
print(recon_amp.shape)
#recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

# tensor to numpy
recon_amp = recon_amp.squeeze().cpu().detach().numpy()
target_amp = target_amp.squeeze().cpu().detach().numpy()

recon_amp = recon_amp.transpose(1, 2, 0)
target_amp = target_amp.transpose(1, 2, 0)

# save reconstructed image in srgb domain
recon_srgb = srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
imageio.imwrite(f'./Results/{dir}Merge.bmp', (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))

# Placeholders for metrics
psnrs = {'amp': [], 'lin': [], 'srgb': []}
ssims = {'amp': [], 'lin': [], 'srgb': []}
idxs = []

# calculate metrics
psnr_val, ssim_val = get_psnr_ssim(recon_amp, target_amp, multichannel=3)

for domain in ['amp', 'lin', 'srgb']:
    psnrs[domain].append(psnr_val[domain])
    ssims[domain].append(ssim_val[domain])
    print(f'PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, ')

