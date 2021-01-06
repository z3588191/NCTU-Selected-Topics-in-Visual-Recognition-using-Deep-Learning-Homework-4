import math
import torch
import skimage.color as sc
import numpy as np


def calc_psnr(img1, img2):
    # image value: [0, 1]
    # image shape: (n, 3, h, w)
    img1 = img1[:, :, 3:-3, 3:-3]
    img2 = img2[:, :, 3:-3, 3:-3]
    mse = ((img1 - img2) ** 2).mean(dim=(1, 2, 3)) + 1e-8
    psnr = 20 * (1.0 / mse.sqrt()).log10()
    return psnr.mean()


def tf2np_byte(tensor):
    tensor = tensor.cpu().clamp_(0, 1)
    img_np = tensor.permute(0, 2, 3, 1).numpy()
    img_np = (img_np * 255.0).round()
    
    return img_np.astype(np.uint8)


def calc_psnr_Y(sr, hr):
    np_sr = tf2np_byte(sr)
    np_hr = tf2np_byte(hr)
    
    for i in range(sr.size(0)):
        np_sr[i] = sc.rgb2ycbcr(np_sr[i])
        np_hr[i] = sc.rgb2ycbcr(np_hr[i])
    
    np_sr_Y = np_sr[:, :, :, 0] / 255.
    np_hr_Y = np_hr[:, :, :, 0] / 255.
    
    mse = ((np_sr_Y - np_hr_Y) ** 2).mean()

    return -10 * math.log10(mse)

