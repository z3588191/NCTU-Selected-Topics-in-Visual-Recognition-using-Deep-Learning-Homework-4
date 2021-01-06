import glob
import os
from tqdm import tqdm
import torch
from torchvision.transforms import functional as FT
import matplotlib.pyplot as plt

from RFDN.RFDN import *
from utils import *


torch.cuda.set_device(0)
scale = 3

test_images = glob.glob("testing_lr_images/*.png")

psnr = 0.
model = RFDN(upscale=3)
model.load_state_dict(torch.load("RFDN.pkl"))
model = model.cuda()

to_tensor = transforms.ToTensor()

with torch.no_grad():
    tqbar = tqdm(test_images)
    for img_name in tqbar:
        lr_img = Image.open(img_name).convert('RGB')
        hr_img = Image.open("set14/{}".format(os.path.basename(img_name))).convert('RGB')
        w, h = hr_img.size
        hr_img = hr_img.crop((0, 0, w // scale * scale, h // scale * scale))
        
        hr_img, lr_img = to_tensor(hr_img).unsqueeze(0).cuda(), to_tensor(lr_img).unsqueeze(0).cuda()
        sr_img = model.forward_x8(lr_img).clamp(0, 1)
        
        psnr += calc_psnr(sr_img, hr_img)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 20))
        hr = hr_img[0].permute(1, 2, 0).mul(255).round().cpu().numpy().astype(np.uint8)
        sr = sr_img[0].permute(1, 2, 0).mul(255).round().cpu().numpy().astype(np.uint8)
        ax1.imshow(sr)
        ax2.imshow(hr)
        
        sr_img = sr_img.squeeze().data.cpu()
        sr_img = FT.to_pil_image(sr_img)
        sr_img.save("result/" + os.path.basename(img_name))
        
psnr = psnr / len(test_images)
print("PSNR: {:>.3f}".format(psnr), flush=True)