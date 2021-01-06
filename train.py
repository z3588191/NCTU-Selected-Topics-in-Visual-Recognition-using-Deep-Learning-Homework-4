import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from RFDN.RFDN import *
from dataset import *
from utils import *
from tqdm import tqdm


# hyperparameters
torch.cuda.set_device(0)
scale = 3
patch_size = 192
workers = 8
batch_size = 16
test_freq = 1
epochs = 1000
lr = 0.0005
step_size = 1000
gamma = 0.5


# dataset & dataloader
train_dataset = PatchDataset(scale, patch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


# model & optimizer
model = RFDN(upscale=3).cuda()
criterion = nn.L1Loss().cuda()
trainable = filter(lambda x: x.requires_grad, model.parameters())
optimizer = optim.Adam(trainable, lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# training
best_loss = 0.
for epoch in range(epochs):
    avg_loss = 0.
    train_psnr = 0.
    start = time.time()
    model.train()
    tqbar = tqdm(train_dataloader)
    for batch_idx, (hr_img, lr_img) in enumerate(tqbar):
        hr_img, lr_img = hr_img.cuda(), lr_img.cuda()
        optimizer.zero_grad()
        
        sr_img = model(lr_img)
        loss = criterion(sr_img, hr_img)
        avg_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            train_psnr += calc_psnr(sr_img[:, 0].unsqueeze(1).clamp(0, 1), hr_img[:, 0].unsqueeze(1))
    
    scheduler.step()
    avg_loss = avg_loss / len(train_dataloader)
    train_psnr = train_psnr / len(train_dataloader)
    
    print("Epoch {:>4d}, loss: {:>7.4f}, PSNR: {:>7.4f}, time: {:>7.2f}".format(epoch + 1, 
            avg_loss, train_psnr, end - start), flush=True)

    if avg_loss > best_loss:
        best_psnr = avg_loss
        torch.save(model.state_dict(), "RFDN.pkl")