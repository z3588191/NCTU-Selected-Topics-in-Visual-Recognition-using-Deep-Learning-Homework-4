from PIL import Image
import glob
import os

crop_size = 200
stride = 80

train_images = glob.glob('training_hr_images/*.png')
for image_name in train_images:
    img = Image.open(image_name).convert('RGB')
    w, h = img.size
    n = 0
    for i in range(0, w, stride):
        left = i
        if left + crop_size > w:
            right = w
            left = min(0, w - crop_size)
        else:
            right = left + crop_size
        
        for j in range(0, h, stride):
            top = j
            if top + crop_size > h:
                bottom = h
                bottom = min(0, h - crop_size)
            else:
                bottom = top + crop_size
                
            crop_img = img.crop((left, top, right, bottom))
            n += 1
            name = "crop_images/{}_{}.png".format(os.path.splitext(os.path.basename(image_name))[0], n)
            crop_img.save(name)
            
            if bottom == h:
                break
        
        if right == w:
            break