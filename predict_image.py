import os
import h5py
import numpy as np
from PIL import Image
import matplotlib as plt
from matplotlib import cm as CM

import torch
from torchvision import transforms
from model import CANNet

import warnings
warnings.filterwarnings('ignore')

# test paths
root = '...Input/ShanghaiTech'
img_path = os.path.join(root, 'part_B/test_data/images', 'IMG_1.jpg')
gt_path = os.path.join(root, 'part_B/test_data/ground-truth', 'IMG_1.h5')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

# load model
model = CANNet()
model = model.cuda()
checkpoint = torch.load('part_B_pre.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# get image
img = Image.open(img_path).convert('RGB')
img = transform(img).cuda()
img = img.unsqueeze(0)

# get density
density = model(img).data.cpu().numpy()
pred_sum = density.sum()

# normalize density
density_min = np.min(density, axis=tuple(range(density.ndim-1)), keepdims=1)
density_max = np.max(density, axis=tuple(range(density.ndim-1)), keepdims=1)
density = (density - density_min)/ (density_max - density_min)

# apply colormap
im = Image.fromarray((CM.jet(density.squeeze())*255).astype(np.uint8))
if im.mode != 'RGB':
    im = im.convert('RGB')
#im.save('prediction.jpg')

groundtruth = h5py.File(gt_path, 'r')
groundtruth = np.asarray(groundtruth['density'])
original_sum = np.sum(groundtruth)

print('original: ',original_sum)
print('predicted: ',pred_sum)

# apply transparency
im.putalpha(128)
#im.save('transparent.png')

# overlay background and foreground
background = Image.open(img_path).convert('RGBA')
width, height = background.size
im = im.resize((width, height))
background.paste(im, (0, 0), im)
background.save('density.png')