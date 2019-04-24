import sys, os
import cv2
import glob
import copy
import random
import numpy as np
import pandas as pd
#import visdom
import arg_parser
#from PIL import Image
import matplotlib.image as mpimage
#import matplotlib.pyplot as plt
from scipy.misc import imsave

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.transforms import RandomSizedCrop, RandomHorizontalFlip
from torchvision.transforms import ToTensor, ToPILImage


from train import train
from network import EncDec, encdec
from load_data import FashionDataset, RemoveRandomBlocks

use_gpu = torch.cuda.is_available()

    #rrb_transform = RemoveRandomBlocks(args.no_of_patches, args.patch_size, args.patch_multiplier)
pil_transform = ToPILImage()
transform     = transforms.Compose([
RandomHorizontalFlip(),
])
tensor_transform = ToTensor()

trainDataset = FashionDataset('./tiny_imagenet', 'train', pil_transform, transform, tensor_transform)
validDataset = FashionDataset('./tiny_imagenet', 'valid', pil_transform, transform, tensor_transform)

trainLoader  = DataLoader(trainDataset, batch_size = 256, shuffle = True)
validLoader  = DataLoader(validDataset, batch_size = 256, shuffle = False)

net = EncDec(3, 3)
net.load_state_dict(torch.load('saved_model.pth'))
net.eval()
net.cuda()
if use_gpu:
    net = net.cuda(0)

i = 0;
for data in validLoader:
    inputs, targets = data
    if use_gpu:
        inputs  = inputs.cuda()
        targets = targets.cuda()
    outputs = net(inputs)
    outputs = outputs.cpu().detach().numpy()
    inputs  = inputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    for j in range(outputs.shape[0]):
        output = outputs[j].transpose(1, 2, 0)*255
        input_ = inputs[j].transpose(1, 2, 0)*255
        target = targets[j].transpose(1, 2, 0)*255

        name = './saved_images/inputs/' + str(i) + str('_') + str(j) + str('.jpg')
        imsave(name, input_)
        name = './saved_images/outputs/' + str(i) + str('_') + str(j) + str('.jpg')
        imsave(name, output)
        name = './saved_images/targets/' + str(i) + str('_') + str(j) + str('.jpg')
        imsave(name, target)
    i = i+1




