import sys, os
import cv2
import glob
import copy
import random
import numpy as np
import pandas as pd
#import visdom
import arg_parser
from PIL import Image
import matplotlib.image as mpimage
import matplotlib.pyplot as plt

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




def main(args):
    trainLossList = []
    validLossList = []
    use_gpu = torch.cuda.is_available()

    #rrb_transform = RemoveRandomBlocks(args.no_of_patches, args.patch_size, args.patch_multiplier)
    pil_transform = ToPILImage()
    transform     = transforms.Compose([
        RandomHorizontalFlip(),
        ])
    tensor_transform = ToTensor()

    trainDataset = FashionDataset('./tiny_imagenet', 'train', pil_transform, transform, tensor_transform)
    validDataset = FashionDataset('./tiny_imagenet', 'valid', pil_transform, transform, tensor_transform)

    trainLoader  = DataLoader(trainDataset, batch_size = args.batch_size, shuffle = True)
    validLoader  = DataLoader(validDataset, batch_size = args.batch_size, shuffle = False)

    net = EncDec(3, 3)
    if use_gpu:
        net = net.cuda(0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)

    model = {
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer
    }
    print('Training Begins')
    for epoch in range(args.num_epoch):
        epochTrainLoss, epochValidLoss = train(model, trainLoader, validLoader, epoch, use_gpu, args)
        if epoch!=0:
            if(epochValidLoss < bestValidLoss):
                bestValidLoss = epochValidLoss
                torch.save(net.state_dict(), 'saved_model.pth')
        else: 
            bestValidLoss = epochValidLoss
        trainLossList.append(epochTrainLoss)
        validLossList.append(epochValidLoss)
        print('[Epoch: {:.0f}/{:.0f}| Train Loss: {:.5f}| Valid Loss: {:.5f}]'.format(epoch+1, args.num_epoch, epochTrainLoss, epochValidLoss))
    print('Training Ends')
    plt.plot(range(len(trainLossList)), trainLossList, 'r--')
    plt.plot(range(len(validLossList)), validLossList, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();



if __name__ == '__main__':
    args = arg_parser.parse_arguments()
    main(args)




