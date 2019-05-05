import sys, os
import cv2
import glob
import copy
import random
import numpy as np
import pandas as pd
import arg_parser
from PIL import Image
import matplotlib.image as mpimage

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.transforms import RandomSizedCrop, RandomHorizontalFlip
from torchvision.transforms import ToTensor, ToPILImage


from train import train
#from network import EncDec, encdec
#from network2 import EncDec
from network3 import EncDec1, EncDec2
from load_data import TinyDataset, RemoveRandomBlocks


def main(args):
    trainLossList = []
    validLossList = []
    use_gpu = torch.cuda.is_available()

    pil_transform = ToPILImage()
    transform     = transforms.Compose([
        RandomHorizontalFlip(),
    ])
    tensor_transform = ToTensor()

    trainDataset = TinyDataset('./Dataset/EncDec/', 'Train_70/', pil_transform, transform, tensor_transform)
    validDataset = TinyDataset('./Dataset/EncDec/', 'Valid/',    pil_transform, transform, tensor_transform)
    trainLoader  = DataLoader(dataset=trainDataset, batch_size=128, shuffle=True)
    validLoader  = DataLoader(dataset=validDataset, batch_size=128, shuffle=False)
    net = EncDec2()
    net.load_state_dict(torch.load('saved_model_smallmask_encdec2.pth'))
    net.eval()
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
                torch.save(net.state_dict(), 'saved_model_largemask(ct)_encdec2.pth')
        else: 
            bestValidLoss = epochValidLoss
        trainLossList.append(epochTrainLoss)
        validLossList.append(epochValidLoss)
        print('[Epoch: {:.0f}/{:.0f}| Train Loss: {:.5f}| Valid Loss: {:.5f}]'.format(epoch+1, args.num_epoch, epochTrainLoss, epochValidLoss))
    print('Training Ends')
    torch.save(net.state_dict(), 'saved_model_largemask(ct)_encdec2_final.pth')
    '''
    plt.plot(range(len(trainLossList)), trainLossList, 'r--')
    plt.plot(range(len(validLossList)), validLossList, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();
    '''


if __name__ == '__main__':
    args = arg_parser.parse_arguments()
    main(args)




