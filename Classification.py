import tqdm
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import ToTensor, ToPILImage, RandomHorizontalFlip
from torch.utils.data import Dataset, DataLoader

import arg_parser
#from network2 import EncDec
from network3 import EncDec1, Classifier1, EncDec2, Classifier2
from load_data import ClassificationDataset

args = arg_parser.parse_arguments()

pil_transform = ToPILImage()
transform     = transforms.Compose([
    RandomHorizontalFlip(),
    ])
tensor_transform = ToTensor()

trainDataset = ClassificationDataset('./Dataset/Classifier/', 'Train_30/', 'train_labels_30.npy',  pil_transform, transform, tensor_transform)
validDataset = ClassificationDataset('./Dataset/Classifier/', 'Valid/', 'valid_labels.npy',        pil_transform, transform, tensor_transform)
trainLoader  = DataLoader(trainDataset, batch_size = 128, shuffle = True)
validLoader  = DataLoader(validDataset, batch_size = 128, shuffle = False)


'''
class Classifier1(nn.Module):
    def __init__(self, n_classes):
        super(Classifier1, self).__init__()
        self.conv1a = nn.Conv2d(3, 64, 3,  padding = 1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool1  = nn.MaxPool2d(2, 2)
        
        self.conv2a = nn.Conv2d(64, 128, 3,  padding = 1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding = 1)
        self.pool2  = nn.MaxPool2d(2, 2)
        
        self.conv3a = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3c = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3d = nn.Conv2d(256, 256, 3, padding = 1)
        self.pool3  = nn.MaxPool2d(2, 2, return_indices = True)
        
        self.conv4a = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4c = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4d = nn.Conv2d(512, 512, 3, padding = 1)
        self.pool4  = nn.MaxPool2d(2, 2, return_indices = True)
        
        self.fc1 = nn.Linear(128*16*16, 2048)
        #self.fc2 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(2048, n_classes)
        
    def forward(self, x):
        conv1a = F.relu(self.conv1a(x), inplace = True)
        conv1b = F.relu(self.conv1b(conv1a), inplace = True)
        pool1  = self.pool1(conv1b)
        
        conv2a = F.relu(self.conv2a(pool1), inplace = True)
        conv2b = F.relu(self.conv2b(conv2a), inplace = True)
        pool2  = self.pool2(conv2b)
        
        conv3a = F.relu(self.conv3a(pool2),  inplace = True)
        conv3b = F.relu(self.conv3b(conv3a), inplace = True)
        conv3c = F.relu(self.conv3c(conv3b), inplace = True)
        conv3d = F.relu(self.conv3d(conv3c), inplace = True)
        pool3, idxs3 = self.pool3(conv3d)
 
        conv4a = F.relu(self.conv4a(pool3),  inplace = True)
        conv4b = F.relu(self.conv4b(conv4a), inplace = True)
        conv4c = F.relu(self.conv4c(conv4b), inplace = True)
        conv4d = F.relu(self.conv4d(conv4c), inplace = True)
        pool4, idxs4 = self.pool4(conv4d)
       
        flatten = pool2.view(-1, 128*16*16)
        fc1 = F.relu(self.fc1(flatten), inplace = True)
        #fc2 = F.relu(self.fc2(fc1),     inplace = True)
        fc2 = self.fc2(fc1)
        return fc2
        



class Classifier2(nn.Module):
    def __init__(self, n_classes, net):
        super(Classifier2, self).__init__()
        self.prenet = net
        self.conv1a = self.prenet.conv1a
        self.do1a   = nn.Dropout(p = 0.05)
        self.conv1b = self.prenet.conv1b
        self.bn1b   = self.prenet.bn1b
        self.do1b   = nn.Dropout(p = 0.05)

        self.conv2a = self.prenet.conv2a
        self.do2a   = nn.Dropout(p = 0.05)
        self.conv2b = self.prenet.conv2b
        self.bn2b   = self.prenet.bn2b
        self.do2b   = nn.Dropout(p = 0.05)
        
        self.conv3a = self.prenet.conv3a
        self.conv3b = self.prenet.conv3b
        self.conv3c = self.prenet.conv3c
        self.conv3d = self.prenet.conv3d
        self.pool3  = self.prenet.pool3

        self.conv4a = self.prenet.conv4a
        self.conv4b = self.prenet.conv4b
        self.conv4c = self.prenet.conv4c
        self.conv4d = self.prenet.conv4c
        self.pool4  = self.prenet.pool4
        
        self.fc1 = nn.Linear(128*16*16, 2048)
        self.do1   = nn.Dropout(p = 0.05)
        #self.fc2 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(2048, n_classes)
        
    def forward(self, x):
        conv1a = F.leaky_relu(self.do1a(self.conv1a(x)),                 negative_slope = 0.2, inplace = True)
        conv1b = F.leaky_relu(self.do1b(self.bn1b(self.conv1b(conv1a))), negative_slope = 0.2, inplace = True)
        
        conv2a = F.leaky_relu(self.do2a(self.conv2a(conv1b)),            negative_slope = 0.2, inplace = True)
        conv2b = F.leaky_relu(self.do2b(self.bn2b(self.conv2b(conv2a))), negative_slope = 0.2, inplace = True)
        
        conv3a = F.relu(self.conv3a(pool2),  inplace = True)
        conv3b = F.relu(self.conv3b(conv3a), inplace = True)
        conv3c = F.relu(self.conv3c(conv3b), inplace = True)
        conv3d = F.relu(self.conv3d(conv3c), inplace = True)
        pool3, idxs3 = self.pool3(conv3d)
 
        conv4a = F.relu(self.conv4a(pool3),  inplace = True)
        conv4b = F.relu(self.conv4b(conv4a), inplace = True)
        conv4c = F.relu(self.conv4c(conv4b), inplace = True)
        conv4d = F.relu(self.conv4d(conv4c), inplace = True)
        pool4, idxs4 = self.pool4(conv4d)
       
        flatten = conv2b.view(-1, 128*16*16)
        fc1 = F.relu(self.do1(self.fc1(flatten)), inplace = True)
        #fc2 = F.relu(self.fc2(fc1),     inplace = True)
        fc2 = self.fc2(fc1)
        return fc2
'''


netED = EncDec2()
netED.load_state_dict(torch.load('saved_model_largemask(ct)_encdec2.pth'))
netED.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
	netED = netED.cuda()

net = Classifier2(netED)
'''
net = models.vgg16(pretrained = False)
net.classifier[0] = nn.Linear(2048, 1024, bias = True)
net.classifier[3] = nn.Linear(1024, 1024, bias = True)
net.classifier[6] = nn.Linear(1024, 200, bias = True)
'''
if use_gpu:
	net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-5)


epochs = 200
trainLoss = []
validLoss = []
trainAcc  = []
validAcc  = []
for epoch in range(epochs):
    epochTrainLoss = 0
    epochValidLoss = 0
    epochTrainAcc  = 0
    epochValidAcc  = 0

    net.train(True)
    for data in tqdm.tqdm(trainLoader):
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.long().cuda()
        outputs = net(images)
        optimizer.zero_grad()
        loss    = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        loss.backward()
        optimizer.step()
        epochTrainLoss += loss.item()
        epochTrainAcc  += (predicted == labels).sum().item()/images.size(0)


    net.train(False)
    for data in tqdm.tqdm(validLoader):
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.long().cuda()
        outputs = net(images)
        loss    = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        epochValidLoss += loss.item()
        epochValidAcc  += (predicted == labels).sum().item()/images.size(0)
    epochTrainAcc  = epochTrainAcc/len(trainLoader)
    epochValidAcc  = epochValidAcc/len(validLoader)
    epochTrainLoss = epochTrainLoss/len(trainLoader)
    epochValidLoss = epochValidLoss/len(validLoader)
    trainLoss.append(epochTrainLoss)
    validLoss.append(epochValidLoss)
    trainAcc.append(epochTrainAcc  )
    validAcc.append(epochTrainAcc  )

    if epoch!=0:
        if(epochValidAcc > bestValidAcc):
            bestValidAcc = epochValidAcc
            torch.save(net.state_dict(), 'largemask_classifier2.pth')
    else: 
        bestValidAcc = epochValidAcc     
    print('[Epoch: {:.0f}/{:.0f}| Train Loss: {:.5f}| Valid Loss: {:.5f}| Train Acc: {:.5f}| Valid Acc: {:.5f}]'.format(epoch+1, epochs, epochTrainLoss, epochValidLoss, epochTrainAcc, epochValidAcc))
