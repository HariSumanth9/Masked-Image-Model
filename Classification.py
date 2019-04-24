import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import EncDec


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)


class Classifier1(nn.Module):
    def __init__(self, n_classes):
        super(Classifier1, self).__init__()
        self.conv1a = nn.Conv2d(3, 64, 3,  padding = 1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool1  = nn.MaxPool2d(2, 2)
        
        self.conv2a = nn.Conv2d(64, 128, 3,  padding = 1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding = 1)
        self.pool2  = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128*8*8, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        conv1a = F.relu(self.conv1a(x), inplace = True)
        conv1b = F.relu(self.conv1b(conv1a), inplace = True)
        pool1  = self.pool1(conv1b)
        
        conv2a = F.relu(self.conv2a(pool1), inplace = True)
        conv2b = F.relu(self.conv2b(conv2a), inplace = True)
        pool2  = self.pool2(conv2b)
        
        flatten = pool2.view(-1, 128*8*8)
        fc1 = F.relu(self.fc1(flatten), inplace = True)
        fc2 = self.fc2(fc1)
        return fc2



class Classifier2(nn.Module):
    def __init__(self, n_classes, net):
        super(Classifier2, self).__init__()
        self.prenet = net
        self.conv1a = self.prenet.conv1a
        self.conv1b = self.prenet.conv1b
        self.pool1  = self.prenet.pool1
        
        self.conv2a = self.prenet.conv2a
        self.conv2b = self.prenet.conv2b
        self.pool2  = self.prenet.pool2
        
        self.fc1 = nn.Linear(128*8*8, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        conv1a = F.relu(self.conv1a(x), inplace = True)
        conv1b = F.relu(self.conv1b(conv1a), inplace = True)
        pool1,idxs1  = self.pool1(conv1b)
        
        conv2a = F.relu(self.conv2a(pool1), inplace = True)
        conv2b = F.relu(self.conv2b(conv2a), inplace = True)
        pool2, idxs2  = self.pool2(conv2b)
        
        flatten = pool2.view(-1, 128*8*8)
        fc1 = F.relu(self.fc1(flatten), inplace = True)
        fc2 = self.fc2(fc1)
        return fc2




netED = EncDec(3, 3)
netED.load_state_dict(torch.load('saved_model_large_lr_small.pth'))
netED.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
	netED.cuda()

net = Classifier2(10, netED)
if use_gpu:
	net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-3)


epochs = 20
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
    for data in trainloader:
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.long().cuda()
        outputs = net(images)
        #print(outputs)
        #print(outputs[0].sum())
        optimizer.zero_grad()
        loss    = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        loss.backward()
        optimizer.step()
        epochTrainLoss += loss.item()
        epochTrainAcc  += (predicted == labels).sum().item()/images.size(0)


    net.train(False)
    for data in testloader:
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.long().cuda()
        outputs = net(images)
        loss    = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        epochValidLoss += loss.item()
        epochValidAcc  += (predicted == labels).sum().item()/images.size(0)
    epochTrainAcc  = epochTrainAcc/len(trainloader)
    epochValidAcc  = epochValidAcc/len(testloader)
    epochTrainLoss = epochTrainLoss/len(trainloader)
    epochValidLoss = epochValidLoss/len(testloader)
    trainLoss.append(epochTrainLoss)
    validLoss.append(epochValidLoss)
    trainAcc.append(epochTrainAcc  )
    validAcc.append(epochTrainAcc  )

    if epoch!=0:
        if(epochValidAcc > bestValidAcc):
            bestValidAcc = epochValidAcc
            torch.save(net.state_dict(), 'classifier_large_masklr_small.pth')
    else: 
        bestValidAcc = epochValidAcc     
    print('[Epoch: {:.0f}/{:.0f}| Train Loss: {:.5f}| Valid Loss: {:.5f}| Train Acc: {:.5f}| Valid Acc: {:.5f}]'.format(epoch+1, epochs, epochTrainLoss, epochValidLoss, epochTrainAcc, epochValidAcc))
