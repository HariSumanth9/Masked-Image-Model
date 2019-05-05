import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDec1(nn.Module):
    def __init__(self):
        super(EncDec1, self).__init__()
        self.conv1  = nn.Conv2d(3,   64,  4, 2, 1, bias = False)
        self.conv2  = nn.Conv2d(64,  128, 4, 2, 1, bias = False)
        self.bn2    = nn.BatchNorm2d(128)
        self.conv3  = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.bn3    = nn.BatchNorm2d(256)
        self.conv4  = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
        self.bn4    = nn.BatchNorm2d(512)

        self.convT4 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
        self.dbn4   = nn.BatchNorm2d(256)
        self.convT3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        self.dbn3   = nn.BatchNorm2d(128)
        self.convT2 = nn.ConvTranspose2d(128, 64,  4, 2, 1, bias = False)
        self.dbn2   = nn.BatchNorm2d(64)
        self.convT1 = nn.ConvTranspose2d(64,  3,   4, 2, 1, bias = False)


    def forward(self, x):
        conv1  = F.leaky_relu(self.conv1(x),               negative_slope = 0.2, inplace = True)
        conv2  = F.leaky_relu(self.bn2(self.conv2(conv1)), negative_slope = 0.2, inplace = True)
        conv3  = F.leaky_relu(self.bn3(self.conv3(conv2)), negative_slope = 0.2, inplace = True)
        conv4  = F.leaky_relu(self.bn4(self.conv4(conv3)), negative_slope = 0.2, inplace = True)

        convT4 = F.relu(self.dbn4(self.convT4(conv4)),  inplace = True)
        convT3 = F.relu(self.dbn3(self.convT3(convT4)), inplace = True)
        convT2 = F.relu(self.dbn2(self.convT2(convT3)), inplace = True)
        convT1 = torch.sigmoid(self.convT1(convT2))
        return convT1


class EncDec2(nn.Module):
    def __init__(self):
        super(EncDec2, self).__init__()
        self.conv1  = nn.Conv2d(3,   64,  3, padding = 1)
        self.conv2  = nn.Conv2d(64,  128, 3, padding = 1)
        self.bn2    = nn.BatchNorm2d(128)
        self.mp2    = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3  = nn.Conv2d(128, 256, 3, padding = 1)
        self.bn3    = nn.BatchNorm2d(256)
        self.mp3    = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4  = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn4    = nn.BatchNorm2d(256)
        self.mp4    = nn.MaxPool2d(2, 2, return_indices = True)

        self.up4    = nn.MaxUnpool2d(2, 2)
        self.dconv4 = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn4   = nn.BatchNorm2d(256)
        self.up3    = nn.MaxUnpool2d(2, 2)
        self.dconv3 = nn.Conv2d(512, 128, 3, padding = 1)
        self.dbn3   = nn.BatchNorm2d(128)
        self.up2    = nn.MaxUnpool2d(2, 2)
        self.dconv2 = nn.Conv2d(256, 64, 3,  padding = 1)
        self.dbn2   = nn.BatchNorm2d(64)
        self.up1    = nn.MaxUnpool2d(2, 2)
        self.dconv1 = nn.Conv2d(128, 64, 3,  padding = 1)
        self.dbn1   = nn.BatchNorm2d(64)        
        self.out    = nn.Conv2d(64, 3, 1)
        

    def forward(self, x):
        conv1      = F.leaky_relu(self.conv1(x),               negative_slope = 0.2, inplace = True)
        conv2      = F.leaky_relu(self.bn2(self.conv2(conv1)), negative_slope = 0.2, inplace = True)
        mp2, idxs2 = self.mp2(conv2) 
        conv3      = F.leaky_relu(self.bn3(self.conv3(mp2)), negative_slope = 0.2, inplace = True)
        mp3, idxs3 = self.mp3(conv3) 
        conv4      = F.leaky_relu(self.bn4(self.conv4(mp3)), negative_slope = 0.2, inplace = True)
        mp4, idxs4 = self.mp4(conv4)
        
        up4    = self.up4(mp4, idxs4)
        dconv4 = F.relu(self.dbn4(self.dconv4(torch.cat([up4, conv4], 1))), inplace = True)
        up3    = self.up3(dconv4, idxs3)
        dconv3 = F.relu(self.dbn3(self.dconv3(torch.cat([up3, conv3], 1))), inplace = True)
        up2    = self.up2(dconv3, idxs2)
        dconv2 = F.relu(self.dbn2(self.dconv2(torch.cat([up2, conv2], 1))), inplace = True)
        dconv1 = F.relu(self.dbn1(self.dconv1(torch.cat([dconv2, conv1], 1))), inplace = True)
        out    = self.out(dconv1)
        return torch.sigmoid(out)


class Classifier1(nn.Module):
    def __init__(self, prenet):
        super(Classifier1, self).__init__()
        self.prenet = prenet
        self.conv1  = self.prenet.conv1
        self.conv2  = self.prenet.conv2
        self.bn2    = self.prenet.bn2
        self.conv3  = self.prenet.conv3
        self.bn3    = self.prenet.bn3
        self.conv4  = self.prenet.conv4
        self.bn4    = self.prenet.bn4
        #self.out    = nn.Conv2d(512, 200, 4, stride = 4)
        self.out    = nn.Linear(512*4*4, 200)
        
        
    def forward(self, x):
        conv1   = F.leaky_relu(self.conv1(x),               negative_slope = 0.2, inplace = True)
        conv2   = F.leaky_relu(self.bn2(self.conv2(conv1)), negative_slope = 0.2, inplace = True)
        conv3   = F.leaky_relu(self.bn3(self.conv3(conv2)), negative_slope = 0.2, inplace = True)
        conv4   = F.leaky_relu(self.bn4(self.conv4(conv3)), negative_slope = 0.2, inplace = True)
        #out     = self.out(conv4).squeeze(3).squeeze(2)
        flatten = conv4.view(-1, 512*4*4)
        out     = self.out(flatten)
        return out



class Classifier2(nn.Module):
    def __init__(self, prenet):
        super(Classifier2, self).__init__()
        self.prenet = prenet
        self.conv1  = self.prenet.conv1
        self.conv2  = self.prenet.conv2
        self.bn2    = self.prenet.bn2
        self.mp2    = self.prenet.mp2
        self.conv3  = self.prenet.conv3
        self.bn3    = self.prenet.bn3
        self.mp3    = self.prenet.mp3
        self.conv4  = self.prenet.conv4
        self.bn4    = self.prenet.bn4
        self.mp4    = self.prenet.mp4
        self.linear = nn.Linear(256*8*8, 200)
        
    def forward(self, x):
        conv1      = F.leaky_relu(self.conv1(x),               negative_slope = 0.2, inplace = True)
        conv2      = F.leaky_relu(self.bn2(self.conv2(conv1)), negative_slope = 0.2, inplace = True)
        mp2, idxs2 = self.mp2(conv2) 
        conv3      = F.leaky_relu(self.bn3(self.conv3(mp2)), negative_slope = 0.2, inplace = True)
        mp3, idxs3 = self.mp3(conv3) 
        conv4      = F.leaky_relu(self.bn4(self.conv4(mp3)), negative_slope = 0.2, inplace = True)
        mp4, idxs4 = self.mp4(conv4)
        flatten    = mp4.view(-1, 256*8*8)
        out        = self.linear(flatten)
        return out
