import torch
import torch.nn as nn
import torch.nn.functional as F

class EncDec(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncDec, self).__init__()
        self.conv1a = nn.Conv2d(3,  64, 3, padding = 1)
        self.conv1b = nn.Conv2d(64, 64, 3, stride = 2,   padding = 1)
        self.bn1b   = nn.BatchNorm2d(64)
        
        self.conv2a = nn.Conv2d(64,  128, 3, padding = 1)
        self.conv2b = nn.Conv2d(128, 128, 3, stride = 2, padding = 1)
        self.bn2b   = nn.BatchNorm2d(128)
        
        self.conv3a = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn3b   = nn.BatchNorm2d(256)
        self.conv3c = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3d = nn.Conv2d(256, 256, 3, stride = 2, padding = 1)
        self.bn3d   = nn.BatchNorm2d(256)
        
        self.conv4a = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn4b   = nn.BatchNorm2d(512)
        self.conv4c = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4d = nn.Conv2d(512, 512, 3, stride = 2, padding = 1)
        self.bn4d   = nn.BatchNorm2d(512)
        
        self.dconv4d = nn.ConvTranspose2d(512, 512, 2, stride = 2)
        self.dconv4c = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn4c   = nn.BatchNorm2d(512)
        self.dconv4b = nn.Conv2d(512, 512, 3, padding = 1)
        self.dconv4a = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn4a   = nn.BatchNorm2d(256)
        
        self.dconv3d = nn.ConvTranspose2d(256, 256, 2, stride = 2)
        self.dconv3c = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn3c   = nn.BatchNorm2d(256)
        self.dconv3b = nn.Conv2d(256, 256, 3, padding = 1)
        self.dconv3a = nn.Conv2d(256, 128, 3, padding = 1)
        self.dbn3a   = nn.BatchNorm2d(128)
        
        self.dconv2b = nn.ConvTranspose2d(128, 128, 2, stride = 2)
        self.dconv2a = nn.Conv2d(256, 128,  3, padding = 1)
        self.dbn2a    = nn.BatchNorm2d(128)
        
        self.dconv1b = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.dconv1a = nn.Conv2d(128, 64,  3, padding = 1)
        self.dbn1a   = nn.BatchNorm2d(64)
        
        self.output  = nn.Conv2d(64, 3, 1)
    
    
    def forward(self, x):
        conv1a = F.leaky_relu(self.conv1a(x),                 negative_slope = 0.2, inplace = True)
        conv1b = F.leaky_relu(self.bn1b(self.conv1b(conv1a)), negative_slope = 0.2, inplace = True)
        
        conv2a = F.leaky_relu(self.conv2a(conv1b),            negative_slope = 0.2, inplace = True)
        conv2b = F.leaky_relu(self.bn2b(self.conv2b(conv2a)), negative_slope = 0.2, inplace = True)
        
        conv3a = F.leaky_relu(self.conv3a(conv2b),            negative_slope = 0.2, inplace = True)
        conv3b = F.leaky_relu(self.bn3b(self.conv3b(conv3a)), negative_slope = 0.2, inplace = True)
        conv3c = F.leaky_relu(self.conv3c(conv3b),            negative_slope = 0.2, inplace = True)
        conv3d = F.leaky_relu(self.bn3d(self.conv3d(conv3c)), negative_slope = 0.2, inplace = True)
        
        conv4a = F.leaky_relu(self.conv4a(conv3d),            negative_slope = 0.2, inplace = True)
        conv4b = F.leaky_relu(self.bn4b(self.conv4b(conv4a)), negative_slope = 0.2, inplace = True)
        conv4c = F.leaky_relu(self.conv4c(conv4b),            negative_slope = 0.2, inplace = True)
        conv4d = F.leaky_relu(self.bn4d(self.conv4d(conv4c)), negative_slope = 0.2, inplace = True)
        
        dconv4d = F.relu(self.dconv4d(conv4d), inplace = True)
        dconv4c = F.relu(self.dbn4c(self.dconv4c(torch.cat([dconv4d, conv4c], dim = 1))), inplace = True)
        dconv4b = F.relu(self.dconv4b(dconv4c), inplace = True)
        dconv4a = F.relu(self.dbn4a(self.dconv4a(dconv4b)), inplace = True)
        
        dconv3d = F.relu(self.dconv3d(dconv4a), inplace = True)
        dconv3c = F.relu(self.dbn3c(self.dconv3c(torch.cat([dconv3d, conv3c], dim = 1))), inplace = True)
        dconv3b = F.relu(self.dconv3b(dconv3c), inplace = True)
        dconv3a = F.relu(self.dbn3a(self.dconv3a(dconv3b)), inplace = True)
        
        dconv2b = F.relu(self.dconv2b(dconv3a), inplace = True)
        dconv2a = F.relu(self.dbn2a(self.dconv2a(torch.cat([dconv2b, conv2a], dim = 1))), inplace = True)
        
        dconv1b = F.relu(self.dconv1b(dconv2a), inplace = True)
        dconv1a = F.relu(self.dbn1a(self.dconv1a(torch.cat([dconv1b, conv1a], dim = 1))), inplace = True)
        return torch.sigmoid(self.output(dconv1a))
