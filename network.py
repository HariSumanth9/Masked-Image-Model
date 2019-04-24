import torch
import torch.nn as nn
#from torchsummary import summary
import torch.nn.functional as F

class EncDec(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncDec, self).__init__()
        self.conv1a = nn.Conv2d(3,  64, 3, padding = 1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool1  = nn.MaxPool2d(2, 2, return_indices = True)
        
        self.conv2a = nn.Conv2d(64,  128, 3, padding = 1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding = 1)
        self.pool2  = nn.MaxPool2d(2, 2, return_indices = True)
        
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
        
        
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dconv4d = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dconv4c = nn.Conv2d(512, 512, 3, padding = 1)
        self.dconv4b = nn.Conv2d(512, 512, 3, padding = 1)
        self.dconv4a = nn.Conv2d(512, 256, 3, padding = 1)
        
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dconv3d = nn.Conv2d(512, 256, 3, padding = 1)
        self.dconv3c = nn.Conv2d(256, 256, 3, padding = 1)
        self.dconv3b = nn.Conv2d(256, 256, 3, padding = 1)
        self.dconv3a = nn.Conv2d(256, 128, 3, padding = 1)
        
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dconv2b = nn.Conv2d(256, 128, 3, padding = 1)
        self.dconv2a = nn.Conv2d(128, 64,  3, padding = 1)
        
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dconv1b = nn.Conv2d(128, 64, 3, padding = 1)
        self.dconv1a = nn.Conv2d(64, 64,  3, padding = 1)
        
        self.output  = nn.Conv2d(64, 3, 1)
        
        
    def forward(self, x):
        conv1a = F.relu(self.conv1a(x), inplace = True)
        conv1b = F.relu(self.conv1b(conv1a), inplace = True)
        pool1, idxs1 = self.pool1(conv1b)
        
        conv2a = F.relu(self.conv2a(pool1), inplace = True)
        conv2b = F.relu(self.conv2b(conv2a), inplace = True)
        pool2, idxs2 = self.pool2(conv2b)
        
        conv3a = F.relu(self.conv3a(pool2), inplace = True)
        conv3b = F.relu(self.conv3b(conv3a), inplace = True)
        conv3c = F.relu(self.conv3c(conv3b), inplace = True)
        conv3d = F.relu(self.conv3d(conv3c), inplace = True)
        pool3, idxs3 = self.pool3(conv3d)
        
        conv4a = F.relu(self.conv4a(pool3), inplace = True)
        conv4b = F.relu(self.conv4b(conv4a), inplace = True)
        conv4c = F.relu(self.conv4c(conv4b), inplace = True)
        conv4d = F.relu(self.conv4d(conv4c), inplace = True)
        pool4, idxs4 = self.pool4(conv4d)
        
                
        unpool4 = self.unpool4(pool4, idxs4)
        concat4 = torch.cat([unpool4, conv4d], 1)
        dconv4d = F.relu(self.dconv4d(concat4), inplace = True)
        dconv4c = F.relu(self.dconv4c(dconv4d), inplace = True)
        dconv4b = F.relu(self.dconv4b(dconv4c), inplace = True)
        dconv4a = F.relu(self.dconv4a(dconv4b), inplace = True)
        
        unpool3 = self.unpool3(dconv4a, idxs3)
        concat3 = torch.cat([unpool3, conv3d], 1)
        dconv3d = F.relu(self.dconv3d(concat3), inplace = True)
        dconv3c = F.relu(self.dconv3c(dconv3d), inplace = True)
        dconv3b = F.relu(self.dconv3b(dconv3c), inplace = True)
        dconv3a = F.relu(self.dconv3a(dconv3b), inplace = True)
        
        unpool2 = self.unpool2(dconv3a, idxs2)
        concat2 = torch.cat([unpool2, conv2b], 1)
        dconv2b = F.relu(self.dconv2b(concat2), inplace = True)
        dconv2a = F.relu(self.dconv2a(dconv2b), inplace = True)
        
        unpool1 = self.unpool1(dconv2a, idxs1)
        concat1 = torch.cat([unpool1, conv1b], 1)
        dconv1b = F.relu(self.dconv1b(concat1), inplace = True)
        dconv1a = F.relu(self.dconv1a(dconv1b), inplace = True)        
        return torch.sigmoid(self.output(dconv1a))



class encdec(nn.Module):
	def __init__(self,input_nc,output_nc,inc):
		super(encdec,self).__init__	()

		self.conv0= nn.Conv2d(input_nc,input_nc,1)

		self.conv1a= nn.Conv2d(input_nc,inc,3,padding=(1,1))		
		self.conv1b= nn.Conv2d(inc,inc,3,padding=(1,1))
		self.pool1= nn.MaxPool2d(2, 2, return_indices = True) #4

		self.conv2a= nn.Conv2d(inc,inc*2,3,padding=(1,1))		
		self.conv2b= nn.Conv2d(inc*2,inc*2,3,padding=(1,1))
		self.pool2= nn.MaxPool2d(2, 2, return_indices = True) #9

		self.conv3a= nn.Conv2d(inc*2,inc*4,3,padding=(1,1))		
		self.conv3b= nn.Conv2d(inc*4,inc*4,3,padding=(1,1))
		self.conv3c= nn.Conv2d(inc*4,inc*4,3,padding=(1,1))  
		self.conv3d= nn.Conv2d(inc*4,inc*4,3,padding=(1,1)) 
		self.pool3= nn.MaxPool2d(2, 2, return_indices = True)  #16

		self.conv4a= nn.Conv2d(inc*4,inc*8,3,padding=(1,1))		
		self.conv4b= nn.Conv2d(inc*8,inc*8,3,padding=(1,1))
		self.conv4c= nn.Conv2d(inc*8,inc*8,3,padding=(1,1))
		self.conv4d= nn.Conv2d(inc*8,inc*8,3,padding=(1,1))
		self.pool4= nn.MaxPool2d(2, 2, return_indices = True)  #16
		
		self.linear1= nn.Linear(100352, 3136)
		self.linear2= nn.Linear(3136, 784)
		self.linear3= nn.Linear(784, 3136)
		
		self.dconv0= nn.ConvTranspose2d(16,inc*8,3,padding=(1,1))
		self.unpool0= nn.MaxUnpool2d(2,2)

		self.dconv1a= nn.ConvTranspose2d(inc*16,inc*8,3,padding=(1,1))
		self.dconv1b= nn.ConvTranspose2d(inc*8,inc*8,3,padding=(1,1))
		self.dconv1c= nn.ConvTranspose2d(inc*8,inc*8,3,padding=(1,1))
		self.dconv1d= nn.ConvTranspose2d(inc*8,inc*4,3,padding=(1,1))
		self.unpool1= nn.MaxUnpool2d(2,2)

		self.dconv2a= nn.ConvTranspose2d(inc*8,inc*4,3,padding=(1,1))
		self.dconv2b= nn.ConvTranspose2d(inc*4,inc*4,3,padding=(1,1))
		self.dconv2c= nn.ConvTranspose2d(inc*4,inc*4,3,padding=(1,1))
		self.dconv2d= nn.ConvTranspose2d(inc*4,inc*2,3,padding=(1,1))
		self.unpool2= nn.MaxUnpool2d(2,2)

		self.dconv3a= nn.ConvTranspose2d(inc*4,inc*2,3,padding=(1,1))
		self.dconv3b= nn.ConvTranspose2d(inc*2,inc,3,padding=(1,1))
		self.unpool3= nn.MaxUnpool2d(2,2)

		self.dconv4a= nn.ConvTranspose2d(inc*2,inc,3,padding=(1,1))
		self.dconv4b= nn.ConvTranspose2d(inc,input_nc,3,padding=(1,1))
		self.unpool4= nn.MaxUnpool2d(2,2)

		self.output= nn.Conv2d(input_nc,input_nc,1)
		
		# self.conv_block= conv_block	
		# self.conv_transpose_block= conv_block_transpose	
		# self.upSamplingBlock=upSamplingBlock
		# self.ResidualBlock= ResidualBlock
	

	def forward(self,x):

		conv0= F.relu(self.conv0(x), inplace = True)

		conv1a= F.relu(self.conv1a(conv0), inplace = True)
		conv1b=  F.relu(self.conv1b(conv1a), inplace = True)
		pool1, indices1 = self.pool1(conv1b)

		conv2a= F.relu(self.conv2a(pool1), inplace = True)
		conv2b=  F.relu(self.conv2b(conv2a), inplace = True)
		pool2, indices2= self.pool2(conv2b)
		
		conv3a= F.relu(self.conv3a(pool2), inplace = True)
		conv3b=  F.relu(self.conv3b(conv3a), inplace = True)
		conv3c=  F.relu(self.conv3c(conv3b), inplace = True)
		conv3d=  F.relu(self.conv3d(conv3c), inplace = True)
		pool3, indices3= self.pool3(conv3d)

		conv4a= F.relu(self.conv4a(pool3), inplace = True)
		conv4b=  F.relu(self.conv4b(conv4a), inplace = True)
		conv4c=  F.relu(self.conv4c(conv4b), inplace = True)
		conv4d=  F.relu(self.conv4d(conv4c), inplace = True)
		pool4, indices4= self.pool3(conv4d)

		flatten = pool4.view(pool4.size(0),-1)
		#print(flatten.size())
		linear1= self.linear1(flatten)
		linear2= self.linear2(linear1)
		linear3= self.linear3(linear2)

		dconv = linear1.view(linear1.size(0),16,14,14)
		dconv0 = F.relu(self.dconv0(dconv))

		unpool0= self.unpool0(dconv0,indices4)
		concat0= torch.cat([unpool0,conv4d],1)

		dconv1a= F.relu(self.dconv1a(concat0), inplace = True)
		dconv1b= F.relu(self.dconv1b(dconv1a), inplace = True)
		dconv1c= F.relu(self.dconv1c(dconv1b), inplace = True)
		dconv1d= F.relu(self.dconv1d(dconv1c), inplace = True)

		unpool1= self.unpool1(dconv1d,indices3)
		concat1= torch.cat([unpool1,conv3d],1)

		dconv2a= F.relu(self.dconv2a(concat1), inplace = True)
		dconv2b= F.relu(self.dconv2b(dconv2a), inplace = True)
		dconv2c= F.relu(self.dconv2c(dconv2b), inplace = True)
		dconv2d= F.relu(self.dconv2d(dconv2c), inplace = True)

		unpool2= self.unpool2(dconv2d,indices2)
		concat2= torch.cat([unpool2,conv2b],1)

		dconv3a= F.relu(self.dconv3a(concat2), inplace = True)
		dconv3b= F.relu(self.dconv3b(dconv3a), inplace = True)
		
		unpool3= self.unpool3(dconv3b,indices1)
		concat3= torch.cat([unpool3,conv1b],1)

		dconv4a= F.relu(self.dconv4a(concat3), inplace = True)
		dconv4b= F.relu(self.dconv4b(dconv4a), inplace = True)
		
		output= self.output(dconv4b)

		return torch.sigmoid(output)
