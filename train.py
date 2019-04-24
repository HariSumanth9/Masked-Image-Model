import torch
import random
import copy
import tqdm
import warnings
warnings.filterwarnings("ignore")


def train(model, trainLoader, validLoader, epoch, use_gpu, args):
	trainRunningLoss = 0
	validRunningLoss = 0
	net       = model['net']
	optimizer = model['optimizer']
	criterion = model['criterion']

	net.train(True)
	for data in tqdm.tqdm(trainLoader):
		images, targets = data
		if use_gpu:
			images  = images.cuda(0)
			targets = targets.cuda(0)
		outputs = net(images)
		optimizer.zero_grad()
		loss = criterion(outputs.view(-1), targets.view(-1))
		loss.backward()
		optimizer.step()
		trainRunningLoss += loss.item()

	net.train(False)
	for data in tqdm.tqdm(validLoader):
		images, targets = data
		if use_gpu:
			images  = images.cuda(0)
			targets = targets.cuda(0)
		outputs = net(images)
		loss = criterion(outputs.view(-1), targets.view(-1))
		validRunningLoss += loss.item() 
	return trainRunningLoss/(len(trainLoader)), validRunningLoss/(len(validLoader))


