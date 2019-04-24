import glob
import os
import random
import copy
import numpy as np
from PIL import Image
import matplotlib.image as mpimage
from torch.utils.data import Dataset



class RemoveRandomBlocks(object):
    def __init__(self, noOfSeeds, patchSize, patchMultiplier):
        self.noOfSeeds       = noOfSeeds
        self.patchSize       = patchSize
        self.patchMultiplier = patchMultiplier
        
    def __call__(self, sample):
        height, width, channels = sample.shape
        sample_copy = copy.copy(sample)
        for i in range(self.noOfSeeds):
            finalPatchSize    = copy.copy(self.patchSize)
            finalPatchSize[0] = int(np.floor(self.patchSize[0]*self.patchMultiplier))
            finalPatchSize[1] = int(np.floor(self.patchSize[1]*self.patchMultiplier))
            xSeed = random.randrange(finalPatchSize[0], height - finalPatchSize[0])
            ySeed = random.randrange(finalPatchSize[1], width - finalPatchSize[1])
            sample_copy[xSeed: xSeed + finalPatchSize[0], ySeed: ySeed + finalPatchSize[1]] = 0
        return sample_copy


class FashionDataset(Dataset):
    def __init__(self, root, opt, pil_transform = None, transform = None, tensor_transform = None):
        self.root    = root
        self.images  = sorted(glob.glob(os.path.join(self.root, opt, 'masks_large', '*.jpg')))
        #print(os.path.join(self.root, opt, 'masks', '*.png'))
        self.targets = sorted(glob.glob(os.path.join(self.root, opt, 'orig', '*.jpg')))
        self.pil_transform = pil_transform
        self.transform     = transform
        self.tensor_transform = tensor_transform

    def __getitem__(self, index):
        image  = mpimage.imread(self.images[index])
        target = mpimage.imread(self.targets[index])
        prob = random.uniform(0, 1)
        '''
        if prob >= 0.8:
            image = copy.copy(target) 
        '''
        if self.pil_transform is not None:
            image  = self.pil_transform(image)
            target = self.pil_transform(target)
        if self.transform is not None:
            image, target  = self.transform([image, target])
        if self.tensor_transform is not None:
            image  = self.tensor_transform(image)
            target = self.tensor_transform(target)
        return image, target

    def __len__(self):
        return len(self.images)
