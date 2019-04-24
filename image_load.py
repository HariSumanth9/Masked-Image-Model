import glob
import os
import random
from PIL import Image
import matplotlib.image as mpimage
from torch.utils.data import Dataset



class FashionDataset(Dataset):
    def __init__(self, root, opt, transform = None, target_transform = None):
        self.root   = root
        self.images = glob.glob(os.path.join(self.root, 'orig', opt , '*.jpg'))
        self.masks  = glob.glob(os.path.join(self.root, 'masked', opt , '*.jpg'))
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        image = Image.open(self.images[index])
        mask  = Image.open(self.masks[index] )
        image = image.convert('RGB')
        mask  = mask.convert('RGB' )
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask  = self.target_transform(mask)
        return image, mask


    def __len__(self):
        return len(self.images)


