import os
import cv2
import random
import scipy.misc
import numpy as np 
from PIL import Image
import copy
from glob import glob
from scipy.misc import imsave
import matplotlib.image as mpimage

inputFileNameExt = '*jpg' #File name extension
inDataPath   = './data/orig/'
outDataPath  = './data/masked/'
imageFolders = ['train','val']
noOfSeeds    = 2


def generateMaskedImages(imageIn, imageSize, noOfSeeds, patchSize, patchMultiplier):
    for i in range(noOfSeeds):
        finalPatchSize    = patchSize
        finalPatchSize[0] = int(np.floor(patchSize[0]*patchMultiplier))
        finalPatchSize[1] = int(np.floor(patchSize[1]*patchMultiplier))
        xSeed = random.randrange(finalPatchSize[0], imageSize[0]-finalPatchSize[0])
        ySeed = random.randrange(finalPatchSize[1], imageSize[1]-finalPatchSize[1])
        imageIn[xSeed:xSeed+finalPatchSize[0], ySeed:ySeed+finalPatchSize[1]] = 0
    return imageIn


folders_list = os.listdir('./tiny-imagenet-200/train')
output_images = './tiny_imagenet/train/orig/'
output_masks  = './tiny_imagenet/train/masks/'
i = 0
for folder in folders_list:
    folder_path = './tiny-imagenet-200/train/' + folder + '/images/'
    images_list = os.listdir(folder_path)
    for image in images_list:
        image_path = folder_path + image
        rawImage = cv2.imread(image_path)
        rawImage_cpy = copy.copy(rawImage)
        height, width, channels = rawImage.shape
        size      = [height, width]
        patchSize = [10, 20]
        outputImage = generateMaskedImages(rawImage_cpy, size, noOfSeeds, patchSize, 1.5)
        name = output_masks + str(i) + '.jpg'
        cv2.imwrite(name, outputImage)
        name = output_images + str(i) + '.jpg'
        cv2.imwrite(name, rawImage)
        i += 1

'''




for folder in imageFolders:
	dataFiles  = os.path.join(inDataPath, folder, inputFileNameExt)
	fileNames  = glob(dataFiles)

	for inFile in fileNames: 
		rawImage = cv2.imread(inFile)  # reading images
		height, width, channels = rawImage.shape
		size      = [height, width]
		patchSize = [10, 20]
		
		outImage    = generateMaskedImages(rawImage, size, noOfSeeds, patchSize, 1.5)
		outFileName = inFile.split('/')[-1]
		outFile     = os.path.join(outDataPath, folders, outFileName)
		cv2.imwrite(outFile, outImage)
'''









