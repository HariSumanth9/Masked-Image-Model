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


def generateMaskedImages(imageIn, imageSize, noOfSeeds, patchSizeSmall, patchSizeLarge, patchMultiplier):
    imageIn_cpy = copy.copy(imageIn)
    for i in range(noOfSeeds):
        finalPatchSizeSmall    = copy.copy(patchSizeSmall)
        finalPatchSizeLarge    = copy.copy(patchSizeLarge)
        finalPatchSizeSmall[0] = int(np.floor(patchSizeSmall[0]*patchMultiplier))
        finalPatchSizeSmall[1] = int(np.floor(patchSizeSmall[1]*patchMultiplier))
        finalPatchSizeLarge    = copy.copy(patchSizeLarge)
        finalPatchSizeLarge[0] = int(np.floor(patchSizeLarge[0]*patchMultiplier))
        finalPatchSizeLarge[1] = int(np.floor(patchSizeLarge[1]*patchMultiplier))
        #print(finalPatchSize)
        xSeed = random.randrange(finalPatchSizeSmall[0], imageSize[0]-finalPatchSizeSmall[0])
        ySeed = random.randrange(finalPatchSizeSmall[1], imageSize[1]-finalPatchSizeSmall[1])
        #xSeedLarge = random.randrange(finalPatchSizeLarge[0], imageSize[0]-finalPatchSizeLarge[0])
        #ySeedLarge = random.randrange(finalPatchSizeLarge[1], imageSize[1]-finalPatchSizeLarge[1])
        #print(xSeed, ySeed)
        imageIn[xSeed:xSeed+finalPatchSizeSmall[0], ySeed:ySeed+finalPatchSizeSmall[1]] = 0
        imageIn_cpy[xSeed:xSeed+finalPatchSizeLarge[0], ySeed:ySeed+finalPatchSizeLarge[1]] = 0
    return imageIn, imageIn_cpy


folders_list = os.listdir('./tiny-imagenet-200/train')
output_images = './tiny_imagenet/train/orig/'
output_masks_small  = './tiny_imagenet/train/masks_small/'
output_masks_large  = './tiny_imagenet/train/masks_large/'
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
        patchSizeSmall = [4, 7]
        patchSizeLarge = [7, 14]
        outputImageSmall, outputImageLarge = generateMaskedImages(rawImage_cpy, size, noOfSeeds, patchSizeSmall, patchSizeLarge, 1.5)
        name = output_masks_small + str(i) + '.jpg'
        cv2.imwrite(name, outputImageSmall)
        name = output_masks_large + str(i) + '.jpg'
        cv2.imwrite(name, outputImageLarge)
        name = output_images + str(i) + '.jpg'
        cv2.imwrite(name, rawImage)
        i += 1


#folders_list = os.listdir('./tiny-imagenet-200/t')
output_images = './tiny_imagenet/valid/orig/'
output_masks_small  = './tiny_imagenet/valid/masks_small/'
output_masks_large  = './tiny_imagenet/valid/masks_large/'
i = 0
folder_path = './tiny-imagenet-200/val/images/'
images_list = os.listdir(folder_path)
for image in images_list:
    image_path = folder_path + image
    rawImage = cv2.imread(image_path)
    rawImage_cpy = copy.copy(rawImage)
    height, width, channels = rawImage.shape
    size      = [height, width]
    patchSizeSmall = [4, 7]
    patchSizeLarge = [7, 14]
    outputImageSmall, outputImageLarge = generateMaskedImages(rawImage_cpy, size, noOfSeeds, patchSizeSmall, patchSizeLarge, 1.5)
    name = output_masks_small + str(i) + '.jpg'
    cv2.imwrite(name, outputImageSmall)
    name = output_masks_large + str(i) + '.jpg'
    cv2.imwrite(name, outputImageLarge)
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









