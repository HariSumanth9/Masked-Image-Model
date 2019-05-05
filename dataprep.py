import os
import cv2
import copy
import random
import numpy as np
import matplotlib.image as mpimage

folders_path = './tiny-imagenet-200/train'
folders_list = os.listdir(folders_path)
if('.DS_Store' in folders_list):
    folders_list.remove('.DS_Store')
folders_list.sort()

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
        xSeed = random.randrange(finalPatchSizeSmall[0], imageSize[0]-finalPatchSizeSmall[0])
        ySeed = random.randrange(finalPatchSizeSmall[1], imageSize[1]-finalPatchSizeSmall[1])
        imageIn[xSeed:xSeed+finalPatchSizeSmall[0], ySeed:ySeed+finalPatchSizeSmall[1]]     = 0
        imageIn_cpy[xSeed:xSeed+finalPatchSizeLarge[0], ySeed:ySeed+finalPatchSizeLarge[1]] = 0
    return imageIn, imageIn_cpy

i = 0
f = 0
s = 0
t = 0
labels_100 = []
labels_70  = []
labels_30  = []
for folder in folders_list:
    images_list = os.listdir(folders_path + '/' + folder + '/images')
    if('.DS_Store' in images_list):
        images_list.remove('.DS_Store')
    images_list_70 = images_list[:350]
    images_list_30 = images_list[350:]
    for image_name in images_list_70:
        image      = cv2.imread(folders_path + '/' + folder + '/images/' + image_name)
        image_cpy  = copy.copy(image)
        small_mask, large_mask = generateMaskedImages(image_cpy, [64, 64], 2, [4, 7], [7, 14], 1.5)
        labels_100.append(i)
        labels_70.append(i )
        name = 'Dataset/Classifier/Train_100/Images/image' + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/Classifier/Train_70/Images/image'  + '_{0:05d}'.format(s) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Train_100/Images/image'     + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Train_70/Images/image'      + '_{0:05d}'.format(s) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Train_100/Masks/Small/mask' + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, small_mask)
        name = 'Dataset/EncDec/Train_70/Masks/Small/mask'  + '_{0:05d}'.format(s) + '.jpg'
        cv2.imwrite(name, small_mask)
        name = 'Dataset/EncDec/Train_100/Masks/Large/mask' + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, large_mask)
        name = 'Dataset/EncDec/Train_70/Masks/Large/mask'  + '_{0:05d}'.format(s) + '.jpg'
        cv2.imwrite(name, large_mask)
        f += 1
        s += 1


    for image_name in images_list_30:
        image      = cv2.imread(folders_path + '/' + folder + '/images/' + image_name)
        image_cpy  = copy.copy(image)
        small_mask, large_mask = generateMaskedImages(image_cpy, [64, 64], 2, [4, 7], [7, 14], 1.5)
        labels_100.append(i)
        labels_30.append(i )
        name = 'Dataset/Classifier/Train_100/Images/image' + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/Classifier/Train_30/Images/image'  + '_{0:05d}'.format(t) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Train_100/Images/image'     + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Train_30/Images/image'      + '_{0:05d}'.format(t) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Train_100/Masks/Small/mask' + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, small_mask)
        name = 'Dataset/EncDec/Train_30/Masks/Small/mask'  + '_{0:05d}'.format(t) + '.jpg'
        cv2.imwrite(name, small_mask)
        name = 'Dataset/EncDec/Train_100/Masks/Large/mask' + '_{0:05d}'.format(f) + '.jpg'
        cv2.imwrite(name, large_mask)
        name = 'Dataset/EncDec/Train_30/Masks/Large/mask'  + '_{0:05d}'.format(t) + '.jpg'
        cv2.imwrite(name, large_mask)
        f += 1
        t += 1
    i += 1


np.save('./Dataset/train_labels_100.npy', labels_100)
np.save('./Dataset/train_labels_70.npy', labels_70)
np.save('./Dataset/train_labels_30.npy', labels_30)

valid_images_path = './tiny-imagenet-200/val/images/'
labels_file       = open('./tiny-imagenet-200/val/val_annotations.txt', 'r')

text = labels_file.read()
labels_file.close()
text = text.split()
i = 1
n = len(text)
Dict = {}
while(1):
    Dict[text[i-1]] = text[i]
    i += 6
    if(i > n):
        break

valid_labels = []
i = 0
for image_name in Dict:
    label_name = Dict[image_name]
    label      = folders_list.index(label_name)
    image      = cv2.imread(valid_images_path + image_name)
    if(len(image.shape) == 3):
        print(image.shape)
        image_cpy  = copy.copy(image)
        valid_labels.append(label)
        small_mask, large_mask = generateMaskedImages(image_cpy, [64, 64], 2, [4, 7], [7, 14], 1.5)
        name = 'Dataset/Classifier/Valid/Images/image' + '_{0:05d}'.format(i) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Valid/Images/image'     + '_{0:05d}'.format(i) + '.jpg'
        cv2.imwrite(name, image)
        name = 'Dataset/EncDec/Valid/Masks/Small/mask' + '_{0:05d}'.format(i) + '.jpg'
        cv2.imwrite(name, small_mask)
        name = 'Dataset/EncDec/Valid/Masks/Large/mask' + '_{0:05d}'.format(i) + '.jpg'
        cv2.imwrite(name, large_mask)
        i += 1
np.save('./Dataset/valid_labels.npy', valid_labels)


