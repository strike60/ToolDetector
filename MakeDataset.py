#!/usr/bin/python3
#-*- coding:UTF-8 -*-
import numpy as np
import cv2
import sys
import argparse
import os

_ImageWidth = 1280
_ImageHeight = 720


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='./data',
    help='Directory to the data.')

parser.add_argument(
    '--destiny_dir', type=str, default='./Dataset',
    help='Directory to store the dataset')


#################################################
# Read the label.txt file and put the in the list
#################################################
def ReadLabels(labelpath):
    with open(labelpath, 'r') as f:
        labels = f.read()
    labels = np.array(list(labels), dtype=np.uint8)
    return labels


########################################################
# Crop and resize the image from (1280, 720) to (32, 32)
########################################################
def ImageCropAndResize(img):
    img = np.reshape(img, (_ImageHeight, _ImageWidth, 3))
    img = np.transpose(img, [1, 0, 2])
    img = img[300:_ImageWidth-300, 200:_ImageHeight-200, :]
    img = np.transpose(img, [1, 0, 2])
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    return img


##########################################################
# Read the images in image directory and put them in a list
##########################################################
def ReadImages(imagespath): 
    imgs = []
    imagesname = sorted(os.listdir(imagespath))
    for imgname in imagesname:
        imgpath = os.path.join(imagespath, imgname)
        img = cv2.imread(imgpath)
        img = ImageCropAndResize(img)
        imgs.append(img)
    return imgs


####################
# Write Dataset files
####################
def WriteDataset(datadir):
    if not os.path.exists(FLAGS.destiny_dir):
        os.makedirs(FLAGS.destiny_dir)
    dataindex = sorted(os.listdir(datadir))
    for index in dataindex:
        datapath = os.path.join(datadir, index)
        labelpath = os.path.join(datapath, 'label.txt')
        imagespath = os.path.join(datapath, 'image')
        labels = ReadLabels(labelpath)
        #print(labels)
        images = ReadImages(imagespath)
        print('Creating Dataset'+index+' ...')
        with open('./Dataset/dataset'+index, 'wb') as f:
            for i in range(len(labels)):
                f.write(labels[i])
                f.write(images[i])


def main():
    WriteDataset(FLAGS.data_dir)

if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main()
